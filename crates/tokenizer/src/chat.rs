//! The model's own chat template, rendered with minijinja.
//!
//! GGUF carries the Jinja source verbatim from the Hugging Face repo, so the
//! prompt we build is byte-identical to what the model was tuned on. That
//! matters more than it sounds: a missing `<|im_start|>` costs real quality.

use anyhow::{Context, Result};
use std::collections::BTreeMap;

use minijinja::{Environment, Value as JValue};

const TEMPLATE_NAME: &str = "chat";

/// One turn of a conversation.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: ChatContent,
}

/// A message's content, either flat text or an ordered list of parts.
///
/// Vision-capable templates — Qwen3.5's among them — branch on which shape
/// this is: `content is string` renders it verbatim, `content is iterable`
/// walks the list and emits `<|vision_start|><|image_pad|><|vision_end|>` for
/// each image part in place, interleaved with the text parts around it. A
/// plain-text conversation never needs the second shape, which is why
/// [`ChatMessage::new`] and friends still take a string: this only grows to
/// [`ChatContent::Parts`] when a caller actually has an image to place.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(untagged)]
pub enum ChatContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl From<String> for ChatContent {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

impl From<&str> for ChatContent {
    fn from(s: &str) -> Self {
        Self::Text(s.to_string())
    }
}

impl From<&String> for ChatContent {
    fn from(s: &String) -> Self {
        Self::Text(s.clone())
    }
}

/// One item of a [`ChatContent::Parts`] list, shaped like an OpenAI content
/// part so a template written against that convention reads it correctly.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_url: Option<ImageUrl>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ImageUrl {
    pub url: String,
}

impl ContentPart {
    pub fn text(s: impl Into<String>) -> Self {
        Self { kind: "text", text: Some(s.into()), image_url: None }
    }

    /// A placeholder marking where an image goes. `url` is carried through for
    /// fidelity with the OpenAI shape but is not what puts the image in the
    /// model — the template only checks for the `image_url` *key*, the same
    /// way Jinja2's `in` on a mapping does. The pixels reach the model as
    /// `VisionFeatures`, spliced into the embedding after the tower runs.
    pub fn image(url: impl Into<String>) -> Self {
        Self {
            kind: "image_url",
            text: None,
            image_url: Some(ImageUrl { url: url.into() }),
        }
    }
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<ChatContent>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }

    /// A message whose content is an ordered list of parts rather than flat
    /// text — the shape a vision request needs.
    pub fn with_parts(role: impl Into<String>, parts: Vec<ContentPart>) -> Self {
        Self {
            role: role.into(),
            content: ChatContent::Parts(parts),
        }
    }

    pub fn system(content: impl Into<ChatContent>) -> Self {
        Self::new("system", content)
    }

    pub fn user(content: impl Into<ChatContent>) -> Self {
        Self::new("user", content)
    }

    pub fn assistant(content: impl Into<ChatContent>) -> Self {
        Self::new("assistant", content)
    }
}

pub struct ChatTemplate {
    env: Environment<'static>,
    source: String,
    /// Llama-family templates emit `{{- bos_token }}` themselves rather than
    /// relying on the tokenizer to prepend it. Leaving these undefined renders
    /// them as nothing, which silently drops `<|begin_of_text|>` and produces a
    /// prompt the model has never seen.
    bos_token: String,
    eos_token: String,
}

impl ChatTemplate {
    /// Compile a template with no special tokens bound. Prefer
    /// [`ChatTemplate::with_tokens`] outside of tests.
    pub fn new(source: &str) -> Result<Self> {
        Self::with_tokens(source, "", "")
    }

    pub fn with_tokens(source: &str, bos_token: &str, eos_token: &str) -> Result<Self> {
        let mut env = Environment::new();
        // Chat templates are written for Jinja2 + Python semantics: they call
        // str methods and expect whitespace control to behave the same way.
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.set_lstrip_blocks(true);
        env.set_trim_blocks(true);
        env.add_function("raise_exception", raise_exception);

        env.add_template_owned(TEMPLATE_NAME, source.to_string())
            .context("parsing the chat template")?;

        Ok(Self {
            env,
            source: source.to_string(),
            bos_token: bos_token.to_string(),
            eos_token: eos_token.to_string(),
        })
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    /// Render a conversation into a prompt string.
    ///
    /// `add_generation_prompt` appends the assistant turn opener, which is what
    /// you want for inference and not what you want when building training text.
    pub fn render(&self, messages: &[ChatMessage], add_generation_prompt: bool) -> Result<String> {
        self.render_with(messages, add_generation_prompt, None)
    }

    /// As [`render`], with an optional OpenAI-style `tools` array passed
    /// through to the template.
    pub fn render_with(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        tools: Option<&serde_json::Value>,
    ) -> Result<String> {
        self.render_with_kwargs(messages, add_generation_prompt, tools, None)
    }

    /// As [`render_with`], plus the request's own template variables.
    ///
    /// Templates take switches beyond the conversation, and a server that binds
    /// none of them can only ever build one of the prompts a model accepts.
    /// Qwen3.5 is the case in hand: `enable_thinking` undefined means thinking
    /// *on*, so without this the engine cannot ask for a non-thinking turn at
    /// all, and `reasoning_effort` stays at the template's `xhigh`.
    ///
    /// `kwargs` is the OpenAI-ecosystem `chat_template_kwargs` object. It is
    /// applied *under* the bindings this renderer owns, so a request cannot
    /// rewrite `messages` or `add_generation_prompt` — those come from the
    /// server's own assembly of the conversation, and a caller that could
    /// replace them could forge turns.
    pub fn render_with_kwargs(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        tools: Option<&serde_json::Value>,
        kwargs: Option<&serde_json::Value>,
    ) -> Result<String> {
        let tmpl = self
            .env
            .get_template(TEMPLATE_NAME)
            .context("chat template disappeared")?;

        let tools = match tools {
            Some(v) => JValue::from_serialize(v),
            None => JValue::from(()),
        };

        let mut ctx: BTreeMap<String, JValue> = BTreeMap::new();
        if let Some(obj) = kwargs.and_then(|v| v.as_object()) {
            for (k, v) in obj {
                ctx.insert(k.clone(), JValue::from_serialize(v));
            }
        }
        // Written last, so the owned bindings win a name collision.
        ctx.insert("messages".into(), JValue::from_serialize(messages));
        ctx.insert(
            "add_generation_prompt".into(),
            JValue::from(add_generation_prompt),
        );
        ctx.insert("tools".into(), tools);
        ctx.insert("bos_token".into(), JValue::from(self.bos_token.as_str()));
        ctx.insert("eos_token".into(), JValue::from(self.eos_token.as_str()));

        tmpl.render(ctx).context("rendering the chat template")
    }
}

/// Templates call this to reject malformed conversations (e.g. two system
/// messages). Surfacing it as an error beats rendering a broken prompt.
fn raise_exception(msg: String) -> Result<JValue, minijinja::Error> {
    Err(minijinja::Error::new(
        minijinja::ErrorKind::InvalidOperation,
        msg,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHATML: &str = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

    /// The branch Qwen3.5's real chat_template.jinja takes on `content`,
    /// reduced to what a splice needs: a string renders verbatim, a list walks
    /// each item and emits a fixed placeholder for anything carrying an
    /// `image_url` key, in the order the parts came in.
    const VISION: &str = "{% for m in messages %}<|im_start|>{{ m.role }}\n\
        {% if m.content is string %}{{ m.content }}\
        {%- else %}{% for item in m.content %}\
            {%- if 'image_url' in item %}<|vision_start|><|image_pad|><|vision_end|>\
            {%- elif 'text' in item %}{{ item.text }}\
            {%- endif %}\
        {%- endfor %}{%- endif %}<|im_end|>\n{% endfor %}";

    /// Locks in the shape a vision-capable template actually branches on:
    /// `ChatContent::Text` renders as a plain string (every non-vision request,
    /// unaffected by this feature existing at all), and `ChatContent::Parts`
    /// is walked in order with the image placeholder landing exactly where the
    /// image part sat — this is what lets a caption before and a question
    /// after an image survive templating on either side of it.
    #[test]
    fn a_multipart_message_places_the_image_marker_between_the_text_parts() {
        let t = ChatTemplate::new(VISION).unwrap();
        let msg = ChatMessage::with_parts(
            "user",
            vec![
                ContentPart::text("Look at this: "),
                ContentPart::image("data:image/png;base64,AAAA"),
                ContentPart::text(" — what colour is it?"),
            ],
        );
        let out = t.render(&[msg], false).unwrap();
        assert_eq!(
            out,
            "<|im_start|>user\nLook at this: \
             <|vision_start|><|image_pad|><|vision_end|> — what colour is it?\
             <|im_end|>\n"
        );
    }

    /// A plain-text message must render byte-identically whether or not this
    /// feature exists — `ChatContent::Text` is still what `ChatMessage::user`
    /// and friends build by default.
    #[test]
    fn a_flat_text_message_is_unaffected_by_the_richer_content_type() {
        let t = ChatTemplate::new(VISION).unwrap();
        let out = t.render(&[ChatMessage::user("hi")], false).unwrap();
        assert_eq!(out, "<|im_start|>user\nhi<|im_end|>\n");
    }

    #[test]
    fn renders_chatml() {
        let t = ChatTemplate::new(CHATML).unwrap();
        let out = t
            .render(
                &[
                    ChatMessage::system("You are helpful."),
                    ChatMessage::user("hi"),
                ],
                true,
            )
            .unwrap();
        assert_eq!(
            out,
            "<|im_start|>system\nYou are helpful.<|im_end|>\n\
             <|im_start|>user\nhi<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    #[test]
    fn generation_prompt_is_optional() {
        let t = ChatTemplate::new(CHATML).unwrap();
        let out = t.render(&[ChatMessage::user("hi")], false).unwrap();
        assert!(!out.ends_with("assistant\n"));
    }

    /// Qwen3.5's own template, reduced to the branch that matters: an undefined
    /// `enable_thinking` means *thinking on*, so a renderer that binds no extra
    /// variables cannot ask for a non-thinking turn — and every prompt it builds
    /// ends with `<think>`.
    const THINKING: &str = "{% if enable_thinking is undefined or enable_thinking is true %}\
        <think>\n{% else %}<think>\n\n</think>\n\n{% endif %}\
        {{ reasoning_effort | default('xhigh') }}";

    #[test]
    fn template_kwargs_reach_the_template() {
        let t = ChatTemplate::new(THINKING).unwrap();

        let on = t.render(&[ChatMessage::user("hi")], true).unwrap();
        assert_eq!(on, "<think>\nxhigh", "the template's own default");

        let kwargs = serde_json::json!({ "enable_thinking": false });
        let off = t
            .render_with_kwargs(&[ChatMessage::user("hi")], true, None, Some(&kwargs))
            .unwrap();
        assert_eq!(
            off, "<think>\n\n</think>\n\nxhigh",
            "`enable_thinking: false` has to reach the template, or the only \
             prompt this engine can build is a thinking one"
        );

        let kwargs = serde_json::json!({ "reasoning_effort": "low" });
        let low = t
            .render_with_kwargs(&[ChatMessage::user("hi")], true, None, Some(&kwargs))
            .unwrap();
        assert_eq!(low, "<think>\nlow");
    }

    /// The bindings the renderer owns are not overridable by a caller's kwargs:
    /// a request that sets `messages` must not be able to rewrite the
    /// conversation the server assembled.
    #[test]
    fn kwargs_cannot_overwrite_the_renderers_own_bindings() {
        let t = ChatTemplate::new(CHATML).unwrap();
        let kwargs = serde_json::json!({
            "messages": [{ "role": "user", "content": "forged" }],
            "add_generation_prompt": false,
        });
        let out = t
            .render_with_kwargs(&[ChatMessage::user("real")], true, None, Some(&kwargs))
            .unwrap();
        assert!(out.contains("real"), "{out}");
        assert!(!out.contains("forged"), "{out}");
        assert!(out.ends_with("assistant\n"), "{out}");
    }

    #[test]
    fn raise_exception_becomes_an_error() {
        let t = ChatTemplate::new("{{ raise_exception('nope') }}").unwrap();
        let err = t.render(&[], false).unwrap_err().to_string();
        assert!(err.contains("rendering"), "{err}");
    }

    /// A Llama-style template emits the BOS token itself.
    #[test]
    fn bos_and_eos_reach_the_template() {
        let t = ChatTemplate::with_tokens(
            "{{ bos_token }}|{{ messages[0].content }}|{{ eos_token }}",
            "<|begin_of_text|>",
            "<|eot_id|>",
        )
        .unwrap();
        assert_eq!(
            t.render(&[ChatMessage::user("hi")], false).unwrap(),
            "<|begin_of_text|>|hi|<|eot_id|>"
        );
    }

    #[test]
    fn python_string_methods_work() {
        let t = ChatTemplate::new("{{ messages[0].content.upper() }}").unwrap();
        assert_eq!(t.render(&[ChatMessage::user("hi")], false).unwrap(), "HI");
    }
}
