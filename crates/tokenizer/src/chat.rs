//! The model's own chat template, rendered with minijinja.
//!
//! GGUF carries the Jinja source verbatim from the Hugging Face repo, so the
//! prompt we build is byte-identical to what the model was tuned on. That
//! matters more than it sounds: a missing `<|im_start|>` costs real quality.

use anyhow::{Context, Result};
use minijinja::{Environment, Value as JValue, context};

const TEMPLATE_NAME: &str = "chat";

/// One turn of a conversation.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
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
        let tmpl = self
            .env
            .get_template(TEMPLATE_NAME)
            .context("chat template disappeared")?;

        let tools = match tools {
            Some(v) => JValue::from_serialize(v),
            None => JValue::from(()),
        };

        tmpl.render(context! {
            messages => JValue::from_serialize(messages),
            add_generation_prompt => add_generation_prompt,
            tools => tools,
            bos_token => self.bos_token.as_str(),
            eos_token => self.eos_token.as_str(),
        })
        .context("rendering the chat template")
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
