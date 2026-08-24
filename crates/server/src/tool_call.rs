//! Parsing Qwen3.5's tool-call format out of generated text.
//!
//! The model was told, in its own system prompt, to answer a tool call like
//! this — see the `<IMPORTANT>` block `chat_template.jinja` injects:
//!
//! ```text
//! <tool_call>
//! <function=example_function_name>
//! <parameter=example_parameter_1>
//! value_1
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! Nested tags, not JSON — this is Qwen3.5's own format, and it differs from
//! Qwen3's (`<tool_call>{"name": ..., "arguments": {...}}</tool_call>`) and
//! from Llama's (header-based turns, no tags in content at all). A parser
//! written for one does not read another; this one is for this template only.
//!
//! Parameter values arrive as raw text, not JSON — the template's own
//! instructions say a value "can span multiple lines" — so turning them into
//! OpenAI's typed `function.arguments` needs the tool's own declared schema:
//! an `"integer"` parameter's `"5"` has to become the number `5`, and a
//! `"string"` parameter's `"5"` has to stay `"5"`. Guessing generically (try
//! parsing as a number, fall back to a string) gets exactly the cases wrong
//! that a schema exists to state: an account number or a zip code that looks
//! numeric and is not one.

use std::collections::HashMap;

use serde_json::Value;

pub struct ParsedToolCall {
    pub name: String,
    /// A JSON-encoded object, ready for OpenAI's `function.arguments` (which
    /// the spec defines as a JSON *string*, not a nested object).
    pub arguments: String,
}

/// The result of scanning one assistant turn for tool calls.
pub struct ToolCallScan {
    /// Text before the first `<tool_call>`, trimmed. Empty when the turn is
    /// nothing but tool calls, which the template's own instructions call for
    /// ("NOT after" a function call) but do not forbid before.
    pub leading_text: String,
    pub calls: Vec<ParsedToolCall>,
    /// A `<tool_call>` was opened but never closed — generation ran out of
    /// budget mid-block, most likely. The caller should not report
    /// `finish_reason: "tool_calls"` for a scan with this set: the arguments
    /// this would have produced are a guess at best, and OpenAI's own clients
    /// treat that finish reason as a promise the call is complete.
    pub truncated: bool,
}

/// Per-parameter type, read from a tool's JSON Schema. Only the JSON Schema
/// primitive types that change how a raw string is parsed; anything else
/// (`"string"`, absent, unrecognised) keeps the value as a JSON string, which
/// is also the only sound default — a schema this parser does not understand
/// is not licence to guess at one it does.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ParamType {
    String,
    Integer,
    Number,
    Boolean,
    /// Array or object: the raw text is itself expected to be JSON.
    Json,
}

/// Read every function's parameter types out of an OpenAI-shaped `tools`
/// array, keyed by `"function_name.param_name"`.
///
/// Built once per request rather than searched per parameter: a tool call
/// scan can produce many parameters across many calls, and `tools` is small
/// and request-scoped.
fn param_types(tools: &[Value]) -> HashMap<(String, String), ParamType> {
    let mut out = HashMap::new();
    for t in tools {
        let f = &t["function"];
        let Some(name) = f["name"].as_str() else { continue };
        let Some(props) = f["parameters"]["properties"].as_object() else { continue };
        for (pname, schema) in props {
            let ty = match schema["type"].as_str() {
                Some("integer") => ParamType::Integer,
                Some("number") => ParamType::Number,
                Some("boolean") => ParamType::Boolean,
                Some("array") | Some("object") => ParamType::Json,
                _ => ParamType::String,
            };
            out.insert((name.to_string(), pname.clone()), ty);
        }
    }
    out
}

fn coerce(ty: ParamType, raw: &str) -> Value {
    match ty {
        ParamType::String => Value::String(raw.to_string()),
        ParamType::Integer => raw
            .parse::<i64>()
            .map(Value::from)
            .unwrap_or_else(|_| Value::String(raw.to_string())),
        ParamType::Number => raw
            .parse::<f64>()
            .ok()
            .and_then(serde_json::Number::from_f64)
            .map(Value::Number)
            .unwrap_or_else(|| Value::String(raw.to_string())),
        // Case-insensitive: measured against the real 27B, which wrote a
        // boolean parameter as Python-style `True` rather than JSON's lower-
        // case `true`. The model was told a natural-language description of
        // the parameter, not the literal spelling JSON wants, so treating
        // this as the model's mistake to fix rather than a schema violation
        // is the right call — the alternative is `"detailed": "True"` staying
        // a string, which every consumer of this parameter would get wrong.
        ParamType::Boolean => match raw.trim().to_ascii_lowercase().as_str() {
            "true" => Value::Bool(true),
            "false" => Value::Bool(false),
            _ => Value::String(raw.to_string()),
        },
        // A model that emits invalid JSON for an array/object parameter has
        // made a mistake no type coercion can recover from; falling back to
        // the raw string at least survives it rather than dropping the call.
        ParamType::Json => serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string())),
    }
}

/// One `<function=...>...</function>` block's contents, read past its
/// `<function=` opener. Returns the byte offset just past a well-formed
/// block's `</function>`, or `None` if the block never closes — the caller's
/// signal to treat everything from `<tool_call>` onward as truncated.
fn parse_function(rest: &str) -> Option<(String, Vec<(String, String)>, usize)> {
    let name_end = rest.find('>')?;
    let name = rest[..name_end].to_string();
    let mut pos = name_end + 1;
    let mut params = Vec::new();
    loop {
        let tail = &rest[pos..];
        if let Some(close) = tail.find("</function>") {
            // Nothing left but whitespace before the close: no more params.
            if tail[..close].trim().is_empty() {
                return Some((name, params, pos + close + "</function>".len()));
            }
        }
        let Some(p_start) = tail.find("<parameter=") else {
            return None;
        };
        // A non-whitespace stray between params/the last close is not this
        // format; refuse to guess past it rather than silently skipping it.
        if !tail[..p_start].trim().is_empty() {
            return None;
        }
        let after_tag = p_start + "<parameter=".len();
        let name_end = tail[after_tag..].find('>')? + after_tag;
        let pname = tail[after_tag..name_end].to_string();
        let value_start = name_end + 1;
        let close_rel = tail[value_start..].find("</parameter>")?;
        let value = tail[value_start..value_start + close_rel].trim().to_string();
        params.push((pname, value));
        pos += value_start + close_rel + "</parameter>".len();
    }
}

/// Scan one assistant turn's generated text for `<tool_call>` blocks.
///
/// `tools` is the request's own tool list, read only for parameter types —
/// see the module note. An empty slice is fine; every parameter then stays a
/// JSON string, which is correct, just untyped.
pub fn scan(text: &str, tools: &[Value]) -> ToolCallScan {
    let types = param_types(tools);
    let mut calls = Vec::new();
    let leading_end = text.find("<tool_call>").unwrap_or(text.len());
    let leading_text = text[..leading_end].trim().to_string();
    let mut cursor = leading_end;

    while let Some(rel) = text[cursor..].find("<tool_call>") {
        let start = cursor + rel;
        let body_start = start + "<tool_call>".len();
        let body = &text[body_start..];
        let Some(func_rel) = body.find("<function=") else {
            return ToolCallScan { leading_text, calls, truncated: true };
        };
        if !body[..func_rel].trim().is_empty() {
            // Text between `<tool_call>` and `<function=` that is not
            // whitespace is not this format; stop rather than guess.
            return ToolCallScan { leading_text, calls, truncated: true };
        }
        let after_open = func_rel + "<function=".len();
        let Some((name, params, func_end)) = parse_function(&body[after_open..]) else {
            return ToolCallScan { leading_text, calls, truncated: true };
        };
        let after_func = &body[after_open + func_end..];
        let Some(close_rel) = after_func.find("</tool_call>") else {
            return ToolCallScan { leading_text, calls, truncated: true };
        };
        if !after_func[..close_rel].trim().is_empty() {
            return ToolCallScan { leading_text, calls, truncated: true };
        }

        let mut obj = serde_json::Map::new();
        for (pname, raw) in params {
            let ty = types
                .get(&(name.clone(), pname.clone()))
                .copied()
                .unwrap_or(ParamType::String);
            obj.insert(pname, coerce(ty, &raw));
        }
        calls.push(ParsedToolCall {
            name,
            arguments: serde_json::to_string(&Value::Object(obj)).expect("a map serializes"),
        });

        cursor = body_start + after_open + func_end + close_rel + "</tool_call>".len();
    }
    ToolCallScan { leading_text, calls, truncated: false }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool(name: &str, props: Value) -> Value {
        serde_json::json!({
            "type": "function",
            "function": { "name": name, "parameters": { "type": "object", "properties": props } }
        })
    }

    #[test]
    fn a_single_call_with_one_string_parameter() {
        let text = "<tool_call>\n<function=get_weather>\n<parameter=city>\nBeijing\n</parameter>\n</function>\n</tool_call>";
        let scan_result = scan(text, &[]);
        assert!(!scan_result.truncated);
        assert_eq!(scan_result.leading_text, "");
        assert_eq!(scan_result.calls.len(), 1);
        assert_eq!(scan_result.calls[0].name, "get_weather");
        assert_eq!(scan_result.calls[0].arguments, r#"{"city":"Beijing"}"#);
    }

    #[test]
    fn natural_language_reasoning_before_the_call_is_kept_as_leading_text() {
        let text = "Let me check the weather for you.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nBeijing\n</parameter>\n</function>\n</tool_call>";
        let s = scan(text, &[]);
        assert_eq!(s.leading_text, "Let me check the weather for you.");
        assert_eq!(s.calls.len(), 1);
    }

    #[test]
    fn a_multiline_value_is_kept_whole() {
        let text = "<tool_call>\n<function=write_file>\n<parameter=content>\nline one\nline two\nline three\n</parameter>\n</function>\n</tool_call>";
        let s = scan(text, &[]);
        let v: Value = serde_json::from_str(&s.calls[0].arguments).unwrap();
        assert_eq!(v["content"], "line one\nline two\nline three");
    }

    #[test]
    fn two_tool_calls_in_one_turn_both_parse() {
        let text = "<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=b>\n<parameter=y>\n2\n</parameter>\n</function>\n</tool_call>";
        let s = scan(text, &[]);
        assert_eq!(s.calls.len(), 2);
        assert_eq!(s.calls[0].name, "a");
        assert_eq!(s.calls[1].name, "b");
    }

    #[test]
    fn parameters_are_coerced_against_the_tools_declared_schema() {
        let tools = vec![tool(
            "book_flight",
            serde_json::json!({
                "passengers": {"type": "integer"},
                "price": {"type": "number"},
                "refundable": {"type": "boolean"},
                "extras": {"type": "array"},
                "reference": {"type": "string"},
            }),
        )];
        let text = "<tool_call>\n<function=book_flight>\n\
            <parameter=passengers>\n3\n</parameter>\n\
            <parameter=price>\n199.5\n</parameter>\n\
            <parameter=refundable>\ntrue\n</parameter>\n\
            <parameter=extras>\n[\"bag\",\"meal\"]\n</parameter>\n\
            <parameter=reference>\n00921\n</parameter>\n\
            </function>\n</tool_call>";
        let s = scan(text, &tools);
        let v: Value = serde_json::from_str(&s.calls[0].arguments).unwrap();
        assert_eq!(v["passengers"], 3);
        assert_eq!(v["price"], 199.5);
        assert_eq!(v["refundable"], true);
        assert_eq!(v["extras"], serde_json::json!(["bag", "meal"]));
        // The reason a schema is consulted at all rather than guessing: this
        // looks numeric and the schema says it is a string, so it must stay
        // one. A generic "try a number, else a string" heuristic gets this
        // one wrong every time.
        assert_eq!(v["reference"], "00921");
    }

    /// Found against the real 27B: it wrote a boolean parameter as `True`,
    /// Python's capitalisation, not JSON's `true`.
    #[test]
    fn a_capitalised_boolean_still_coerces() {
        let tools = vec![tool("f", serde_json::json!({"detailed": {"type": "boolean"}}))];
        for (raw, want) in [("True", true), ("False", false), ("TRUE", true)] {
            let text = format!(
                "<tool_call>\n<function=f>\n<parameter=detailed>\n{raw}\n</parameter>\n</function>\n</tool_call>"
            );
            let s = scan(&text, &tools);
            let v: Value = serde_json::from_str(&s.calls[0].arguments).unwrap();
            assert_eq!(v["detailed"], want, "raw={raw:?}");
        }
    }

    #[test]
    fn an_unparseable_typed_value_falls_back_to_a_string_rather_than_vanishing() {
        let tools = vec![tool("f", serde_json::json!({"n": {"type": "integer"}}))];
        let text = "<tool_call>\n<function=f>\n<parameter=n>\nnot-a-number\n</parameter>\n</function>\n</tool_call>";
        let s = scan(text, &tools);
        let v: Value = serde_json::from_str(&s.calls[0].arguments).unwrap();
        assert_eq!(v["n"], "not-a-number");
    }

    #[test]
    fn a_call_with_no_parameters_still_parses() {
        let text = "<tool_call>\n<function=ping>\n</function>\n</tool_call>";
        let s = scan(text, &[]);
        assert_eq!(s.calls.len(), 1);
        assert_eq!(s.calls[0].arguments, "{}");
    }

    #[test]
    fn plain_text_with_no_tool_call_is_untouched() {
        let s = scan("just an ordinary answer", &[]);
        assert_eq!(s.leading_text, "just an ordinary answer");
        assert!(s.calls.is_empty());
        assert!(!s.truncated);
    }

    #[test]
    fn a_call_cut_off_by_the_token_budget_is_marked_truncated_not_parsed() {
        // `max_tokens` can land anywhere; this is what it looks like stopped
        // mid-parameter, with neither `</parameter>` nor `</function>` ever
        // written.
        let text = "<tool_call>\n<function=get_weather>\n<parameter=city>\nBei";
        let s = scan(text, &[]);
        assert!(s.truncated, "an unclosed block must not report a parsed call");
        assert!(s.calls.is_empty());
    }

    #[test]
    fn a_truncated_second_call_does_not_discard_the_first() {
        let text = "<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=b>\n<parameter=y>\nunfinished";
        let s = scan(text, &[]);
        assert!(s.truncated);
        assert_eq!(s.calls.len(), 1);
        assert_eq!(s.calls[0].name, "a");
    }
}
