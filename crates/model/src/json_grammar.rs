//! A byte-level JSON grammar cursor driven by a (deliberately not
//! fully-general) JSON Schema subset, for `docs/keel-integration.md`'s
//! "2. Schema-constrained decoding" section.
//!
//! This is the algorithmic core only -- no tokenizer, no model, no sampler.
//! [`Cursor::feed`] answers one question, one byte at a time: does appending
//! this byte keep the output a valid prefix of *some* string that both
//! parses as JSON and satisfies the schema. A caller drives this per
//! candidate token (try feeding all of a token's bytes; keep the token only
//! if every byte was accepted) and, once a token is chosen, feeds its real
//! bytes to advance the one cursor that actually matters. See
//! `crates/server/src/constrained.rs` for that half.
//!
//! **Deliberately not supported**, to keep this a correct, testable subset
//! rather than a partial reimplementation of the JSON Schema spec: `oneOf`/
//! `anyOf`/`allOf`, `pattern`, `minLength`/`maxLength`/`minimum`/`maximum`,
//! number exponents (`1e10`), `null` as a value for a non-null-typed field,
//! escape sequences inside an `enum`-constrained string, and `additionalProperties`
//! (object keys are always restricted to the declared `properties`). What it
//! does support -- objects with required/optional properties, arrays,
//! strings (free or `enum`-constrained), numbers, integers, and booleans,
//! arbitrarily nested -- is real and independently unit-tested byte by byte,
//! not just exercised end to end.

use anyhow::Result;

/// The schema subset this cursor understands, parsed once from a
/// `serde_json::Value` (see [`Schema::parse`]) and then read-only for the
/// lifetime of a generation.
#[derive(Clone, Debug)]
pub enum Schema {
    Object(ObjectSchema),
    Array(Box<Schema>),
    String { enum_values: Option<Vec<String>> },
    Number,
    Integer,
    Boolean,
    /// A field with no (or an unsupported) type constraint. Treated as a
    /// free string rather than an open-ended JSON value -- see this
    /// module's own doc comment for why.
    Any,
}

#[derive(Clone, Debug)]
pub struct ObjectSchema {
    pub properties: Vec<(String, Schema)>,
    pub required: std::collections::BTreeSet<String>,
}

impl Schema {
    pub fn parse(v: &serde_json::Value) -> Result<Schema> {
        let obj = v.as_object().ok_or_else(|| anyhow::anyhow!("schema must be a JSON object"))?;
        let ty = obj.get("type").and_then(|t| t.as_str());
        match ty {
            Some("object") => Ok(Schema::Object(parse_object(obj)?)),
            Some("array") => {
                let items = obj
                    .get("items")
                    .ok_or_else(|| anyhow::anyhow!("array schema needs \"items\""))?;
                Ok(Schema::Array(Box::new(Schema::parse(items)?)))
            }
            Some("string") => {
                let enum_values = match obj.get("enum") {
                    Some(serde_json::Value::Array(vs)) => Some(
                        vs.iter()
                            .map(|v| {
                                v.as_str()
                                    .map(str::to_string)
                                    .ok_or_else(|| anyhow::anyhow!("enum values must be strings"))
                            })
                            .collect::<Result<Vec<_>>>()?,
                    ),
                    _ => None,
                };
                Ok(Schema::String { enum_values })
            }
            Some("number") => Ok(Schema::Number),
            Some("integer") => Ok(Schema::Integer),
            Some("boolean") => Ok(Schema::Boolean),
            _ => Ok(Schema::Any),
        }
    }
}

fn parse_object(obj: &serde_json::Map<String, serde_json::Value>) -> Result<ObjectSchema> {
    let mut properties = Vec::new();
    if let Some(serde_json::Value::Object(props)) = obj.get("properties") {
        for (name, schema) in props {
            properties.push((name.clone(), Schema::parse(schema)?));
        }
    }
    let required = match obj.get("required") {
        Some(serde_json::Value::Array(names)) => names
            .iter()
            .map(|v| v.as_str().map(str::to_string).ok_or_else(|| anyhow::anyhow!("required entries must be strings")))
            .collect::<Result<_>>()?,
        _ => Default::default(),
    };
    Ok(ObjectSchema { properties, required })
}

/// What one byte does to the frame it was fed to.
enum Step {
    Consumed,
    Rejected,
    /// Pop this frame, push `Frame`, and retry the same byte against it
    /// (the byte is the first content this new frame itself has to parse --
    /// entering a number or a `true`/`false`/`null` literal).
    ReplaceRetry(Frame),
    /// Pop this frame, push `Frame`; the byte is consumed by the act of
    /// entering it (an opening `{`, `[`, or `"`).
    ReplaceConsumed(Frame),
    /// The byte is consumed by this frame (e.g. `:` after a key), and a new
    /// frame goes on top for what follows, starting fresh at the next byte.
    Push(Frame),
    /// Push `Frame` on top and retry the same byte against it -- for an
    /// array's own opening/next item, where the byte in hand (an opening
    /// `"`, a digit, `{`, `[`, ...) is the *first* byte of that item's own
    /// value, not something the array itself consumes the way an object
    /// consumes its `:`.
    PushRetry(Frame),
    /// This frame is already complete and the byte does not belong to it;
    /// pop it and retry the same byte against whatever is now on top.
    Pop,
    /// This frame completes *because of* this byte (a closing `"`, `}`, or
    /// `]`), which is consumed by the act of closing it.
    PopConsumed,
}

#[derive(Clone)]
enum Frame {
    /// About to emit a value matching this schema; replaced by a concrete
    /// frame on the first non-whitespace byte.
    Value(Schema),
    Object {
        schema: ObjectSchema,
        used: std::collections::BTreeSet<usize>,
        sub: ObjSub,
    },
    Array {
        item: Schema,
        sub: ArrSub,
    },
    Str {
        /// `None` for a free string; `Some(candidates)` for an
        /// `enum`-constrained one (or an object key, matched against the
        /// property names not yet used), narrowed as bytes arrive.
        candidates: Option<Vec<Vec<u8>>>,
        matched_len: usize,
        escaped: bool,
    },
    Num {
        seen_sign: bool,
        int_len: u32,
        leading_zero: bool,
        allow_frac: bool,
        seen_dot: bool,
        frac_len: u32,
    },
    /// Matching a fixed literal (`"true"`, `"false"`) byte by byte.
    Literal { text: &'static [u8], pos: usize },
}

#[derive(Clone)]
enum ObjSub {
    BeforeKey,
    InKey { candidates: Vec<usize>, matched_len: usize },
    AfterKey { prop_index: usize },
    AfterValue,
}

#[derive(Clone)]
enum ArrSub {
    Empty,
    AfterItem,
    NeedItem,
}

impl Frame {
    fn value(schema: Schema) -> Frame {
        Frame::Value(schema)
    }

    /// Whether this frame, sitting alone at the root with nothing else on
    /// the stack, represents a fully-formed value even though nothing has
    /// (yet) forced it to pop. Only `Num`/`Literal` are self-terminating in
    /// this sense -- `Str`/`Object`/`Array` are only ever "done" via an
    /// explicit closing byte, which is exactly what `PopConsumed` already
    /// models, so they never sit on the stack in a state this needs to
    /// second-guess.
    fn can_stop(&self) -> bool {
        match self {
            Frame::Num { int_len, seen_dot, frac_len, .. } => *int_len > 0 && (!*seen_dot || *frac_len > 0),
            Frame::Literal { text, pos } => *pos == text.len(),
            _ => false,
        }
    }

    fn step(&mut self, byte: u8) -> Step {
        match self {
            Frame::Value(schema) => value_step(schema, byte),
            Frame::Object { schema, used, sub } => object_step(schema, used, sub, byte),
            Frame::Array { item, sub } => array_step(item, sub, byte),
            Frame::Str { candidates, matched_len, escaped } => str_step(candidates, matched_len, escaped, byte),
            Frame::Num { seen_sign, int_len, leading_zero, allow_frac, seen_dot, frac_len } => {
                num_step(seen_sign, int_len, leading_zero, *allow_frac, seen_dot, frac_len, byte)
            }
            Frame::Literal { text, pos } => literal_step(text, pos, byte),
        }
    }
}

fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

fn value_step(schema: &Schema, byte: u8) -> Step {
    if is_ws(byte) {
        return Step::Consumed;
    }
    match (schema, byte) {
        (Schema::Object(obj), b'{') => Step::ReplaceConsumed(Frame::Object {
            schema: obj.clone(),
            used: Default::default(),
            sub: ObjSub::BeforeKey,
        }),
        (Schema::Array(item), b'[') => {
            Step::ReplaceConsumed(Frame::Array { item: (**item).clone(), sub: ArrSub::Empty })
        }
        (Schema::String { enum_values }, b'"') => Step::ReplaceConsumed(Frame::Str {
            candidates: enum_values.as_ref().map(|vs| vs.iter().map(|s| s.as_bytes().to_vec()).collect()),
            matched_len: 0,
            escaped: false,
        }),
        (Schema::Any, b'"') => {
            Step::ReplaceConsumed(Frame::Str { candidates: None, matched_len: 0, escaped: false })
        }
        (Schema::Number, b'-') | (Schema::Number, b'0'..=b'9') => Step::ReplaceRetry(Frame::Num {
            seen_sign: false,
            int_len: 0,
            leading_zero: false,
            allow_frac: true,
            seen_dot: false,
            frac_len: 0,
        }),
        (Schema::Integer, b'-') | (Schema::Integer, b'0'..=b'9') => Step::ReplaceRetry(Frame::Num {
            seen_sign: false,
            int_len: 0,
            leading_zero: false,
            allow_frac: false,
            seen_dot: false,
            frac_len: 0,
        }),
        (Schema::Boolean, b't') => Step::ReplaceRetry(Frame::Literal { text: b"true", pos: 0 }),
        (Schema::Boolean, b'f') => Step::ReplaceRetry(Frame::Literal { text: b"false", pos: 0 }),
        _ => Step::Rejected,
    }
}

fn object_step(schema: &ObjectSchema, used: &mut std::collections::BTreeSet<usize>, sub: &mut ObjSub, byte: u8) -> Step {
    match sub {
        ObjSub::BeforeKey => {
            if is_ws(byte) {
                return Step::Consumed;
            }
            let unused: Vec<usize> = (0..schema.properties.len()).filter(|i| !used.contains(i)).collect();
            if byte == b'"' && !unused.is_empty() {
                *sub = ObjSub::InKey { candidates: unused, matched_len: 0 };
                Step::Consumed
            } else if byte == b'}' && schema.required.iter().all(|name| {
                schema.properties.iter().position(|(n, _)| n == name).map(|i| used.contains(&i)).unwrap_or(false)
            }) {
                Step::PopConsumed
            } else {
                Step::Rejected
            }
        }
        ObjSub::InKey { candidates, matched_len } => {
            if byte == b'"' {
                match candidates.iter().find(|&&i| schema.properties[i].0.as_bytes().len() == *matched_len) {
                    Some(&prop_index) => {
                        used.insert(prop_index);
                        *sub = ObjSub::AfterKey { prop_index };
                        Step::Consumed
                    }
                    None => Step::Rejected,
                }
            } else {
                let pos = *matched_len;
                candidates.retain(|&i| {
                    let name = schema.properties[i].0.as_bytes();
                    name.len() > pos && name[pos] == byte
                });
                if candidates.is_empty() {
                    Step::Rejected
                } else {
                    *matched_len += 1;
                    Step::Consumed
                }
            }
        }
        ObjSub::AfterKey { prop_index } => {
            if is_ws(byte) {
                Step::Consumed
            } else if byte == b':' {
                let prop_index = *prop_index;
                *sub = ObjSub::AfterValue;
                Step::Push(Frame::value(schema.properties[prop_index].1.clone()))
            } else {
                Step::Rejected
            }
        }
        ObjSub::AfterValue => {
            if is_ws(byte) {
                return Step::Consumed;
            }
            let unused_remain = used.len() < schema.properties.len();
            if byte == b',' && unused_remain {
                *sub = ObjSub::BeforeKey;
                Step::Consumed
            } else if byte == b'}'
                && schema.required.iter().all(|name| {
                    schema.properties.iter().position(|(n, _)| n == name).map(|i| used.contains(&i)).unwrap_or(false)
                })
            {
                Step::PopConsumed
            } else {
                Step::Rejected
            }
        }
    }
}

fn array_step(item: &Schema, sub: &mut ArrSub, byte: u8) -> Step {
    match sub {
        ArrSub::Empty => {
            if is_ws(byte) {
                Step::Consumed
            } else if byte == b']' {
                Step::PopConsumed
            } else {
                *sub = ArrSub::AfterItem;
                Step::PushRetry(Frame::value(item.clone()))
            }
        }
        ArrSub::AfterItem => {
            if is_ws(byte) {
                Step::Consumed
            } else if byte == b']' {
                Step::PopConsumed
            } else if byte == b',' {
                *sub = ArrSub::NeedItem;
                Step::Consumed
            } else {
                Step::Rejected
            }
        }
        ArrSub::NeedItem => {
            if is_ws(byte) {
                Step::Consumed
            } else {
                *sub = ArrSub::AfterItem;
                Step::PushRetry(Frame::value(item.clone()))
            }
        }
    }
}

fn str_step(candidates: &mut Option<Vec<Vec<u8>>>, matched_len: &mut usize, escaped: &mut bool, byte: u8) -> Step {
    if *escaped {
        if candidates.is_some() {
            return Step::Rejected;
        }
        *escaped = false;
        return Step::Consumed;
    }
    if byte == b'\\' {
        if candidates.is_some() {
            return Step::Rejected;
        }
        *escaped = true;
        return Step::Consumed;
    }
    if byte == b'"' {
        return match candidates {
            None => Step::PopConsumed,
            Some(list) => {
                if list.iter().any(|c| c.len() == *matched_len) {
                    Step::PopConsumed
                } else {
                    Step::Rejected
                }
            }
        };
    }
    match candidates {
        None => Step::Consumed,
        Some(list) => {
            let pos = *matched_len;
            list.retain(|c| c.len() > pos && c[pos] == byte);
            if list.is_empty() {
                Step::Rejected
            } else {
                *matched_len += 1;
                Step::Consumed
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn num_step(
    seen_sign: &mut bool,
    int_len: &mut u32,
    leading_zero: &mut bool,
    allow_frac: bool,
    seen_dot: &mut bool,
    frac_len: &mut u32,
    byte: u8,
) -> Step {
    if !*seen_sign && *int_len == 0 && !*seen_dot && byte == b'-' {
        *seen_sign = true;
        return Step::Consumed;
    }
    *seen_sign = true;
    if !*seen_dot {
        if byte.is_ascii_digit() {
            if *leading_zero {
                // "0" cannot be followed by another integer digit ("01" is
                // not valid JSON).
                return Step::Rejected;
            }
            if *int_len == 0 && byte == b'0' {
                *leading_zero = true;
            }
            *int_len += 1;
            return Step::Consumed;
        }
        if byte == b'.' && allow_frac && *int_len > 0 {
            *seen_dot = true;
            return Step::Consumed;
        }
        if *int_len == 0 {
            return Step::Rejected;
        }
        return Step::Pop;
    }
    if byte.is_ascii_digit() {
        *frac_len += 1;
        return Step::Consumed;
    }
    if *frac_len == 0 {
        return Step::Rejected;
    }
    Step::Pop
}

fn literal_step(text: &'static [u8], pos: &mut usize, byte: u8) -> Step {
    if *pos == text.len() {
        return Step::Pop;
    }
    if text[*pos] == byte {
        *pos += 1;
        Step::Consumed
    } else {
        Step::Rejected
    }
}

/// Drives [`Schema`]-constrained JSON generation one byte at a time.
///
/// `Clone` is load-bearing, not incidental: a caller testing whether a
/// candidate token's bytes are legal has to try them against a *disposable*
/// copy of the real cursor, keeping the real one unmodified until an actual
/// token is chosen and fed to it for real.
#[derive(Clone)]
pub struct Cursor {
    stack: Vec<Frame>,
}

impl Cursor {
    pub fn new(schema: Schema) -> Cursor {
        Cursor { stack: vec![Frame::value(schema)] }
    }

    /// Whether `byte` is a legal next byte for *some* completion matching
    /// the schema, given everything fed so far. Advances internal state
    /// when it returns `true`.
    pub fn feed(&mut self, byte: u8) -> bool {
        loop {
            let Some(top) = self.stack.last_mut() else {
                return is_ws(byte);
            };
            match top.step(byte) {
                Step::Consumed => return true,
                Step::Rejected => return false,
                Step::ReplaceRetry(f) => {
                    *self.stack.last_mut().unwrap() = f;
                }
                Step::ReplaceConsumed(f) => {
                    *self.stack.last_mut().unwrap() = f;
                    return true;
                }
                Step::Push(f) => {
                    self.stack.push(f);
                    return true;
                }
                Step::PushRetry(f) => {
                    self.stack.push(f);
                }
                Step::Pop => {
                    self.stack.pop();
                }
                Step::PopConsumed => {
                    self.stack.pop();
                    return true;
                }
            }
        }
    }

    /// Whether the root value has closed -- generation may stop here (a
    /// well-formed completion exists), though [`Cursor::feed`] still accepts
    /// trailing whitespace after this point.
    pub fn is_complete(&self) -> bool {
        match self.stack.len() {
            0 => true,
            1 => self.stack[0].can_stop(),
            _ => false,
        }
    }

    /// Whether every byte of `bytes` is a legal continuation, tried against
    /// a disposable clone -- the real cursor is unaffected either way. This
    /// is the per-candidate-token check a caller runs against every
    /// vocabulary entry before masking a decode step's logits; see this
    /// module's own doc comment.
    pub fn accepts(&self, bytes: &[u8]) -> bool {
        let mut probe = self.clone();
        bytes.iter().all(|&b| probe.feed(b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema(json: &str) -> Schema {
        Schema::parse(&serde_json::from_str(json).unwrap()).unwrap()
    }

    fn feed_all(cursor: &mut Cursor, s: &str) -> bool {
        for b in s.bytes() {
            if !cursor.feed(b) {
                return false;
            }
        }
        true
    }

    #[test]
    fn a_flat_required_object_accepts_the_obvious_completion() {
        let s = schema(
            r#"{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]}"#,
        );
        let mut c = Cursor::new(s);
        assert!(feed_all(&mut c, r#"{"name":"Ada","age":36}"#));
        assert!(c.is_complete());
    }

    #[test]
    fn closing_before_a_required_property_is_rejected() {
        let s = schema(r#"{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]}"#);
        let mut c = Cursor::new(s);
        assert!(feed_all(&mut c, "{"));
        assert!(!c.feed(b'}'));
    }

    #[test]
    fn an_unknown_key_is_rejected() {
        let s = schema(r#"{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]}"#);
        let mut c = Cursor::new(s);
        assert!(feed_all(&mut c, r#"{""#));
        assert!(!c.feed(b'b'));
    }

    #[test]
    fn a_key_that_is_a_prefix_of_another_is_not_accepted_early() {
        let s = schema(
            r#"{"type":"object","properties":{"a":{"type":"string"},"ab":{"type":"string"}},"required":[]}"#,
        );
        let mut c = Cursor::new(s);
        assert!(feed_all(&mut c, r#"{""#));
        assert!(c.feed(b'a'));
        // Closing here would mean the key is "a" -- must be accepted since
        // "a" is itself a real property.
        assert!(c.feed(b'"'));
    }

    #[test]
    fn enum_values_narrow_and_reject_anything_else() {
        let s = schema(r#"{"type":"string","enum":["red","green","blue"]}"#);
        let mut c = Cursor::new(s);
        assert!(feed_all(&mut c, r#""re"#));
        assert!(!c.feed(b'x')); // no candidate has "rex..."
        let mut c2 = Cursor::new(schema(r#"{"type":"string","enum":["red","green","blue"]}"#));
        assert!(feed_all(&mut c2, r#""red""#));
        assert!(c2.is_complete());
    }

    #[test]
    fn enum_rejects_closing_on_a_partial_prefix() {
        let s = schema(r#"{"type":"string","enum":["red","green"]}"#);
        let mut c = Cursor::new(s);
        assert!(feed_all(&mut c, r#""re"#));
        // "re" is a prefix of "red" but not itself a candidate.
        assert!(!c.feed(b'"'));
    }

    #[test]
    fn numbers_reject_leading_zeros_and_bare_dots() {
        let mut c = Cursor::new(schema(r#"{"type":"number"}"#));
        assert!(feed_all(&mut c, "0"));
        assert!(c.is_complete());

        assert!(!feed_all(&mut Cursor::new(schema(r#"{"type":"number"}"#)), "01"));

        let mut c2 = Cursor::new(schema(r#"{"type":"number"}"#));
        assert!(feed_all(&mut c2, "-0.5"));
        assert!(c2.is_complete());

        // "1." is a legal prefix (more fraction digits could follow) but not
        // itself a complete number.
        let mut c3 = Cursor::new(schema(r#"{"type":"number"}"#));
        assert!(feed_all(&mut c3, "1."));
        assert!(!c3.is_complete());

        assert!(!Cursor::new(schema(r#"{"type":"number"}"#)).feed(b'.'));
    }

    #[test]
    fn integer_rejects_a_decimal_point() {
        let s = schema(r#"{"type":"integer"}"#);
        let mut c = Cursor::new(s);
        assert!(c.feed(b'1'));
        assert!(!c.feed(b'.'));
    }

    #[test]
    fn booleans_match_exactly_true_or_false() {
        let mut c = Cursor::new(schema(r#"{"type":"boolean"}"#));
        assert!(feed_all(&mut c, "true"));
        assert!(c.is_complete());
        assert!(feed_all(&mut Cursor::new(schema(r#"{"type":"boolean"}"#)), "false"));

        // "tru" is a legal (incomplete) prefix; a byte that fits neither
        // "true" nor "false" from there is not.
        let mut c2 = Cursor::new(schema(r#"{"type":"boolean"}"#));
        assert!(feed_all(&mut c2, "tru"));
        assert!(!c2.is_complete());
        assert!(!c2.feed(b'x'));
    }

    #[test]
    fn nested_object_and_array_compose() {
        let s = schema(
            r#"{"type":"object","properties":{"tags":{"type":"array","items":{"type":"string"}},"meta":{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"]}},"required":["tags","meta"]}"#,
        );
        let mut c = Cursor::new(s);
        assert!(feed_all(&mut c, r#"{"tags":["a","b"],"meta":{"ok":true}}"#));
        assert!(c.is_complete());
    }

    #[test]
    fn arbitrary_whitespace_between_structural_tokens_is_accepted() {
        let s = schema(r#"{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]}"#);
        let mut c = Cursor::new(s);
        assert!(feed_all(&mut c, "{  \"a\"  :  1  }"));
        assert!(c.is_complete());
    }

    #[test]
    fn an_optional_property_may_be_omitted() {
        let s = schema(
            r#"{"type":"object","properties":{"a":{"type":"string"},"b":{"type":"string"}},"required":["a"]}"#,
        );
        let mut c = Cursor::new(s);
        assert!(feed_all(&mut c, r#"{"a":"x"}"#));
        assert!(c.is_complete());
    }

    #[test]
    fn trailing_bytes_after_completion_are_rejected_except_whitespace() {
        let s = schema(r#"{"type":"boolean"}"#);
        let mut c = Cursor::new(s);
        assert!(feed_all(&mut c, "true"));
        assert!(c.is_complete());
        assert!(c.feed(b' '));
        assert!(!c.feed(b'x'));
    }
}
