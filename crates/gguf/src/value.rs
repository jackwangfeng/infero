//! GGUF metadata values.

use anyhow::{Result, bail};

/// A metadata value. Arrays are homogeneous and split by element type so that
/// the common cases (token lists, merge lists) stay a plain `Vec`.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Array),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Array {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    Bool(Vec<bool>),
    String(Vec<String>),
    /// Nested arrays are legal in the spec but no real model uses them.
    Array(Vec<Array>),
}

impl Array {
    pub fn len(&self) -> usize {
        match self {
            Array::U8(v) => v.len(),
            Array::I8(v) => v.len(),
            Array::U16(v) => v.len(),
            Array::I16(v) => v.len(),
            Array::U32(v) => v.len(),
            Array::I32(v) => v.len(),
            Array::U64(v) => v.len(),
            Array::I64(v) => v.len(),
            Array::F32(v) => v.len(),
            Array::F64(v) => v.len(),
            Array::Bool(v) => v.len(),
            Array::String(v) => v.len(),
            Array::Array(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_strings(&self) -> Option<&[String]> {
        match self {
            Array::String(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<&[f32]> {
        match self {
            Array::F32(v) => Some(v),
            _ => None,
        }
    }

    /// Any integer array widened to i64. Token type arrays show up as I32 in
    /// some writers and U32 in others, so callers shouldn't have to care.
    pub fn to_i64_vec(&self) -> Option<Vec<i64>> {
        Some(match self {
            Array::U8(v) => v.iter().map(|&x| x as i64).collect(),
            Array::I8(v) => v.iter().map(|&x| x as i64).collect(),
            Array::U16(v) => v.iter().map(|&x| x as i64).collect(),
            Array::I16(v) => v.iter().map(|&x| x as i64).collect(),
            Array::U32(v) => v.iter().map(|&x| x as i64).collect(),
            Array::I32(v) => v.iter().map(|&x| x as i64).collect(),
            Array::U64(v) => v.iter().map(|&x| x as i64).collect(),
            Array::I64(v) => v.clone(),
            _ => return None,
        })
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Array::U8(_) => "u8",
            Array::I8(_) => "i8",
            Array::U16(_) => "u16",
            Array::I16(_) => "i16",
            Array::U32(_) => "u32",
            Array::I32(_) => "i32",
            Array::U64(_) => "u64",
            Array::I64(_) => "i64",
            Array::F32(_) => "f32",
            Array::F64(_) => "f64",
            Array::Bool(_) => "bool",
            Array::String(_) => "string",
            Array::Array(_) => "array",
        }
    }
}

impl Value {
    /// Any integer-ish scalar widened to u64. Config fields like
    /// `*.block_count` are u32 in practice but the spec permits others.
    pub fn as_u64(&self) -> Option<u64> {
        Some(match *self {
            Value::U8(v) => v as u64,
            Value::I8(v) if v >= 0 => v as u64,
            Value::U16(v) => v as u64,
            Value::I16(v) if v >= 0 => v as u64,
            Value::U32(v) => v as u64,
            Value::I32(v) if v >= 0 => v as u64,
            Value::U64(v) => v,
            Value::I64(v) if v >= 0 => v as u64,
            Value::Bool(v) => v as u64,
            _ => return None,
        })
    }

    pub fn as_i64(&self) -> Option<i64> {
        Some(match *self {
            Value::U8(v) => v as i64,
            Value::I8(v) => v as i64,
            Value::U16(v) => v as i64,
            Value::I16(v) => v as i64,
            Value::U32(v) => v as i64,
            Value::I32(v) => v as i64,
            Value::U64(v) => v as i64,
            Value::I64(v) => v,
            _ => return None,
        })
    }

    pub fn as_f64(&self) -> Option<f64> {
        Some(match *self {
            Value::F32(v) => v as f64,
            Value::F64(v) => v,
            _ => return self.as_i64().map(|v| v as f64),
        })
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Value::Bool(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&Array> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Value::U8(_) => "u8",
            Value::I8(_) => "i8",
            Value::U16(_) => "u16",
            Value::I16(_) => "i16",
            Value::U32(_) => "u32",
            Value::I32(_) => "i32",
            Value::U64(_) => "u64",
            Value::I64(_) => "i64",
            Value::F32(_) => "f32",
            Value::F64(_) => "f64",
            Value::Bool(_) => "bool",
            Value::String(_) => "string",
            Value::Array(a) => a.type_name(),
        }
    }
}

/// Wire tag for a metadata value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ValueType {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
}

impl ValueType {
    pub fn from_u32(v: u32) -> Result<Self> {
        use ValueType::*;
        Ok(match v {
            0 => U8,
            1 => I8,
            2 => U16,
            3 => I16,
            4 => U32,
            5 => I32,
            6 => F32,
            7 => Bool,
            8 => String,
            9 => Array,
            10 => U64,
            11 => I64,
            12 => F64,
            other => bail!("unknown gguf value type {other}"),
        })
    }
}

/// Short one-line rendering for `gguf-info`, so a 150k-token vocab doesn't
/// print itself.
impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::U8(v) => write!(f, "{v}"),
            Value::I8(v) => write!(f, "{v}"),
            Value::U16(v) => write!(f, "{v}"),
            Value::I16(v) => write!(f, "{v}"),
            Value::U32(v) => write!(f, "{v}"),
            Value::I32(v) => write!(f, "{v}"),
            Value::U64(v) => write!(f, "{v}"),
            Value::I64(v) => write!(f, "{v}"),
            Value::F32(v) => write!(f, "{v}"),
            Value::F64(v) => write!(f, "{v}"),
            Value::Bool(v) => write!(f, "{v}"),
            Value::String(s) if s.len() <= 96 => write!(f, "{s:?}"),
            Value::String(s) => write!(f, "{:?}... ({} bytes)", &s[..90], s.len()),
            Value::Array(a) => write!(f, "[{}; {}]", a.type_name(), a.len()),
        }
    }
}
