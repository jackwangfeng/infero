//! Bounds-checked little-endian cursor over the mapped header.

use anyhow::{Context, Result, bail};

use crate::value::{Array, Value, ValueType};

/// Refuse absurd lengths before allocating. The largest real vocab is a few
/// hundred thousand entries, so anything past this is a corrupt file.
const MAX_ARRAY_LEN: u64 = 1 << 28;
const MAX_STRING_LEN: u64 = 1 << 30;

pub struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .context("gguf header offset overflows")?;
        if end > self.buf.len() {
            bail!(
                "unexpected end of gguf header: want {n} bytes at {}, only {} left",
                self.pos,
                self.buf.len() - self.pos.min(self.buf.len())
            );
        }
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    pub fn i8(&mut self) -> Result<i8> {
        Ok(self.u8()? as i8)
    }

    pub fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    pub fn i16(&mut self) -> Result<i16> {
        Ok(self.u16()? as i16)
    }

    pub fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    pub fn i32(&mut self) -> Result<i32> {
        Ok(self.u32()? as i32)
    }

    pub fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    pub fn i64(&mut self) -> Result<i64> {
        Ok(self.u64()? as i64)
    }

    pub fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    pub fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    pub fn bool(&mut self) -> Result<bool> {
        Ok(self.u8()? != 0)
    }

    /// Length-prefixed UTF-8. Vocabularies contain byte-fallback tokens that
    /// are not valid UTF-8 on their own, so invalid sequences are replaced
    /// rather than rejected.
    pub fn string(&mut self) -> Result<String> {
        let len = self.u64()?;
        if len > MAX_STRING_LEN {
            bail!("implausible gguf string length {len}");
        }
        let bytes = self.take(len as usize)?;
        Ok(match std::str::from_utf8(bytes) {
            Ok(s) => s.to_string(),
            Err(_) => String::from_utf8_lossy(bytes).into_owned(),
        })
    }

    pub fn value(&mut self) -> Result<Value> {
        let ty = ValueType::from_u32(self.u32()?)?;
        self.value_of(ty)
    }

    fn value_of(&mut self, ty: ValueType) -> Result<Value> {
        Ok(match ty {
            ValueType::U8 => Value::U8(self.u8()?),
            ValueType::I8 => Value::I8(self.i8()?),
            ValueType::U16 => Value::U16(self.u16()?),
            ValueType::I16 => Value::I16(self.i16()?),
            ValueType::U32 => Value::U32(self.u32()?),
            ValueType::I32 => Value::I32(self.i32()?),
            ValueType::U64 => Value::U64(self.u64()?),
            ValueType::I64 => Value::I64(self.i64()?),
            ValueType::F32 => Value::F32(self.f32()?),
            ValueType::F64 => Value::F64(self.f64()?),
            ValueType::Bool => Value::Bool(self.bool()?),
            ValueType::String => Value::String(self.string()?),
            ValueType::Array => Value::Array(self.array()?),
        })
    }

    fn array(&mut self) -> Result<Array> {
        let elem = ValueType::from_u32(self.u32()?)?;
        let len = self.u64()?;
        if len > MAX_ARRAY_LEN {
            bail!("implausible gguf array length {len}");
        }
        let len = len as usize;

        // Sanity-check the fixed-width cases against what's left in the file
        // before reserving, so a corrupt length can't drive a huge allocation.
        let width = match elem {
            ValueType::U8 | ValueType::I8 | ValueType::Bool => 1,
            ValueType::U16 | ValueType::I16 => 2,
            ValueType::U32 | ValueType::I32 | ValueType::F32 => 4,
            ValueType::U64 | ValueType::I64 | ValueType::F64 => 8,
            ValueType::String | ValueType::Array => 0,
        };
        if width > 0 && len.saturating_mul(width) > self.buf.len() - self.pos {
            bail!("array of {len} {elem:?} does not fit in the file");
        }

        macro_rules! collect {
            ($variant:ident, $read:ident) => {{
                let mut v = Vec::with_capacity(len);
                for _ in 0..len {
                    v.push(self.$read()?);
                }
                Array::$variant(v)
            }};
        }

        Ok(match elem {
            ValueType::U8 => collect!(U8, u8),
            ValueType::I8 => collect!(I8, i8),
            ValueType::U16 => collect!(U16, u16),
            ValueType::I16 => collect!(I16, i16),
            ValueType::U32 => collect!(U32, u32),
            ValueType::I32 => collect!(I32, i32),
            ValueType::U64 => collect!(U64, u64),
            ValueType::I64 => collect!(I64, i64),
            ValueType::F32 => collect!(F32, f32),
            ValueType::F64 => collect!(F64, f64),
            ValueType::Bool => collect!(Bool, bool),
            ValueType::String => collect!(String, string),
            ValueType::Array => {
                let mut v = Vec::with_capacity(len.min(1024));
                for _ in 0..len {
                    v.push(self.array()?);
                }
                Array::Array(v)
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_past_end_are_errors() {
        let mut r = Reader::new(&[1, 2, 3]);
        assert!(r.u32().is_err());
        assert_eq!(r.pos(), 0, "a failed read must not advance");
    }

    #[test]
    fn string_roundtrip() {
        let mut buf = 5u64.to_le_bytes().to_vec();
        buf.extend_from_slice(b"hello");
        let mut r = Reader::new(&buf);
        assert_eq!(r.string().unwrap(), "hello");
    }

    #[test]
    fn corrupt_array_length_is_rejected() {
        let mut buf = (ValueType::U32 as u32).to_le_bytes().to_vec();
        buf.extend_from_slice(&u64::MAX.to_le_bytes());
        let mut r = Reader::new(&buf);
        assert!(r.array().is_err());
    }
}
