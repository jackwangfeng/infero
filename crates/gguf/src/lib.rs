//! GGUF container parsing.
//!
//! The file is mapped, never read into a buffer: tensor payloads are handed out
//! as slices into the mapping so weight upload is one `cudaMemcpy` from page
//! cache. Only the header — metadata and the tensor index — is materialized.
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! let f = tuili_gguf::Gguf::open("models/qwen2.5-0.5b-instruct-q8_0.gguf")?;
//! println!("{} ({} tensors)", f.arch()?, f.tensors().len());
//! let w = f.tensor("blk.0.attn_q.weight")?;
//! println!("{} {:?} {}", w.name, w.dims, w.ty);
//! # Ok(()) }
//! ```

mod reader;
mod types;
mod value;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use memmap2::Mmap;

use reader::Reader;
pub use types::{GgmlType, QK_K};
pub use value::{Array, Value, ValueType};

const MAGIC: u32 = u32::from_le_bytes(*b"GGUF");
const DEFAULT_ALIGNMENT: usize = 32;

/// Where one tensor lives inside the file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// ggml order: `dims[0]` is the fastest-moving axis. A linear weight that
    /// torch would call `[out_features, in_features]` is `[in, out]` here.
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    /// Byte offset from the start of the tensor data section.
    pub offset: u64,
    pub n_elements: usize,
    pub n_bytes: usize,
}

impl TensorInfo {
    /// `dims` reversed, i.e. row-major / torch order.
    pub fn shape(&self) -> Vec<u64> {
        self.dims.iter().rev().copied().collect()
    }
}

pub struct Gguf {
    path: PathBuf,
    map: Mmap,
    version: u32,
    alignment: usize,
    data_offset: usize,
    metadata: BTreeMap<String, Value>,
    tensors: BTreeMap<String, TensorInfo>,
}

impl Gguf {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file =
            std::fs::File::open(&path).with_context(|| format!("opening {}", path.display()))?;
        // Safety: we treat the mapping as immutable. A concurrent writer
        // truncating the file would be UB, which is the same contract
        // llama.cpp works under.
        let map =
            unsafe { Mmap::map(&file) }.with_context(|| format!("mmapping {}", path.display()))?;
        Self::from_mmap(path, map)
    }

    fn from_mmap(path: PathBuf, map: Mmap) -> Result<Self> {
        let mut r = Reader::new(&map);

        let magic = r.u32()?;
        if magic != MAGIC {
            bail!(
                "{} is not a GGUF file (magic {magic:#010x})",
                path.display()
            );
        }
        let version = r.u32()?;
        if !(2..=3).contains(&version) {
            bail!("unsupported GGUF version {version}");
        }
        let tensor_count = r.u64()? as usize;
        let kv_count = r.u64()? as usize;

        let mut metadata = BTreeMap::new();
        for i in 0..kv_count {
            let key = r
                .string()
                .with_context(|| format!("reading metadata key #{i}"))?;
            let value = r
                .value()
                .with_context(|| format!("reading metadata value for `{key}`"))?;
            metadata.insert(key, value);
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(Value::as_u64)
            .unwrap_or(DEFAULT_ALIGNMENT as u64) as usize;
        if !alignment.is_power_of_two() {
            bail!("general.alignment {alignment} is not a power of two");
        }

        let mut tensors = BTreeMap::new();
        for i in 0..tensor_count {
            let info =
                read_tensor_info(&mut r).with_context(|| format!("reading tensor info #{i}"))?;
            if let Some(prev) = tensors.insert(info.name.clone(), info) {
                bail!("duplicate tensor `{}`", prev.name);
            }
        }

        let data_offset = r.pos().next_multiple_of(alignment);
        if data_offset > map.len() {
            bail!("tensor data starts past end of file");
        }

        let this = Self {
            path,
            map,
            version,
            alignment,
            data_offset,
            metadata,
            tensors,
        };
        this.validate_extents()?;
        Ok(this)
    }

    /// Every tensor must lie inside the mapping. Checking once up front means
    /// `data()` can be a plain slice index.
    fn validate_extents(&self) -> Result<()> {
        let available = self.map.len() - self.data_offset;
        for t in self.tensors.values() {
            let end = t
                .offset
                .checked_add(t.n_bytes as u64)
                .context("tensor extent overflows")?;
            if end > available as u64 {
                bail!(
                    "tensor `{}` runs to {end} but only {available} bytes of data follow the header",
                    t.name
                );
            }
        }
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn alignment(&self) -> usize {
        self.alignment
    }

    /// Size of the tensor data section in bytes.
    pub fn data_len(&self) -> usize {
        self.map.len() - self.data_offset
    }

    pub fn metadata(&self) -> &BTreeMap<String, Value> {
        &self.metadata
    }

    pub fn tensors(&self) -> &BTreeMap<String, TensorInfo> {
        &self.tensors
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors
            .get(name)
            .with_context(|| format!("no tensor `{name}` in {}", self.path.display()))
    }

    pub fn get_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// The raw bytes of a tensor, still in its ggml block encoding.
    pub fn data(&self, t: &TensorInfo) -> &[u8] {
        let start = self.data_offset + t.offset as usize;
        &self.map[start..start + t.n_bytes]
    }

    pub fn tensor_data(&self, name: &str) -> Result<&[u8]> {
        Ok(self.data(self.tensor(name)?))
    }

    // ---- metadata access -------------------------------------------------

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.metadata.get(key)
    }

    fn require(&self, key: &str) -> Result<&Value> {
        self.metadata
            .get(key)
            .with_context(|| format!("missing metadata key `{key}`"))
    }

    pub fn u64(&self, key: &str) -> Result<u64> {
        let v = self.require(key)?;
        v.as_u64()
            .with_context(|| format!("`{key}` is {}, want an integer", v.type_name()))
    }

    pub fn u32(&self, key: &str) -> Result<u32> {
        let v = self.u64(key)?;
        u32::try_from(v).with_context(|| format!("`{key}` = {v} does not fit in u32"))
    }

    pub fn usize(&self, key: &str) -> Result<usize> {
        Ok(self.u64(key)? as usize)
    }

    pub fn f32(&self, key: &str) -> Result<f32> {
        let v = self.require(key)?;
        v.as_f64()
            .map(|f| f as f32)
            .with_context(|| format!("`{key}` is {}, want a float", v.type_name()))
    }

    pub fn bool(&self, key: &str) -> Result<bool> {
        let v = self.require(key)?;
        v.as_bool()
            .with_context(|| format!("`{key}` is {}, want a bool", v.type_name()))
    }

    pub fn str(&self, key: &str) -> Result<&str> {
        let v = self.require(key)?;
        v.as_str()
            .with_context(|| format!("`{key}` is {}, want a string", v.type_name()))
    }

    pub fn str_array(&self, key: &str) -> Result<&[String]> {
        let v = self.require(key)?;
        v.as_array()
            .and_then(Array::as_strings)
            .with_context(|| format!("`{key}` is {}, want a string array", v.type_name()))
    }

    pub fn int_array(&self, key: &str) -> Result<Vec<i64>> {
        let v = self.require(key)?;
        v.as_array()
            .and_then(Array::to_i64_vec)
            .with_context(|| format!("`{key}` is {}, want an integer array", v.type_name()))
    }

    pub fn f32_array(&self, key: &str) -> Result<&[f32]> {
        let v = self.require(key)?;
        v.as_array()
            .and_then(Array::as_f32)
            .with_context(|| format!("`{key}` is {}, want an f32 array", v.type_name()))
    }

    /// `general.architecture`, e.g. `"qwen2"` or `"llama"`.
    pub fn arch(&self) -> Result<&str> {
        self.str("general.architecture")
    }

    /// Resolve an architecture-scoped key: `akey("block_count")` becomes
    /// `"qwen2.block_count"` for a Qwen2 file.
    pub fn akey(&self, suffix: &str) -> Result<String> {
        Ok(format!("{}.{suffix}", self.arch()?))
    }

    /// The dominant quantization across the weight tensors, which is what
    /// people mean by "a Q4_K_M model".
    pub fn dominant_type(&self) -> Option<GgmlType> {
        let mut counts: BTreeMap<GgmlType, usize> = BTreeMap::new();
        for t in self.tensors.values() {
            if t.ty.is_quantized() {
                *counts.entry(t.ty).or_default() += t.n_bytes;
            }
        }
        counts.into_iter().max_by_key(|&(_, n)| n).map(|(t, _)| t)
    }
}

impl std::fmt::Debug for Gguf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gguf")
            .field("path", &self.path)
            .field("version", &self.version)
            .field("tensors", &self.tensors.len())
            .field("metadata", &self.metadata.len())
            .field("data_len", &self.data_len())
            .finish()
    }
}

// GgmlType needs Ord for the BTreeMap in dominant_type.
impl PartialOrd for GgmlType {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for GgmlType {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (*self as u32).cmp(&(*other as u32))
    }
}

fn read_tensor_info(r: &mut Reader<'_>) -> Result<TensorInfo> {
    let name = r.string()?;
    let n_dims = r.u32()? as usize;
    if n_dims == 0 || n_dims > 4 {
        bail!("tensor `{name}` has {n_dims} dimensions, expected 1..=4");
    }
    let mut dims = Vec::with_capacity(n_dims);
    let mut n_elements: usize = 1;
    for _ in 0..n_dims {
        let d = r.u64()?;
        n_elements = n_elements
            .checked_mul(d as usize)
            .with_context(|| format!("tensor `{name}` element count overflows"))?;
        dims.push(d);
    }
    let ty = GgmlType::from_u32(r.u32()?)
        .with_context(|| format!("tensor `{name}` has an unknown type"))?;
    let offset = r.u64()?;
    let n_bytes = ty
        .size_for(n_elements)
        .with_context(|| format!("tensor `{name}` has a bad shape for {ty}"))?;

    Ok(TensorInfo {
        name,
        dims,
        ty,
        offset,
        n_elements,
        n_bytes,
    })
}
