//! GGUF container parsing.
//!
//! The file is mapped, never read into a buffer: tensor payloads are handed out
//! as slices into the mapping so weight upload is one `cudaMemcpy` from page
//! cache. Only the header — metadata and the tensor index — is materialized.
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! let f = infero_gguf::Gguf::open("models/qwen2.5-0.5b-instruct-q8_0.gguf")?;
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

    /// A tensor's byte offset from the start of the *file*.
    ///
    /// `TensorInfo::offset` is relative to the data section; a caller that has
    /// mapped the whole file -- to alias it into device memory rather than copy
    /// out of it -- needs the absolute one.
    pub fn file_offset(&self, t: &TensorInfo) -> usize {
        self.data_offset + t.offset as usize
    }

    /// The raw bytes of a tensor, still in its ggml block encoding.
    pub fn data(&self, t: &TensorInfo) -> &[u8] {
        let start = self.data_offset + t.offset as usize;
        &self.map[start..start + t.n_bytes]
    }

    pub fn tensor_data(&self, name: &str) -> Result<&[u8]> {
        Ok(self.data(self.tensor(name)?))
    }

    /// Reads only the rows in `row_range` (torch/row-major order, i.e.
    /// indices into `t.shape()`'s leading dimension) from `t`, for
    /// tensor-parallel sharded loading -- never materializes the full
    /// tensor. `row_bytes` is computed as `t.n_bytes / t.shape()[0]` rather
    /// than derived from `t.ty`'s per-element width: this holds for any
    /// ggml encoding, quantized block formats included, as long as a row's
    /// worth of columns (`t.dims[0]`, ggml's fastest-moving axis) is a whole
    /// number of that format's blocks -- true for every real model shape
    /// this crate loads, since ggml itself requires it.
    pub fn tensor_shard(&self, t: &TensorInfo, row_range: std::ops::Range<usize>) -> Result<Vec<u8>> {
        let shape = t.shape();
        anyhow::ensure!(!shape.is_empty(), "cannot shard a scalar tensor");
        let n_rows = shape[0] as usize;
        anyhow::ensure!(
            row_range.end <= n_rows,
            "shard range {row_range:?} out of bounds for a {n_rows}-row tensor"
        );
        anyhow::ensure!(
            t.n_bytes % n_rows == 0,
            "tensor byte size {} does not divide evenly into {n_rows} rows -- \
             ragged block encoding this sharding scheme cannot slice safely",
            t.n_bytes
        );
        let row_bytes = t.n_bytes / n_rows;
        let full = self.data(t);
        let start = row_range.start * row_bytes;
        let end = row_range.end * row_bytes;
        Ok(full[start..end].to_vec())
    }

    /// Reads only the columns in `col_range` (ggml's fastest-moving axis,
    /// `t.dims[0]` -- the *input*/contraction dimension of a `[in, out]`
    /// linear weight) from every row of `t`, for row-parallel tensor-
    /// parallel sharding (e.g. an output/down projection, sharded along the
    /// dimension each rank's own partial input covers). Unlike
    /// [`Self::tensor_shard`], a column range is NOT one contiguous byte
    /// run -- ggml stores each row's `dims[0]` columns contiguously, so this
    /// does one small contiguous read per row rather than a single big one.
    ///
    /// Quantized formats (this crate's real checkpoints ship `Q8_0` for
    /// every sharded weight matrix, confirmed by inspection) pack
    /// `t.ty.block_size()` elements into one `t.ty.type_size()`-byte block
    /// with a shared scale -- a column boundary that lands mid-block would
    /// either be impossible to slice or silently split a scale factor
    /// between ranks, so `col_range` must be block-aligned; this is
    /// checked, not assumed.
    pub fn tensor_shard_cols(&self, t: &TensorInfo, col_range: std::ops::Range<usize>) -> Result<Vec<u8>> {
        let shape = t.shape();
        anyhow::ensure!(shape.len() >= 2, "tensor_shard_cols needs a >=2D tensor, got shape {shape:?}");
        let n_rows = shape[0] as usize;
        let n_cols = t.dims[0] as usize; // ggml's fastest axis == shape()'s last axis
        anyhow::ensure!(
            col_range.end <= n_cols,
            "shard range {col_range:?} out of bounds for a {n_cols}-column tensor"
        );
        let block = t.ty.block_size();
        anyhow::ensure!(
            col_range.start % block == 0 && col_range.end % block == 0,
            "column shard range {col_range:?} is not aligned to {}'s {block}-element block size",
            t.ty
        );
        let type_size = t.ty.type_size();
        let row_bytes = (n_cols / block) * type_size;
        anyhow::ensure!(
            row_bytes * n_rows == t.n_bytes,
            "computed row size {row_bytes} * {n_rows} rows doesn't match tensor byte size {}",
            t.n_bytes
        );
        let col_start_bytes = (col_range.start / block) * type_size;
        let shard_row_bytes = ((col_range.end - col_range.start) / block) * type_size;
        let full = self.data(t);
        let mut out = Vec::with_capacity(n_rows * shard_row_bytes);
        for row in 0..n_rows {
            let row_start = row * row_bytes + col_start_bytes;
            out.extend_from_slice(&full[row_start..row_start + shard_row_bytes]);
        }
        Ok(out)
    }

    /// Like [`Self::tensor_shard_cols`], but for `ranges.len() > 1` disjoint
    /// column ranges that together form one rank's shard -- interleaved
    /// *per row* into one compact buffer, which simple concatenation of
    /// separate `tensor_shard_cols` calls cannot produce: each of those
    /// calls already returns a full `[n_rows, that_range's_width]` buffer,
    /// so end-to-end concatenation would place one range's entire column
    /// block after every row of the other's, not the per-row-interleaved
    /// layout a real multi-segment column shard needs downstream. Exists for
    /// GDN's `v_heads_tiled` value-head sharding, where one rank's columns
    /// are `heads_per_key` disjoint ranges within the full column axis
    /// (llama.cpp's tiled reordering means a rank's key heads' value data
    /// is scattered across `heads_per_key` separate column bands, not one
    /// contiguous range) rather than a single range.
    pub fn tensor_shard_cols_multi(
        &self,
        t: &TensorInfo,
        ranges: &[std::ops::Range<usize>],
    ) -> Result<Vec<u8>> {
        let shape = t.shape();
        anyhow::ensure!(
            shape.len() >= 2,
            "tensor_shard_cols_multi needs a >=2D tensor, got shape {shape:?}"
        );
        let n_rows = shape[0] as usize;
        let n_cols = t.dims[0] as usize;
        let block = t.ty.block_size();
        for r in ranges {
            anyhow::ensure!(
                r.end <= n_cols,
                "shard range {r:?} out of bounds for a {n_cols}-column tensor"
            );
            anyhow::ensure!(
                r.start % block == 0 && r.end % block == 0,
                "column shard range {r:?} is not aligned to {}'s {block}-element block size",
                t.ty
            );
        }
        let type_size = t.ty.type_size();
        let row_bytes = (n_cols / block) * type_size;
        anyhow::ensure!(
            row_bytes * n_rows == t.n_bytes,
            "computed row size {row_bytes} * {n_rows} rows doesn't match tensor byte size {}",
            t.n_bytes
        );
        let full = self.data(t);
        let per_range_bytes: Vec<usize> = ranges
            .iter()
            .map(|r| ((r.end - r.start) / block) * type_size)
            .collect();
        let total_span_bytes: usize = per_range_bytes.iter().sum();
        let mut out = Vec::with_capacity(n_rows * total_span_bytes);
        for row in 0..n_rows {
            for (r, &span) in ranges.iter().zip(&per_range_bytes) {
                let start_bytes = row * row_bytes + (r.start / block) * type_size;
                out.extend_from_slice(&full[start_bytes..start_bytes + span]);
            }
        }
        Ok(out)
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
