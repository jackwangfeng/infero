//! Safetensors container parsing, for Hugging Face checkpoints.
//!
//! The format is deliberately trivial: eight bytes of little-endian header
//! length, that many bytes of JSON naming every tensor and its byte range, then
//! the payloads back to back. Like the GGUF reader next door this maps rather
//! than reads, so a weight upload is one `cudaMemcpy` straight out of the page
//! cache.
//!
//! A checkpoint is usually split across several files with a
//! `model.safetensors.index.json` mapping tensor names to shards. [`Shards`]
//! presents the whole set as one namespace.
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! let w = tuili_safetensors::Shards::open_dir("models/llama-3.1-8b-awq")?;
//! let t = w.tensor("model.layers.0.self_attn.q_proj.qweight")?;
//! println!("{:?} {:?} {} bytes", t.dtype, t.shape, t.data.len());
//! # Ok(()) }
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use memmap2::Mmap;

/// The element types a checkpoint can carry.
///
/// Only the ones a quantized checkpoint actually uses are named; anything else
/// is rejected at parse time rather than silently mis-read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    F16,
    BF16,
    I32,
    I64,
    U8,
}

impl Dtype {
    fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "F32" => Self::F32,
            "F16" => Self::F16,
            "BF16" => Self::BF16,
            "I32" | "U32" => Self::I32,
            "I64" | "U64" => Self::I64,
            "U8" | "I8" | "BOOL" => Self::U8,
            other => bail!("unsupported safetensors dtype `{other}`"),
        })
    }

    pub fn size(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::I64 => 8,
            Self::U8 => 1,
        }
    }
}

/// One tensor, borrowed straight out of the mapping.
pub struct Tensor<'a> {
    pub name: &'a str,
    pub dtype: Dtype,
    /// Row-major, as torch writes it.
    pub shape: Vec<usize>,
    pub data: &'a [u8],
}

impl Tensor<'_> {
    pub fn n_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// The payload as `f16` bits, for a tensor that is stored as one.
    pub fn as_f16(&self) -> Result<&[half::f16]> {
        anyhow::ensure!(
            self.dtype == Dtype::F16,
            "{} is {:?}, not F16",
            self.name,
            self.dtype
        );
        // Safety: f16 is a transparent u16, the mapping outlives the slice, and
        // safetensors guarantees the payload is a packed little-endian array.
        // Alignment is checked rather than assumed.
        let (head, body, _) = unsafe { self.data.align_to::<half::f16>() };
        anyhow::ensure!(head.is_empty(), "{} is not 2-byte aligned", self.name);
        Ok(&body[..self.n_elements()])
    }

    /// The payload as `i32`, for a tensor that is stored as one.
    pub fn as_i32(&self) -> Result<&[i32]> {
        anyhow::ensure!(
            self.dtype == Dtype::I32,
            "{} is {:?}, not I32",
            self.name,
            self.dtype
        );
        let (head, body, _) = unsafe { self.data.align_to::<i32>() };
        anyhow::ensure!(head.is_empty(), "{} is not 4-byte aligned", self.name);
        Ok(&body[..self.n_elements()])
    }
}

struct Entry {
    dtype: Dtype,
    shape: Vec<usize>,
    /// Byte range within the file's payload section.
    start: usize,
    end: usize,
}

/// One `.safetensors` file.
pub struct File {
    path: PathBuf,
    map: Mmap,
    /// Where the payload section starts; every entry's range is relative to it.
    data_offset: usize,
    tensors: BTreeMap<String, Entry>,
}

impl File {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file =
            std::fs::File::open(&path).with_context(|| format!("opening {}", path.display()))?;
        // Safety: same contract as the GGUF reader — the mapping is treated as
        // immutable, and a concurrent writer truncating the file would be UB.
        let map =
            unsafe { Mmap::map(&file) }.with_context(|| format!("mmapping {}", path.display()))?;

        anyhow::ensure!(map.len() >= 8, "{} is too short", path.display());
        let header_len = u64::from_le_bytes(map[..8].try_into().unwrap()) as usize;
        let data_offset = 8usize
            .checked_add(header_len)
            .filter(|end| *end <= map.len())
            .with_context(|| format!("{}: header runs past end of file", path.display()))?;

        let header: serde_json::Value = serde_json::from_slice(&map[8..data_offset])
            .with_context(|| format!("parsing {} header", path.display()))?;
        let obj = header
            .as_object()
            .with_context(|| format!("{}: header is not an object", path.display()))?;

        let payload = map.len() - data_offset;
        let mut tensors = BTreeMap::new();
        for (name, v) in obj {
            // Free-form provenance the format allows alongside the tensors.
            if name == "__metadata__" {
                continue;
            }
            let dtype = Dtype::parse(v["dtype"].as_str().with_context(|| format!("{name}: no dtype"))?)
                .with_context(|| format!("reading {name}"))?;
            let shape: Vec<usize> = v["shape"]
                .as_array()
                .with_context(|| format!("{name}: no shape"))?
                .iter()
                .map(|d| d.as_u64().map(|d| d as usize).context("shape is not integral"))
                .collect::<Result<_>>()?;
            let offs = v["data_offsets"]
                .as_array()
                .filter(|a| a.len() == 2)
                .with_context(|| format!("{name}: no data_offsets pair"))?;
            let (start, end) = (
                offs[0].as_u64().context("offset is not integral")? as usize,
                offs[1].as_u64().context("offset is not integral")? as usize,
            );
            anyhow::ensure!(
                start <= end && end <= payload,
                "{name}: range {start}..{end} runs past the {payload}-byte payload"
            );
            let want: usize = shape.iter().product::<usize>() * dtype.size();
            anyhow::ensure!(
                end - start == want,
                "{name}: {:?} of {dtype:?} needs {want} bytes, header gives {}",
                shape,
                end - start
            );
            tensors.insert(name.clone(), Entry { dtype, shape, start, end });
        }

        Ok(Self { path, map, data_offset, tensors })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn get(&self, name: &str) -> Option<Tensor<'_>> {
        let (key, e) = self.tensors.get_key_value(name)?;
        Some(Tensor {
            name: key,
            dtype: e.dtype,
            shape: e.shape.clone(),
            data: &self.map[self.data_offset + e.start..self.data_offset + e.end],
        })
    }
}

/// Every shard of a checkpoint, as one namespace.
pub struct Shards {
    files: Vec<File>,
    /// Tensor name to the index in `files` that holds it.
    owner: BTreeMap<String, usize>,
    dir: PathBuf,
}

impl Shards {
    /// Open every `.safetensors` in a directory.
    ///
    /// The index file is not required: it only says which shard holds what,
    /// which is the same thing each shard's own header says. Reading the
    /// headers instead means a checkpoint with a stale or missing index still
    /// loads.
    pub fn open_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
            .with_context(|| format!("reading {}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
            .collect();
        anyhow::ensure!(
            !paths.is_empty(),
            "no .safetensors files in {}",
            dir.display()
        );
        paths.sort();

        let mut files: Vec<File> = Vec::with_capacity(paths.len());
        let mut owner = BTreeMap::new();
        for path in paths {
            let f = File::open(&path)?;
            let idx = files.len();
            for name in f.names() {
                if let Some(prev) = owner.insert(name.to_string(), idx) {
                    bail!(
                        "`{name}` appears in both {} and {}",
                        files[prev].path().display(),
                        path.display()
                    );
                }
            }
            files.push(f);
        }
        tracing::debug!(
            shards = files.len(),
            tensors = owner.len(),
            dir = %dir.display(),
            "safetensors opened"
        );
        Ok(Self { files, owner, dir })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn len(&self) -> usize {
        self.owner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.owner.is_empty()
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.owner.keys().map(String::as_str)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.owner.contains_key(name)
    }

    pub fn get(&self, name: &str) -> Option<Tensor<'_>> {
        let idx = *self.owner.get(name)?;
        self.files[idx].get(name)
    }

    pub fn tensor(&self, name: &str) -> Result<Tensor<'_>> {
        self.get(name)
            .with_context(|| format!("no tensor `{name}` in {}", self.dir.display()))
    }

    /// Parse a JSON file sitting beside the weights, such as `config.json`.
    pub fn json(&self, file: &str) -> Result<serde_json::Value> {
        let path = self.dir.join(file);
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
    }
}
