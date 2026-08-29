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
//! let w = infero_safetensors::Shards::open_dir("models/llama-3.1-8b-awq")?;
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
    /// FP8 with a four-bit exponent and a three-bit mantissa, as an FP8
    /// checkpoint stores its projections. Never meaningful on its own: the
    /// values are held at full E4M3 range and scaled back down by a companion
    /// `weight_scale_inv` tensor, one entry per 128×128 block.
    F8E4M3,
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
            "F8_E4M3" => Self::F8E4M3,
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
            Self::F8E4M3 | Self::U8 => 1,
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

    /// The payload as `f32`, whatever real dtype it is stored as.
    ///
    /// Norm weights and biases are wanted as f32 on the device, and an export
    /// is free to store them in any float width — Qwen3's AWQ checkpoint writes
    /// every float as BF16 (norms, lm_head, and the AWQ scales alike), so a
    /// loader that only accepts F16 rejects the file on `model.norm.weight`
    /// before it reaches a single layer. Converting here keeps the dtype
    /// question out of the model loader, which never wanted the borrowed
    /// `&[f16]` anyway: both call sites mapped it straight to f32.
    pub fn to_f32(&self) -> Result<Vec<f32>> {
        let n = self.n_elements();
        match self.dtype {
            Dtype::F16 => Ok(self.as_f16()?.iter().map(|x| f32::from(*x)).collect()),
            Dtype::BF16 => {
                // Safety: same contract as `as_f16` — a packed little-endian
                // array in a mapping that outlives the slice; alignment checked.
                let (head, body, _) = unsafe { self.data.align_to::<u16>() };
                anyhow::ensure!(head.is_empty(), "{} is not 2-byte aligned", self.name);
                // bf16 is the high half of an f32, so widening is exact.
                Ok(body[..n]
                    .iter()
                    .map(|b| f32::from_bits((*b as u32) << 16))
                    .collect())
            }
            Dtype::F32 => {
                let (head, body, _) = unsafe { self.data.align_to::<f32>() };
                anyhow::ensure!(head.is_empty(), "{} is not 4-byte aligned", self.name);
                Ok(body[..n].to_vec())
            }
            other => bail!("{} is {other:?}, not a float type", self.name),
        }
    }

    /// The payload as `f16`, borrowed when it already is and converted when not.
    ///
    /// The Q8_0 quantizer wants halves, and a big matrix is not worth widening
    /// to f32 just to narrow it again — so this borrows for an F16 checkpoint
    /// and only allocates for a BF16 one.
    ///
    /// bf16 carries f32's exponent range while f16 stops at ±65504, so the
    /// narrowing can overflow. Weights sit far inside that range in practice,
    /// but `f16::from_f32` saturates to infinity rather than complaining, and an
    /// infinity in a projection matrix is the kind of fault that shows up as
    /// plausible-looking output rather than a crash. Refuse instead.
    pub fn to_f16(&self) -> Result<std::borrow::Cow<'_, [half::f16]>> {
        match self.dtype {
            Dtype::F16 => Ok(std::borrow::Cow::Borrowed(self.as_f16()?)),
            Dtype::BF16 => {
                let (head, body, _) = unsafe { self.data.align_to::<u16>() };
                anyhow::ensure!(head.is_empty(), "{} is not 2-byte aligned", self.name);
                let mut out = Vec::with_capacity(self.n_elements());
                for b in &body[..self.n_elements()] {
                    let wide = f32::from_bits((*b as u32) << 16);
                    let narrow = half::f16::from_f32(wide);
                    anyhow::ensure!(
                        narrow.is_finite() || !wide.is_finite(),
                        "{} has a value outside f16 range ({wide:e}); \
                         narrowing it would silently become infinity",
                        self.name
                    );
                    out.push(narrow);
                }
                Ok(std::borrow::Cow::Owned(out))
            }
            other => bail!("{} is {other:?}, not a half-width float", self.name),
        }
    }

    /// Every E4M3 bit pattern, as the f32 it denotes.
    ///
    /// A byte has 256 possible values, so the decode is a table rather than
    /// arithmetic: one lookup and one multiply per element, which matters when
    /// the checkpoint holds 27 billion of them.
    ///
    /// The format is IEEE-shaped — sign, four exponent bits biased by 7, three
    /// mantissa bits — with two departures: `exp == 0` is subnormal at
    /// `(m/8) · 2⁻⁶`, and there are no infinities, so `0x7F` and `0xFF` are
    /// NaN rather than ±∞.
    pub(crate) fn e4m3_table() -> &'static [f32; 256] {
        static T: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
        T.get_or_init(|| {
            let mut t = [0.0f32; 256];
            for (b, slot) in t.iter_mut().enumerate() {
                let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
                let exp = ((b >> 3) & 0x0F) as i32;
                let mant = (b & 0x07) as f32 / 8.0;
                *slot = if exp == 0 {
                    sign * mant * 2f32.powi(-6)
                } else if exp == 0x0F && (b & 0x07) == 0x07 {
                    f32::NAN
                } else {
                    sign * (1.0 + mant) * 2f32.powi(exp - 7)
                };
            }
            t
        })
    }

    /// An FP8 matrix as halves, with its block scales applied.
    ///
    /// `scales` is the checkpoint's companion `weight_scale_inv`: one entry per
    /// `block × block` tile of this tensor, and — despite the name — a
    /// multiplier. The stored bytes carry the full E4M3 range, so the product is
    /// what the weight actually is; dividing instead lands five orders of
    /// magnitude out, which is the check that pins the direction down.
    pub fn dequant_f8_to_f16(&self, scales: &Tensor<'_>, block: usize) -> Result<Vec<half::f16>> {
        anyhow::ensure!(
            self.dtype == Dtype::F8E4M3,
            "{} is {:?}, not F8_E4M3",
            self.name,
            self.dtype
        );
        anyhow::ensure!(
            self.shape.len() == 2 && scales.shape.len() == 2,
            "{}: expected a 2-D matrix and a 2-D scale grid, got {:?} and {:?}",
            self.name,
            self.shape,
            scales.shape
        );
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let (srows, scols) = (scales.shape[0], scales.shape[1]);
        // The grid covers the matrix in ceil-divided tiles; a checkpoint whose
        // last tile is partial is fine, one whose grid is the wrong shape is a
        // silent mis-scaling of every block after the first row.
        anyhow::ensure!(
            srows == rows.div_ceil(block) && scols == cols.div_ceil(block),
            "{}: {rows}×{cols} at block {block} wants a {}×{} scale grid, checkpoint has {srows}×{scols}",
            self.name,
            rows.div_ceil(block),
            cols.div_ceil(block),
        );
        let q = self.data;
        anyhow::ensure!(
            q.len() >= rows * cols,
            "{}: {} bytes for {rows}×{cols}",
            self.name,
            q.len()
        );
        let s = scales.to_f32()?;
        let table = Self::e4m3_table();
        let mut out = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            let srow = (r / block) * scols;
            let qrow = r * cols;
            for c in 0..cols {
                let v = table[q[qrow + c] as usize] * s[srow + c / block];
                out.push(half::f16::from_f32(v));
            }
        }
        Ok(out)
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

/// What one E4M3 byte means.
///
/// Exposed so a test or another crate can name the reference value without
/// writing a second decoder — a second decoder is how two implementations come
/// to agree on a reading neither of them confirmed.
pub fn e4m3_value(byte: u8) -> f32 {
    Tensor::e4m3_table()[byte as usize]
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
