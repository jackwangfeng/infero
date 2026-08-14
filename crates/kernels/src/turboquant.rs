//! Host-side setup for TurboQuant (Zandieh et al., arXiv:2504.19874).
//!
//! Everything here is computed once at load time and never depends on the
//! data — that is the property that makes the scheme usable for a KV cache,
//! where vectors arrive one token at a time and there is nothing to calibrate
//! against.
//!
//! Three pieces:
//!
//! * a random rotation `Π`, which turns an arbitrary unit vector into one
//!   uniform on the sphere, so its coordinates follow a *known* density
//!   regardless of the input (Lemma 1 of the paper);
//! * the Lloyd-Max codebook for that density, solved numerically (Eq. 4);
//! * a Gaussian projection `S` for the 1-bit QJL stage that removes the
//!   inner-product bias an MSE-optimal quantizer necessarily has.
//!
//! All three come from a fixed seed, so a cache quantized in one process
//! decodes identically in another.

use anyhow::{Context, Result, ensure};

/// The seed behind `Π` and `S`. Changing it changes the quantization, so it is
/// pinned rather than drawn at startup.
pub const DEFAULT_SEED: u64 = 0x7175_616e_7431_3233;

/// Grid resolution for the numerical integration behind the codebook. The
/// density is concentrated within a few multiples of `1/sqrt(d)`, so this
/// leaves thousands of samples per standard deviation for any realistic head
/// dimension.
const GRID: usize = 1 << 18;

/// Deterministic split-mix RNG. A fixed, self-contained generator keeps the
/// tables reproducible across platforms and crate versions.
struct SplitMix(u64);

impl SplitMix {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in (0, 1).
    fn next_f64(&mut self) -> f64 {
        // Shift into the 53 bits a double can hold exactly, and keep it off
        // zero so the log in Box-Muller stays finite.
        let bits = self.next_u64() >> 11;
        (bits as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// Standard normal, via Box-Muller.
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64();
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// The precomputed tables TurboQuant needs, in the layout the kernels want.
///
/// Matrices are stored **column-major** (`m[j * d + i]` is row `i`, column
/// `j`) so that a mat-vec's inner loop reads consecutive addresses across
/// threads.
pub struct Tables {
    pub d: usize,
    /// `Π`, the random rotation.
    pub rotation: Vec<f32>,
    /// `Πᵀ`, for mapping the attention output back out of the rotated space.
    pub rotation_t: Vec<f32>,
    /// `S' = S·Πᵀ`, the QJL projection folded into the rotated basis so that
    /// cached vectors never have to be rotated back.
    pub qjl: Vec<f32>,
}

impl Tables {
    pub fn new(d: usize, seed: u64) -> Result<Self> {
        ensure!(
            d > 0 && d.is_multiple_of(8),
            "head dimension {d} must be a multiple of 8"
        );

        let rotation = random_rotation(d, seed);
        let rotation_t = transpose(&rotation, d);

        // S has i.i.d. N(0,1) entries; Π is orthogonal, so S·Πᵀ does too. The
        // substitution is what lets `⟨S·q, qjl⟩` be evaluated as
        // `⟨S'·(Π·q), qjl⟩` with the query rotated only once.
        let s = gaussian_matrix(d, seed ^ 0x5151_5151_5151_5151);
        let qjl = matmul_t(&s, &rotation, d);

        Ok(Self {
            d,
            rotation: to_column_major(&rotation, d),
            rotation_t: to_column_major(&rotation_t, d),
            qjl: to_column_major(&qjl, d),
        })
    }
}

/// A Lloyd-Max codebook for one coordinate of a randomly rotated unit vector.
#[derive(Debug, Clone)]
pub struct Codebook {
    pub bits: u8,
    /// `2^bits` centroids in ascending order, in units of the vector's norm.
    pub levels: Vec<f32>,
    /// `d · C(f_X, b)`: the expected squared error of quantizing a unit
    /// vector this way. Comparable to the paper's `D_mse`.
    pub distortion: f64,
}

impl Codebook {
    /// Solve Eq. (4) — the continuous k-means problem on the coordinate
    /// density of Lemma 1 — for head dimension `d` at `bits` bits.
    pub fn solve(d: usize, bits: u8) -> Result<Self> {
        ensure!(
            (1..=8).contains(&bits),
            "bit-width {bits} out of range 1..=8"
        );
        ensure!(d >= 2, "head dimension must be at least 2");

        let n_levels = 1usize << bits;
        let grid = Grid::new(d);

        // Start from the quantiles of the density: a good initialization keeps
        // Lloyd from settling into a lopsided local optimum at high bit-widths.
        let mut levels: Vec<f64> = (0..n_levels)
            .map(|i| grid.quantile((i as f64 + 0.5) / n_levels as f64))
            .collect();

        for _ in 0..2000 {
            // Voronoi boundaries are the midpoints between adjacent centroids.
            let mut edges = Vec::with_capacity(n_levels + 1);
            edges.push(-1.0f64);
            for i in 0..n_levels - 1 {
                edges.push((levels[i] + levels[i + 1]) / 2.0);
            }
            edges.push(1.0);

            let mut moved: f64 = 0.0;
            for i in 0..n_levels {
                let (mass, first_moment) = grid.moments(edges[i], edges[i + 1]);
                if mass > 1e-300 {
                    let c = first_moment / mass;
                    moved = moved.max((c - levels[i]).abs());
                    levels[i] = c;
                }
                // An empty cell keeps its centroid; the quantile init makes
                // this essentially unreachable, and moving it would only
                // create a different empty cell.
            }
            // The density is symmetric, so the optimal codebook is too.
            // Enforcing it directly costs nothing and removes the slow, noisy
            // last phase where Lloyd inches toward symmetry on its own.
            for i in 0..n_levels / 2 {
                let j = n_levels - 1 - i;
                let half = (levels[i] - levels[j]) / 2.0;
                levels[i] = half;
                levels[j] = -half;
            }
            if n_levels % 2 == 1 {
                levels[n_levels / 2] = 0.0;
            }

            if moved < 1e-14 {
                break;
            }
        }

        let distortion = d as f64 * grid.distortion(&levels);

        Ok(Self {
            bits,
            levels: levels.iter().map(|v| *v as f32).collect(),
            distortion,
        })
    }

    pub fn len(&self) -> usize {
        self.levels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.levels.is_empty()
    }

    /// Quantize a single coordinate of a unit vector. Only used by the CPU
    /// reference in tests; the kernels do this inline.
    pub fn nearest(&self, z: f32) -> usize {
        let mut best = (0usize, f32::INFINITY);
        for (i, c) in self.levels.iter().enumerate() {
            let e = (z - c).abs();
            if e < best.1 {
                best = (i, e);
            }
        }
        best.0
    }
}

/// The coordinate density of Lemma 1, tabulated with its running integrals.
///
/// `f_X(x) ∝ (1 - x²)^((d-3)/2)` on `[-1, 1]`. For `d` in the hundreds this
/// underflows outside a narrow band, so it is built in log space and
/// normalized numerically rather than through its Gamma-function constant.
struct Grid {
    xs: Vec<f64>,
    /// Prefix sums of `f`, `x·f` and `x²·f` over the grid cells.
    mass: Vec<f64>,
    moment1: Vec<f64>,
    moment2: Vec<f64>,
}

impl Grid {
    fn new(d: usize) -> Self {
        let n = GRID;
        let step = 2.0 / n as f64;
        let exponent = (d as f64 - 3.0) / 2.0;

        let mut xs = Vec::with_capacity(n);
        let mut logf = Vec::with_capacity(n);
        for i in 0..n {
            // Cell midpoints, so the endpoints where (1-x²) vanishes are never
            // evaluated directly.
            let x = -1.0 + (i as f64 + 0.5) * step;
            xs.push(x);
            logf.push(exponent * (1.0 - x * x).ln());
        }

        let peak = logf.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut mass = Vec::with_capacity(n + 1);
        let mut moment1 = Vec::with_capacity(n + 1);
        let mut moment2 = Vec::with_capacity(n + 1);
        mass.push(0.0);
        moment1.push(0.0);
        moment2.push(0.0);

        let mut total = 0.0;
        for i in 0..n {
            let w = (logf[i] - peak).exp() * step;
            total += w;
            mass.push(total);
            moment1.push(moment1[i] + w * xs[i]);
            moment2.push(moment2[i] + w * xs[i] * xs[i]);
        }

        // Normalize so `mass` is a proper CDF.
        for v in mass.iter_mut() {
            *v /= total;
        }
        for v in moment1.iter_mut() {
            *v /= total;
        }
        for v in moment2.iter_mut() {
            *v /= total;
        }

        Self {
            xs,
            mass,
            moment1,
            moment2,
        }
    }

    fn index_of(&self, x: f64) -> usize {
        let n = self.xs.len();
        if x <= -1.0 {
            return 0;
        }
        if x >= 1.0 {
            return n;
        }
        let step = 2.0 / n as f64;
        (((x + 1.0) / step).round() as usize).min(n)
    }

    /// `(∫ f, ∫ x·f)` over `[lo, hi]`.
    fn moments(&self, lo: f64, hi: f64) -> (f64, f64) {
        let (a, b) = (self.index_of(lo), self.index_of(hi));
        if b <= a {
            return (0.0, 0.0);
        }
        (
            self.mass[b] - self.mass[a],
            self.moment1[b] - self.moment1[a],
        )
    }

    /// `Σ_i ∫_cell (x - c_i)² f dx`, the optimal scalar quantization cost.
    fn distortion(&self, levels: &[f64]) -> f64 {
        let n = levels.len();
        let mut edges = Vec::with_capacity(n + 1);
        edges.push(-1.0f64);
        for i in 0..n - 1 {
            edges.push((levels[i] + levels[i + 1]) / 2.0);
        }
        edges.push(1.0);

        let mut total = 0.0;
        for i in 0..n {
            let (a, b) = (self.index_of(edges[i]), self.index_of(edges[i + 1]));
            if b <= a {
                continue;
            }
            let m0 = self.mass[b] - self.mass[a];
            let m1 = self.moment1[b] - self.moment1[a];
            let m2 = self.moment2[b] - self.moment2[a];
            // ∫(x-c)²f = ∫x²f - 2c∫xf + c²∫f
            total += m2 - 2.0 * levels[i] * m1 + levels[i] * levels[i] * m0;
        }
        total
    }

    fn quantile(&self, p: f64) -> f64 {
        let target = p.clamp(0.0, 1.0);
        // The CDF is monotone, so a binary search is exact to the grid.
        let mut lo = 0usize;
        let mut hi = self.mass.len() - 1;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.mass[mid] < target {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        self.xs[lo.min(self.xs.len() - 1)]
    }
}

// ---- matrix helpers -----------------------------------------------------

fn gaussian_matrix(d: usize, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix::new(seed);
    (0..d * d).map(|_| rng.next_normal() as f32).collect()
}

/// A random rotation, as the Q factor of a Gaussian matrix.
///
/// Modified Gram-Schmidt rather than the classical form: at `d = 128` the
/// classical version loses enough orthogonality to show up in the distortion.
fn random_rotation(d: usize, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix::new(seed);
    // Columns of a Gaussian matrix, orthonormalized in place.
    let mut cols: Vec<Vec<f64>> = (0..d)
        .map(|_| (0..d).map(|_| rng.next_normal()).collect())
        .collect();

    for i in 0..d {
        let norm = cols[i].iter().map(|v| v * v).sum::<f64>().sqrt();
        for v in cols[i].iter_mut() {
            *v /= norm;
        }
        for j in i + 1..d {
            let dot: f64 = (0..d).map(|k| cols[i][k] * cols[j][k]).sum();
            let (before, after) = cols.split_at_mut(j);
            for (dst, src) in after[0].iter_mut().zip(&before[i]) {
                *dst -= dot * src;
            }
        }
    }

    // Row-major: element (row, col).
    let mut m = vec![0.0f32; d * d];
    for (c, col) in cols.iter().enumerate() {
        for (r, v) in col.iter().enumerate() {
            m[r * d + c] = *v as f32;
        }
    }
    m
}

fn transpose(m: &[f32], d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; d * d];
    for i in 0..d {
        for j in 0..d {
            out[j * d + i] = m[i * d + j];
        }
    }
    out
}

/// `A · Bᵀ` for row-major `d × d` inputs.
fn matmul_t(a: &[f32], b: &[f32], d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; d * d];
    for i in 0..d {
        for j in 0..d {
            let mut acc = 0.0f64;
            for k in 0..d {
                acc += a[i * d + k] as f64 * b[j * d + k] as f64;
            }
            out[i * d + j] = acc as f32;
        }
    }
    out
}

/// Row-major to column-major, so a mat-vec reads coalesced.
fn to_column_major(m: &[f32], d: usize) -> Vec<f32> {
    transpose(m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    const D: usize = 64;

    #[test]
    fn rotation_is_orthogonal() {
        let t = Tables::new(D, DEFAULT_SEED).unwrap();
        // Stored column-major, so rotation[j*D + i] is Π_ij.
        for i in 0..D {
            for j in 0..D {
                let dot: f64 = (0..D)
                    .map(|k| t.rotation[k * D + i] as f64 * t.rotation[k * D + j] as f64)
                    .sum();
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - want).abs() < 1e-5,
                    "ΠΠᵀ[{i},{j}] = {dot}, expected {want}"
                );
            }
        }
    }

    #[test]
    fn transpose_table_really_is_the_inverse() {
        let t = Tables::new(D, DEFAULT_SEED).unwrap();
        for i in 0..D {
            for j in 0..D {
                assert_eq!(t.rotation[j * D + i], t.rotation_t[i * D + j]);
            }
        }
    }

    /// The paper states the optimal centroids for b = 1, 2 in the large-`d`
    /// limit: `{±√(2/π)/√d}` and `{±0.453/√d, ±1.51/√d}`.
    #[test]
    fn codebook_matches_the_published_values() {
        let d = 4096; // large enough for the Beta density to look Gaussian
        let s = (d as f64).sqrt();

        let b1 = Codebook::solve(d, 1).unwrap();
        let want = (2.0 / std::f64::consts::PI).sqrt() / s;
        assert!(
            (b1.levels[1] as f64 - want).abs() / want < 0.01,
            "b=1 level {} vs {want}",
            b1.levels[1]
        );
        assert!(
            (b1.levels[0] + b1.levels[1]).abs() < 1e-6,
            "should be symmetric"
        );

        let b2 = Codebook::solve(d, 2).unwrap();
        for (got, want) in [
            (b2.levels[2] as f64, 0.4528 / s),
            (b2.levels[3] as f64, 1.5104 / s),
        ] {
            assert!(
                (got - want).abs() / want < 0.01,
                "b=2 level {got} vs {want}"
            );
        }
    }

    /// Theorem 1 quotes `D_mse ≈ 0.36, 0.117, 0.03, 0.009` for b = 1..4, which
    /// are Max's classical Lloyd-Max distortions for a unit-variance Gaussian
    /// rounded to one or two significant figures. Since the density converges
    /// to a Gaussian as `d` grows, the full-precision table is the sharper
    /// target — matching it to four figures is what actually confirms the
    /// density, the k-means solve and the `d·C(f_X,b)` scaling.
    #[test]
    fn distortion_matches_theorem_1() {
        let d = 4096;
        // Max (1960), Table I: MSE of the optimal N-level Gaussian quantizer.
        for (bits, exact, quoted) in [
            (1u8, 0.3634, 0.36),
            (2, 0.1175, 0.117),
            (3, 0.03454, 0.03),
            (4, 0.009497, 0.009),
        ] {
            let cb = Codebook::solve(d, bits).unwrap();
            let rel = (cb.distortion - exact).abs() / exact;
            eprintln!(
                "  b={bits}  D_mse={:.5}  exact={exact}  paper quotes {quoted}",
                cb.distortion
            );
            assert!(
                rel < 0.01,
                "b={bits}: D_mse = {:.5}, Lloyd-Max table says {exact}",
                cb.distortion
            );
        }
    }

    /// Distortion must fall by roughly `1/4` per extra bit, the rate the
    /// `1/4^b` bound describes.
    #[test]
    fn each_extra_bit_quarters_the_distortion() {
        let mut prev = f64::INFINITY;
        for bits in 1..=6u8 {
            let cb = Codebook::solve(D, bits).unwrap();
            if prev.is_finite() {
                let ratio = prev / cb.distortion;
                assert!(
                    (2.5..6.0).contains(&ratio),
                    "b={bits}: distortion fell by {ratio:.2}x, expected about 4x"
                );
            }
            prev = cb.distortion;
        }
    }

    #[test]
    fn codebook_is_sorted_and_symmetric() {
        for bits in 1..=6u8 {
            let cb = Codebook::solve(D, bits).unwrap();
            assert_eq!(cb.len(), 1 << bits);
            for w in cb.levels.windows(2) {
                assert!(w[0] < w[1], "b={bits}: levels are not ascending");
            }
            for i in 0..cb.len() / 2 {
                let lo = cb.levels[i];
                let hi = cb.levels[cb.len() - 1 - i];
                assert!((lo + hi).abs() < 1e-5, "b={bits}: not symmetric at {i}");
            }
        }
    }

    /// The head dimension we actually run: the density is measurably not
    /// Gaussian at d = 64, so the codebook should differ from the asymptotic
    /// one — that difference is the reason to solve it per dimension.
    #[test]
    fn small_d_codebook_differs_from_the_asymptotic_one() {
        let small = Codebook::solve(64, 4).unwrap();
        let large = Codebook::solve(8192, 4).unwrap();
        let rescale = (64.0f32 / 8192.0).sqrt();
        let diff: f32 = small
            .levels
            .iter()
            .zip(&large.levels)
            .map(|(a, b)| (a - b / rescale).abs())
            .fold(0.0, f32::max);
        assert!(
            diff > 1e-4,
            "expected the d=64 codebook to differ, max diff {diff}"
        );
    }

    #[test]
    fn tables_are_reproducible() {
        let a = Tables::new(D, DEFAULT_SEED).unwrap();
        let b = Tables::new(D, DEFAULT_SEED).unwrap();
        assert_eq!(a.rotation, b.rotation);
        assert_eq!(a.qjl, b.qjl);
        let c = Tables::new(D, DEFAULT_SEED + 1).unwrap();
        assert_ne!(a.rotation, c.rotation);
    }

    /// `S' = S·Πᵀ` has to keep the i.i.d. standard normal marginals that the
    /// QJL guarantee depends on.
    #[test]
    fn qjl_projection_stays_standard_normal() {
        let t = Tables::new(256, DEFAULT_SEED).unwrap();
        let n = t.qjl.len() as f64;
        let mean: f64 = t.qjl.iter().map(|v| *v as f64).sum::<f64>() / n;
        let var: f64 = t
            .qjl
            .iter()
            .map(|v| (*v as f64 - mean).powi(2))
            .sum::<f64>()
            / n;
        assert!(mean.abs() < 0.02, "mean {mean}");
        assert!((var - 1.0).abs() < 0.05, "variance {var}");
    }
}

// ---- device side --------------------------------------------------------

use cudarc::driver::CudaSlice;
use tuili_cuda::Device;

/// How the KV cache is stored.
///
/// Keys and values carry independent bit-widths because they are not
/// interchangeable: a key is consumed by an inner product that feeds a
/// softmax, a value by a weighted average. Which side a bit buys more quality
/// on is a property of the model, so it is a knob rather than a constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvQuant {
    /// Bits for the MSE stage of keys. 16 means "not quantized".
    k_bits: u8,
    v_bits: u8,
    /// Whether keys get the second, bias-removing QJL stage of Algorithm 2.
    qjl: bool,
}

impl KvQuant {
    /// Dense f16, no quantization.
    #[allow(non_upper_case_globals)]
    pub const F16: Self = Self {
        k_bits: 16,
        v_bits: 16,
        qjl: false,
    };
    /// Keys 2-bit MSE + 1-bit QJL, values 2-bit. The paper's aggressive point.
    #[allow(non_upper_case_globals)]
    pub const Tq2: Self = Self {
        k_bits: 2,
        v_bits: 2,
        qjl: true,
    };
    /// Keys 4-bit MSE + 1-bit QJL, values 4-bit.
    #[allow(non_upper_case_globals)]
    pub const Tq4: Self = Self {
        k_bits: 4,
        v_bits: 4,
        qjl: true,
    };
    /// Far past the useful compression range; it exists so that a quality
    /// regression can be attributed to the bit-width rather than the plumbing.
    #[allow(non_upper_case_globals)]
    pub const Tq8: Self = Self {
        k_bits: 8,
        v_bits: 8,
        qjl: true,
    };
    /// `Tq4` with the QJL stage switched off — TurboQuant_mse on both sides.
    #[allow(non_upper_case_globals)]
    pub const Tq4Mse: Self = Self {
        k_bits: 4,
        v_bits: 4,
        qjl: false,
    };
    /// `Tq2` with the QJL stage switched off.
    #[allow(non_upper_case_globals)]
    pub const Tq2Mse: Self = Self {
        k_bits: 2,
        v_bits: 2,
        qjl: false,
    };

    /// An explicit split, for isolating which side a loss comes from.
    ///
    /// Both widths must be quantized: the cache stores a sequence in one
    /// encoding, so mixing a dense side with a packed one is not a shape it
    /// can hold.
    pub fn new(k_bits: u8, v_bits: u8, qjl: bool) -> Result<Self> {
        ensure!(
            matches!(k_bits, 2 | 4 | 8) && matches!(v_bits, 2 | 4 | 8),
            "bit-widths must be 2, 4 or 8 (got k{k_bits} v{v_bits}); 16 would \
             mean a dense side, which the cache cannot mix with a packed one"
        );
        Ok(Self {
            k_bits,
            v_bits,
            qjl,
        })
    }

    /// Accepts the named presets, or `k<bits>v<bits>[+qjl]` for anything else
    /// (`k4v8`, `k2v4+qjl`).
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim().to_ascii_lowercase();
        Ok(match s.as_str() {
            "f16" | "none" | "off" => Self::F16,
            "tq2" | "turboquant2" => Self::Tq2,
            "tq4" | "turboquant4" => Self::Tq4,
            "tq8" | "turboquant8" => Self::Tq8,
            "tq4-mse" | "tq4mse" => Self::Tq4Mse,
            "tq2-mse" | "tq2mse" => Self::Tq2Mse,
            other => return Self::parse_split(other),
        })
    }

    fn parse_split(s: &str) -> Result<Self> {
        let (spec, qjl) = match s.strip_suffix("+qjl") {
            Some(rest) => (rest, true),
            None => (s, false),
        };
        let body = spec.strip_prefix('k').ok_or_else(|| {
            anyhow::anyhow!(
                "unknown kv quantization `{s}`; expected f16, tq2, tq4, tq8, \
                 tq2-mse, tq4-mse, or k<bits>v<bits>[+qjl]"
            )
        })?;
        let (k, v) = body
            .split_once('v')
            .context("expected the form k<bits>v<bits>")?;
        let k: u8 = k.parse().context("key bit-width")?;
        let v: u8 = v.parse().context("value bit-width")?;
        Self::new(k, v, qjl)
    }

    /// True when either side is compressed.
    pub fn is_quantized(self) -> bool {
        self.k_bits < 16 || self.v_bits < 16
    }

    /// Bits of the MSE stage for keys. Keys spend one further bit on QJL.
    pub fn k_mse_bits(self) -> u8 {
        self.k_bits
    }

    pub fn v_bits(self) -> u8 {
        self.v_bits
    }

    /// Whether keys get the second, bias-removing QJL stage.
    pub fn uses_qjl(self) -> bool {
        self.qjl && self.k_bits < 16
    }

    /// The multiplier the score kernel applies to the QJL term: exactly one or
    /// exactly zero, so the ablation is the estimator the paper compares
    /// against rather than an approximation of it.
    pub fn qjl_scale(self) -> f32 {
        if self.uses_qjl() { 1.0 } else { 0.0 }
    }

    /// Average bits per channel across K and V, counting the per-vector norms
    /// that the codes alone leave out.
    ///
    /// Those norms are the reason the effective rate is above the nominal one,
    /// and the gap widens as `d` shrinks: a 64-dimensional head amortizes them
    /// over half as many channels as the 128-dimensional heads the paper uses.
    pub fn bits_per_channel(self, d: usize) -> f32 {
        if !self.is_quantized() {
            return 16.0;
        }
        let d = d as f32;
        let k = if self.k_bits >= 16 {
            16.0
        } else {
            let qjl = if self.uses_qjl() { 1.0 } else { 0.0 };
            let norms = if self.uses_qjl() { 32.0 } else { 16.0 };
            self.k_bits as f32 + qjl + norms / d
        };
        let v = if self.v_bits >= 16 {
            16.0
        } else {
            self.v_bits as f32 + 16.0 / d
        };
        (k + v) / 2.0
    }

    pub fn name(self) -> String {
        match self {
            s if s == Self::F16 => "f16".into(),
            s if s == Self::Tq2 => "tq2".into(),
            s if s == Self::Tq4 => "tq4".into(),
            s if s == Self::Tq8 => "tq8".into(),
            s if s == Self::Tq4Mse => "tq4-mse".into(),
            s if s == Self::Tq2Mse => "tq2-mse".into(),
            s => format!(
                "k{}v{}{}",
                s.k_bits,
                s.v_bits,
                if s.qjl { "+qjl" } else { "" }
            ),
        }
    }
}

impl std::fmt::Display for KvQuant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.name())
    }
}

/// The tables uploaded to the device, plus the codebooks they were built from.
pub struct DeviceTables {
    /// The codebooks and matrices are public so tests can decode independently.
    pub quant: KvQuant,
    pub d: usize,
    pub rotation: CudaSlice<f32>,
    pub rotation_t: CudaSlice<f32>,
    pub qjl: CudaSlice<f32>,
    pub k_levels: CudaSlice<f32>,
    pub v_levels: CudaSlice<f32>,
    pub k_codebook: Codebook,
    pub v_codebook: Codebook,
}

impl DeviceTables {
    pub fn new(dev: &Device, d: usize, quant: KvQuant) -> Result<Self> {
        ensure!(
            quant.is_quantized(),
            "DeviceTables is only for a quantized cache"
        );
        ensure!(
            d <= 256,
            "head dimension {d} exceeds the 256 the kernels stage in shared memory"
        );
        ensure!(
            d.is_multiple_of(8),
            "head dimension {d} must be a multiple of 8"
        );

        let started = std::time::Instant::now();
        let tables = Tables::new(d, DEFAULT_SEED)?;
        let k_codebook = Codebook::solve(d, quant.k_mse_bits())?;
        let v_codebook = Codebook::solve(d, quant.v_bits())?;

        let stream = dev.stream();
        let this = Self {
            quant,
            d,
            rotation: stream.clone_htod(&tables.rotation)?,
            rotation_t: stream.clone_htod(&tables.rotation_t)?,
            qjl: stream.clone_htod(&tables.qjl)?,
            k_levels: stream.clone_htod(&k_codebook.levels)?,
            v_levels: stream.clone_htod(&v_codebook.levels)?,
            k_codebook,
            v_codebook,
        };
        dev.synchronize()?;

        tracing::info!(
            quant = %quant,
            d,
            k_mse_distortion = format!("{:.4}", this.k_codebook.distortion),
            v_mse_distortion = format!("{:.4}", this.v_codebook.distortion),
            bits_per_channel = format!("{:.2}", quant.bits_per_channel(d)),
            ms = started.elapsed().as_millis(),
            "turboquant tables ready"
        );
        Ok(this)
    }

    /// Bytes one cached key vector occupies: packed codes, QJL signs, and the
    /// two f16 norms.
    pub fn key_bytes(&self) -> usize {
        self.d * self.quant.k_mse_bits() as usize / 8 + self.d / 8 + 4
    }

    pub fn value_bytes(&self) -> usize {
        self.d * self.quant.v_bits() as usize / 8 + 2
    }
}
