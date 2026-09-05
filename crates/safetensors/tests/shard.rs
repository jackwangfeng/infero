//! `Tensor::shard_rows`/`shard_cols` must reconstruct exactly the bytes a full
//! read's own slice would give — the property every tensor-parallel loading
//! path this session built (GGUF and safetensors alike) depends on, and the
//! exact class of bug that's twice turned out to be wrong in adjacent code
//! (a naive per-element-width column slice silently corrupting a quantized
//! block; a contiguous shard silently mixing tiled value heads). Distinct
//! values per row/column make a wrong stride or a transposed read visible
//! immediately rather than passing by coincidence.

use anyhow::Result;
use infero_safetensors::{Dtype, Tensor};

fn tensor<'a>(name: &'a str, dtype: Dtype, shape: Vec<usize>, data: &'a [u8]) -> Tensor<'a> {
    Tensor { name, dtype, shape, data }
}

/// F32 so each element is easy to construct and compare exactly.
fn f32_matrix(rows: usize, cols: usize) -> Vec<u8> {
    (0..rows * cols)
        .flat_map(|i| (i as f32).to_le_bytes())
        .collect()
}

#[test]
fn shard_rows_matches_the_corresponding_slice_of_a_full_read() -> Result<()> {
    let (rows, cols) = (10usize, 4usize);
    let data = f32_matrix(rows, cols);
    let t = tensor("w", Dtype::F32, vec![rows, cols], &data);
    let row_bytes = cols * 4;

    for (start, end) in [(0, 3), (3, 7), (7, 10), (0, 10)] {
        let shard = t.shard_rows(start..end)?;
        let want = &data[start * row_bytes..end * row_bytes];
        assert_eq!(shard, want, "rows {start}..{end}");
    }
    Ok(())
}

#[test]
fn shard_rows_out_of_bounds_is_refused() {
    let data = f32_matrix(4, 4);
    let t = tensor("w", Dtype::F32, vec![4, 4], &data);
    assert!(t.shard_rows(2..5).is_err(), "5 rows requested from a 4-row tensor must fail");
}

#[test]
fn shard_cols_matches_the_corresponding_slice_of_a_full_read() -> Result<()> {
    let (rows, cols) = (5usize, 8usize);
    let data = f32_matrix(rows, cols);
    let t = tensor("w", Dtype::F32, vec![rows, cols], &data);

    for (start, end) in [(0, 3), (3, 5), (5, 8), (0, 8)] {
        let shard = t.shard_cols(start..end)?;
        // Column-shard is not a contiguous slice of the source (strided across
        // rows), so reconstruct the expected bytes the same way a full read
        // sliced per-row would, and compare against that -- not against a raw
        // byte-offset slice of `data`, which would be wrong by construction.
        let mut want = Vec::new();
        for r in 0..rows {
            let row_start = (r * cols + start) * 4;
            let row_end = (r * cols + end) * 4;
            want.extend_from_slice(&data[row_start..row_end]);
        }
        assert_eq!(shard, want, "cols {start}..{end}");
    }
    Ok(())
}

#[test]
fn shard_cols_out_of_bounds_is_refused() {
    let data = f32_matrix(4, 4);
    let t = tensor("w", Dtype::F32, vec![4, 4], &data);
    assert!(t.shard_cols(2..5).is_err(), "5 cols requested from a 4-col tensor must fail");
}

/// Two ranks' shards, concatenated, must equal the full tensor -- the actual
/// property tensor-parallel loading relies on (rank 0's shard plus rank 1's
/// shard covers every byte exactly once, no gap, no overlap).
#[test]
fn two_row_shards_concatenated_reconstruct_the_full_tensor() -> Result<()> {
    let (rows, cols) = (12usize, 3usize);
    let data = f32_matrix(rows, cols);
    let t = tensor("w", Dtype::F32, vec![rows, cols], &data);
    let mid = rows / 2;

    let mut reconstructed = t.shard_rows(0..mid)?.to_vec();
    reconstructed.extend_from_slice(t.shard_rows(mid..rows)?);
    assert_eq!(reconstructed, data);
    Ok(())
}

#[test]
fn two_col_shards_concatenated_reconstruct_the_full_tensor() -> Result<()> {
    let (rows, cols) = (3usize, 12usize);
    let data = f32_matrix(rows, cols);
    let t = tensor("w", Dtype::F32, vec![rows, cols], &data);
    let mid = cols / 2;

    let left = t.shard_cols(0..mid)?;
    let right = t.shard_cols(mid..cols)?;
    // Interleave back row-by-row (the two shards are each already row-major
    // over their own column range, one row after another).
    let half_row_bytes = mid * 4;
    let mut reconstructed = Vec::with_capacity(data.len());
    for r in 0..rows {
        reconstructed.extend_from_slice(&left[r * half_row_bytes..(r + 1) * half_row_bytes]);
        reconstructed.extend_from_slice(&right[r * half_row_bytes..(r + 1) * half_row_bytes]);
    }
    assert_eq!(reconstructed, data);
    Ok(())
}
