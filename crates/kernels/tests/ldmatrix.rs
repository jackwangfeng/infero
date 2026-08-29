//! Does `ldmatrix` load the fragment the MMA expects?
//!
//! `ldmatrix` replaces four scalar shared-memory loads per lane with one
//! cooperative instruction, which is what llama.cpp's MMQ uses to feed its
//! tensor cores. The address each lane supplies is not the address of the data
//! it receives — the hardware permutes — so the mapping is easy to get subtly
//! wrong, and a wrong fragment is not an error, just a wrong matrix product.
//! Compare it against the gather it replaces, element for element.

use anyhow::Result;
use infero_cuda::Device;
use infero_kernels::Kernels;

#[test]
fn ldmatrix_loads_the_same_fragment_as_the_scalar_gather() -> Result<()> {
    let dev = match Device::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping: no cuda device ({e})");
            return Ok(());
        }
    };
    if dev.arch() < 80 {
        eprintln!("skipping: sm_{} predates the int8 mma", dev.arch());
        return Ok(());
    }
    let kern = Kernels::new(dev.clone());
    let stream = dev.stream();

    // Distinct values everywhere, so a permutation cannot pass by coincidence.
    let a: Vec<i8> = (0..16 * 32).map(|i| ((i * 5 + 11) % 251 - 125) as i8).collect();
    let da = stream.clone_htod(&a)?;
    let mut dout = stream.alloc_zeros::<i32>(32 * 8)?;
    kern.ldmatrix_probe(&mut dout.slice_mut(..), &da.slice(..))?;
    let out = stream.clone_dtoh(&dout)?;
    dev.synchronize()?;

    let mut bad = 0;
    for lane in 0..32 {
        for i in 0..4 {
            let got = out[lane * 8 + i];
            let want = out[lane * 8 + 4 + i];
            if got != want {
                if bad < 6 {
                    eprintln!("  lane {lane} reg {i}: ldmatrix {got:#010x}, gather {want:#010x}");
                }
                bad += 1;
            }
        }
    }
    assert_eq!(bad, 0, "{bad} of 128 fragment registers differ");
    eprintln!("  all 128 fragment registers match the scalar gather");
    Ok(())
}

/// The B operand takes `ldmatrix.x2` rather than `.x4` and addresses a
/// different tile shape, so it gets its own check.
#[test]
fn ldmatrix_b_loads_the_same_fragment_as_the_scalar_gather() -> Result<()> {
    let dev = match Device::new(0) {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    if dev.arch() < 80 {
        return Ok(());
    }
    let kern = Kernels::new(dev.clone());
    let stream = dev.stream();

    let b: Vec<i8> = (0..8 * 32).map(|i| ((i * 7 + 3) % 251 - 125) as i8).collect();
    let db = stream.clone_htod(&b)?;
    let mut dout = stream.alloc_zeros::<i32>(32 * 4)?;
    kern.ldmatrix_b_probe(&mut dout.slice_mut(..), &db.slice(..))?;
    let out = stream.clone_dtoh(&dout)?;
    dev.synchronize()?;

    let mut bad = 0;
    for lane in 0..32 {
        for i in 0..2 {
            if out[lane * 4 + i] != out[lane * 4 + 2 + i] {
                if bad < 6 {
                    eprintln!(
                        "  lane {lane} reg {i}: ldmatrix {:#010x}, gather {:#010x}",
                        out[lane * 4 + i],
                        out[lane * 4 + 2 + i]
                    );
                }
                bad += 1;
            }
        }
    }
    assert_eq!(bad, 0, "{bad} of 64 B-fragment registers differ");
    eprintln!("  all 64 B-fragment registers match the scalar gather");
    Ok(())
}
