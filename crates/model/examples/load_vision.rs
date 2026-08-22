//! Load only the vision tower from a checkpoint and report what it read.
//!
//! The loader's risk is 333 tensor names and shapes, and the shapes are checked
//! against the config rather than against each other — so the thing worth
//! running is the loader over the real checkpoint, which no unit test can do.
//!
//!   cargo run --release -p tuili-model --example load_vision -- <model-dir>

use anyhow::Result;

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_max_level(tracing::Level::INFO).init();
    let dir = std::env::args().nth(1).expect("usage: load_vision <model-dir>");
    let device: usize = std::env::var("TUILI_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let dev = tuili_cuda::Device::new(device)?;
    let shards = tuili_safetensors::Shards::open_dir(std::path::Path::new(&dir))?;
    let json = shards.json("config.json")?;
    let cfg = tuili_model::Config::from_hf(&json, "vision-probe")?;
    let vc = cfg.vision.expect("this checkpoint's config has no vision_config");
    println!(
        "config: depth {} hidden {} heads {} intermediate {} out_hidden {} grid {}x{}",
        vc.depth,
        vc.hidden,
        vc.heads,
        vc.intermediate,
        vc.out_hidden,
        vc.grid_per_side(),
        vc.grid_per_side()
    );
    println!("placeholders: image {} video {}", vc.image_token, vc.video_token);

    let tower = tuili_model::weights::load_vision(&dev, &shards, &cfg)?
        .expect("the config describes a tower, so the loader must produce one");
    println!("loaded {} MiB", tower.device_bytes >> 20);
    // Building the borrowed view is what the forward pass does, and it is where
    // a missing block would surface as a length mismatch.
    let w = tower.weights();
    println!("blocks: {}", w.blocks.len());
    assert_eq!(w.blocks.len(), vc.depth);
    Ok(())
}
