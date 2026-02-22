use std::collections::HashMap;
use std::io::{self, Read};
use std::time::Instant;

mod config;
mod domain;
mod gpu;
mod math;
mod render;

use config::{validate_config, IncomingConfig};
use domain::presets::build_scene;
use gpu::GpuRenderer;
use render::validation::validate_scene_against_capabilities;
use render::{gpu_capabilities, RenderSettings, View};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut raw = String::new();
    io::stdin().read_to_string(&mut raw)?;

    let incoming: IncomingConfig = serde_json::from_str(&raw)?;
    let frames = match incoming {
        IncomingConfig::Single(frame) => vec![frame],
        IncomingConfig::Batch(batch) => batch.frames,
    };
    if frames.is_empty() {
        return Err("frames array must not be empty".into());
    }

    let total = frames.len();
    let mut prepared_frames = Vec::with_capacity(total);

    for frame in &frames {
        validate_config(frame)?;
        prepared_frames.push((
            RenderSettings::from_frame(frame),
            View::from_frame(frame),
            frame.scene.clone(),
        ));
    }

    let capabilities = gpu_capabilities();
    let mut scene_cache = HashMap::new();
    let mut gpu_renderer = pollster::block_on(GpuRenderer::new())
        .map_err(|error| format!("GPU initialization failed: {error}"))?;

    for (index, (settings, view, scene_id)) in prepared_frames.iter().enumerate() {
        let cache_key = scene_id.to_ascii_lowercase();
        if !scene_cache.contains_key(&cache_key) {
            let scene = build_scene(scene_id)
                .map_err(|error| format!("Failed to build scene '{}': {error}", scene_id))?;
            validate_scene_against_capabilities(&scene, capabilities).map_err(|error| {
                format!(
                    "Scene '{}' is not supported by current renderer: {error}",
                    scene.id
                )
            })?;
            scene_cache.insert(cache_key.clone(), scene);
        }
        let scene = scene_cache
            .get(&cache_key)
            .ok_or_else(|| format!("internal error: scene cache miss for '{scene_id}'"))?;

        let started = Instant::now();
        let image = gpu_renderer
            .render_frame(settings, view, scene)
            .map_err(|error| format!("GPU render failed: {error}"))?;
        let elapsed_ms = started.elapsed().as_millis();
        image.save(&settings.output_path)?;

        println!(
            "[{}/{}] Rendered scene '{}' [GPU] in {} ms: {}",
            index + 1,
            total,
            scene.id,
            elapsed_ms,
            settings.output_path
        );
    }

    // In this CLI workflow the renderer lifetime matches the process lifetime.
    // Some GPU/driver stacks can crash while tearing down WGPU objects on drop.
    std::mem::forget(gpu_renderer);

    Ok(())
}
