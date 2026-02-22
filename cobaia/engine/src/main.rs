use std::io::{self, Read};
use std::time::Instant;

mod config;
mod cpu;
mod gpu;
mod math;
mod scene;

use config::{validate_config, DebugOptions, IncomingConfig, RenderMode};
use cpu::render_cpu;
use gpu::GpuRenderer;
use scene::Scene;

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

    let scene = Scene::stage4_scene();
    let debug_options = DebugOptions::from_env();
    let total = frames.len();
    let mut gpu_renderer: Option<GpuRenderer> = None;
    let mut gpu_init_error: Option<String> = None;

    for (index, frame) in frames.iter().enumerate() {
        validate_config(frame)?;
        let mode = RenderMode::parse(&frame.renderer_mode);
        let started = Instant::now();
        let image = match mode {
            RenderMode::Cpu => render_cpu(frame, scene, debug_options),
            RenderMode::Gpu => {
                if gpu_renderer.is_none() && gpu_init_error.is_none() {
                    match pollster::block_on(GpuRenderer::new(debug_options)) {
                        Ok(renderer) => {
                            gpu_renderer = Some(renderer);
                        }
                        Err(error) => {
                            gpu_init_error = Some(error);
                        }
                    }
                }

                if let Some(renderer) = gpu_renderer.as_mut() {
                    match renderer.render_frame(frame, scene) {
                        Ok(image) => image,
                        Err(error) => {
                            eprintln!("GPU render failed: {error}. Falling back to CPU.");
                            render_cpu(frame, scene, debug_options)
                        }
                    }
                } else {
                    if let Some(error) = &gpu_init_error {
                        eprintln!("GPU initialization failed: {error}. Falling back to CPU.");
                    }
                    render_cpu(frame, scene, debug_options)
                }
            }
        };
        let elapsed_ms = started.elapsed().as_millis();
        image.save(&frame.output_path)?;

        println!(
            "[{}/{}] Rendered stage 4 frame [{}] in {} ms: {}",
            index + 1,
            total,
            mode.as_str(),
            elapsed_ms,
            frame.output_path
        );
    }

    Ok(())
}
