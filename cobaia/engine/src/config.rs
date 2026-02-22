use serde::Deserialize;
use std::path::Path;

use crate::math::Vec3;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderFrameConfig {
    pub width: u32,
    pub height: u32,
    pub output_path: String,
    pub max_depth: u8,
    #[serde(default = "default_samples_per_pixel")]
    pub samples_per_pixel: u16,
    pub scene: String,
    pub renderer_mode: String,
    pub camera_origin: [f32; 3],
    pub camera_target: [f32; 3],
    #[serde(default = "default_camera_fov_deg")]
    pub camera_fov_deg: f32,
    #[serde(default = "default_quality")]
    pub quality: String,
    #[serde(default)]
    pub march_max_steps: Option<u16>,
    #[serde(default)]
    pub rr_start_bounce: Option<u8>,
    #[serde(default = "default_sampling_mode")]
    pub sampling_mode: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderBatchConfig {
    pub frames: Vec<RenderFrameConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum IncomingConfig {
    Single(RenderFrameConfig),
    Batch(RenderBatchConfig),
}

const fn default_samples_per_pixel() -> u16 {
    1
}

const fn default_camera_fov_deg() -> f32 {
    38.0
}

fn default_quality() -> String {
    "balanced".to_string()
}

fn default_sampling_mode() -> String {
    "halton".to_string()
}

pub fn validate_config(config: &RenderFrameConfig) -> Result<(), Box<dyn std::error::Error>> {
    if config.width == 0 || config.height == 0 {
        return Err("width and height must be positive".into());
    }

    let output_parent = Path::new(&config.output_path)
        .parent()
        .ok_or("outputPath must include a parent directory")?;

    if !output_parent.exists() {
        return Err(format!(
            "output directory does not exist: {}",
            output_parent.display()
        )
        .into());
    }

    if config.max_depth == 0 {
        return Err("maxDepth must be at least 1".into());
    }

    if config.samples_per_pixel == 0 {
        return Err("samplesPerPixel must be at least 1".into());
    }

    if config.scene.trim().is_empty() {
        return Err("scene must be a non-empty identifier".into());
    }

    if !config.renderer_mode.eq_ignore_ascii_case("gpu") {
        return Err("rendererMode must be \"gpu\"".into());
    }

    if !is_finite_vec3(config.camera_origin) || !is_finite_vec3(config.camera_target) {
        return Err("camera vectors must contain finite values".into());
    }

    let camera_origin = vec3_from(config.camera_origin);
    let camera_target = vec3_from(config.camera_target);
    if (camera_origin - camera_target).length() < 0.0001 {
        return Err("cameraOrigin must differ from cameraTarget".into());
    }
    if !config.camera_fov_deg.is_finite()
        || config.camera_fov_deg <= 1.0
        || config.camera_fov_deg >= 179.0
    {
        return Err("cameraFovDeg must be finite and in (1, 179) degrees".into());
    }
    if !is_quality_valid(&config.quality) {
        return Err("quality must be one of: preview, balanced, final".into());
    }
    if let Some(march_max_steps) = config.march_max_steps {
        if march_max_steps == 0 || march_max_steps > 280 {
            return Err("marchMaxSteps must be in [1, 280]".into());
        }
    }
    if let Some(rr_start_bounce) = config.rr_start_bounce {
        if rr_start_bounce > 32 {
            return Err("rrStartBounce must be in [0, 32]".into());
        }
    }
    if !is_sampling_mode_valid(&config.sampling_mode) {
        return Err("samplingMode must be one of: random, halton, sobol".into());
    }

    Ok(())
}

pub fn vec3_from(value: [f32; 3]) -> Vec3 {
    Vec3::new(value[0], value[1], value[2])
}

fn is_finite_vec3(value: [f32; 3]) -> bool {
    value[0].is_finite() && value[1].is_finite() && value[2].is_finite()
}

fn is_quality_valid(value: &str) -> bool {
    value.eq_ignore_ascii_case("preview")
        || value.eq_ignore_ascii_case("balanced")
        || value.eq_ignore_ascii_case("final")
}

fn is_sampling_mode_valid(value: &str) -> bool {
    value.eq_ignore_ascii_case("random")
        || value.eq_ignore_ascii_case("halton")
        || value.eq_ignore_ascii_case("sobol")
}
