use crate::config::RenderFrameConfig;

#[derive(Clone, Copy, Debug)]
pub enum SamplingMode {
    Random = 0,
    Halton = 1,
    Sobol = 2,
}

#[derive(Clone, Copy, Debug)]
pub struct RenderTuning {
    pub march_max_steps: u32,
    pub rr_start_bounce: u32,
    pub sampling_mode: SamplingMode,
    pub hit_epsilon_scale: f32,
    pub step_scale: f32,
    pub shadow_max_steps: u32,
    pub ao_samples: u32,
    pub adaptive_min_samples_fraction: f32,
    pub adaptive_variance_threshold: f32,
    pub adaptive_check_interval: u32,
    pub firefly_clamp_scale: f32,
    pub shadow_distance_scale: f32,
    pub shadow_min_step_scale: f32,
    pub ao_radius_scale: f32,
    pub ao_horizon_bias: f32,
}

#[derive(Clone, Debug)]
pub struct RenderSettings {
    pub width: u32,
    pub height: u32,
    pub max_depth: u32,
    pub samples_per_pixel: u32,
    pub output_path: String,
    pub tuning: RenderTuning,
}

impl RenderSettings {
    pub fn from_frame(frame: &RenderFrameConfig) -> Self {
        let tuning = RenderTuning::from_frame(frame);
        Self {
            width: frame.width,
            height: frame.height,
            max_depth: frame.max_depth.max(1) as u32,
            samples_per_pixel: frame.samples_per_pixel.max(1) as u32,
            output_path: frame.output_path.clone(),
            tuning,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum QualityPreset {
    Preview,
    Balanced,
    Final,
}

impl RenderTuning {
    fn from_frame(frame: &RenderFrameConfig) -> Self {
        let preset = parse_quality(&frame.quality);
        let sampling_mode = parse_sampling_mode(&frame.sampling_mode);
        let mut tuning = match preset {
            QualityPreset::Preview => Self {
                march_max_steps: 160,
                rr_start_bounce: 2,
                sampling_mode,
                hit_epsilon_scale: 1.55,
                step_scale: 1.35,
                shadow_max_steps: 40,
                ao_samples: 4,
                adaptive_min_samples_fraction: 0.28,
                adaptive_variance_threshold: 0.16,
                adaptive_check_interval: 1,
                firefly_clamp_scale: 0.8,
                shadow_distance_scale: 0.82,
                shadow_min_step_scale: 1.2,
                ao_radius_scale: 0.9,
                ao_horizon_bias: 1.05,
            },
            QualityPreset::Balanced => Self {
                march_max_steps: 220,
                rr_start_bounce: 3,
                sampling_mode,
                hit_epsilon_scale: 1.0,
                step_scale: 1.0,
                shadow_max_steps: 64,
                ao_samples: 5,
                adaptive_min_samples_fraction: 0.44,
                adaptive_variance_threshold: 0.10,
                adaptive_check_interval: 2,
                firefly_clamp_scale: 1.0,
                shadow_distance_scale: 1.0,
                shadow_min_step_scale: 1.0,
                ao_radius_scale: 1.0,
                ao_horizon_bias: 1.0,
            },
            QualityPreset::Final => Self {
                march_max_steps: 280,
                rr_start_bounce: 4,
                sampling_mode,
                hit_epsilon_scale: 0.85,
                step_scale: 0.9,
                shadow_max_steps: 80,
                ao_samples: 6,
                adaptive_min_samples_fraction: 0.62,
                adaptive_variance_threshold: 0.065,
                adaptive_check_interval: 2,
                firefly_clamp_scale: 1.28,
                shadow_distance_scale: 1.15,
                shadow_min_step_scale: 0.84,
                ao_radius_scale: 1.2,
                ao_horizon_bias: 0.94,
            },
        };

        if let Some(march_max_steps) = frame.march_max_steps {
            tuning.march_max_steps = march_max_steps.max(1) as u32;
        }
        if let Some(rr_start_bounce) = frame.rr_start_bounce {
            tuning.rr_start_bounce = rr_start_bounce as u32;
        }

        tuning
    }
}

fn parse_quality(value: &str) -> QualityPreset {
    if value.eq_ignore_ascii_case("preview") {
        return QualityPreset::Preview;
    }
    if value.eq_ignore_ascii_case("final") {
        return QualityPreset::Final;
    }
    QualityPreset::Balanced
}

fn parse_sampling_mode(value: &str) -> SamplingMode {
    if value.eq_ignore_ascii_case("random") {
        return SamplingMode::Random;
    }
    if value.eq_ignore_ascii_case("sobol") {
        return SamplingMode::Sobol;
    }
    SamplingMode::Halton
}
