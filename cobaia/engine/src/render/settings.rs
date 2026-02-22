use crate::config::RenderFrameConfig;

#[derive(Clone, Debug)]
pub struct RenderSettings {
    pub width: u32,
    pub height: u32,
    pub max_depth: u32,
    pub samples_per_pixel: u32,
    pub output_path: String,
}

impl RenderSettings {
    pub fn from_frame(frame: &RenderFrameConfig) -> Self {
        Self {
            width: frame.width,
            height: frame.height,
            max_depth: frame.max_depth.max(1) as u32,
            samples_per_pixel: frame.samples_per_pixel.max(1) as u32,
            output_path: frame.output_path.clone(),
        }
    }
}
