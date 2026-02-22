use crate::config::RenderFrameConfig;
use crate::math::Vec3;

#[derive(Clone, Copy, Debug)]
pub struct View {
    pub origin: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub vertical_fov_deg: f32,
}

impl View {
    pub fn from_frame(frame: &RenderFrameConfig) -> Self {
        Self {
            origin: Vec3::new(
                frame.camera_origin[0],
                frame.camera_origin[1],
                frame.camera_origin[2],
            ),
            target: Vec3::new(
                frame.camera_target[0],
                frame.camera_target[1],
                frame.camera_target[2],
            ),
            up: Vec3::new(0.0, 1.0, 0.0),
            vertical_fov_deg: 38.0,
        }
    }
}
