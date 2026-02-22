#[derive(Clone, Copy, Debug)]
pub struct RendererCapabilities {
    pub supports_reflection: bool,
    pub supports_refraction: bool,
    pub max_objects: usize,
    pub max_lights: usize,
    pub supported_scene_ids: &'static [&'static str],
}

pub fn gpu_capabilities() -> RendererCapabilities {
    RendererCapabilities {
        supports_reflection: true,
        supports_refraction: false,
        max_objects: 3,
        max_lights: 1,
        supported_scene_ids: &["menger_glass_on_plane"],
    }
}
