pub const GPU_MAX_OBJECTS: usize = 32;
pub const GPU_MAX_MATERIALS: usize = 32;
pub const GPU_MAX_LIGHTS: usize = 8;

#[derive(Clone, Copy, Debug)]
pub struct RendererCapabilities {
    pub supports_reflection: bool,
    pub supports_refraction: bool,
    pub max_objects: usize,
    pub max_materials: usize,
    pub max_lights: usize,
    pub supported_scene_ids: &'static [&'static str],
}

pub fn gpu_capabilities() -> RendererCapabilities {
    RendererCapabilities {
        supports_reflection: true,
        supports_refraction: true,
        max_objects: GPU_MAX_OBJECTS,
        max_materials: GPU_MAX_MATERIALS,
        max_lights: GPU_MAX_LIGHTS,
        supported_scene_ids: &[],
    }
}
