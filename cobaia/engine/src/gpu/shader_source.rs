use crate::render::capabilities::{GPU_MAX_LIGHTS, GPU_MAX_MATERIALS, GPU_MAX_OBJECTS};

pub(super) fn build_gpu_shader_wgsl() -> String {
    GPU_SHADER_WGSL_TEMPLATE
        .replace("__GPU_MAX_OBJECTS__", &format!("{GPU_MAX_OBJECTS}u"))
        .replace("__GPU_MAX_MATERIALS__", &format!("{GPU_MAX_MATERIALS}u"))
        .replace("__GPU_MAX_LIGHTS__", &format!("{GPU_MAX_LIGHTS}u"))
}

const GPU_SHADER_WGSL_TEMPLATE: &str = include_str!("../shaders/raytrace.wgsl");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn injects_rust_gpu_limits_into_wgsl_template() {
        let shader = build_gpu_shader_wgsl();
        assert!(shader.contains(&format!("const MAX_OBJECTS: u32 = {}u;", GPU_MAX_OBJECTS)));
        assert!(shader.contains(&format!(
            "const MAX_MATERIALS: u32 = {}u;",
            GPU_MAX_MATERIALS
        )));
        assert!(shader.contains(&format!("const MAX_LIGHTS: u32 = {}u;", GPU_MAX_LIGHTS)));
        assert!(!shader.contains("__GPU_MAX_"));
    }
}
