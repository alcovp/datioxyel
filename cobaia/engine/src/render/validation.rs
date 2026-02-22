use crate::domain::{MaterialClass, Scene};

use super::RendererCapabilities;

pub fn validate_scene_against_capabilities(
    scene: &Scene,
    capabilities: RendererCapabilities,
) -> Result<(), String> {
    if scene.objects.len() > capabilities.max_objects {
        return Err(format!(
            "scene has {} objects but renderer supports at most {}",
            scene.objects.len(),
            capabilities.max_objects
        ));
    }
    if scene.lights.len() > capabilities.max_lights {
        return Err(format!(
            "scene has {} lights but renderer supports at most {}",
            scene.lights.len(),
            capabilities.max_lights
        ));
    }
    if !capabilities.supported_scene_ids.is_empty()
        && !capabilities
            .supported_scene_ids
            .iter()
            .any(|supported| scene.id.eq_ignore_ascii_case(supported))
    {
        return Err(format!("scene '{}' is not in renderer whitelist", scene.id));
    }

    let mut material_usage = vec![0usize; scene.materials.len()];
    for object in &scene.objects {
        if scene.material(object.material_id).is_none() {
            return Err(format!(
                "object '{}' references missing material id {}",
                object.name, object.material_id.0
            ));
        }
        material_usage[object.material_id.0] += 1;
    }

    for (index, usage) in material_usage.iter().enumerate() {
        if *usage == 0 {
            return Err(format!(
                "material id {} ('{}') is not used by any object",
                index, scene.materials[index].name
            ));
        }
    }
    for material in &scene.materials {
        material
            .validate_physical()
            .map_err(|error| format!("material '{}' is invalid: {error}", material.name))?;
    }

    let has_glass = material_usage
        .iter()
        .enumerate()
        .any(|(index, usage)| *usage > 0 && scene.materials[index].class == MaterialClass::Glass);
    if has_glass && !capabilities.supports_refraction {
        return Err("scene uses glass materials but renderer does not support refraction".into());
    }
    let has_reflective = material_usage.iter().enumerate().any(|(index, usage)| {
        *usage > 0
            && (scene.materials[index].class == MaterialClass::Glass
                || scene.materials[index].class == MaterialClass::Mirror)
    });
    if has_reflective && !capabilities.supports_reflection {
        return Err(
            "scene uses reflective materials but renderer does not support reflection".into(),
        );
    }

    Ok(())
}
