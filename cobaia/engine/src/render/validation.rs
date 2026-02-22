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
    if scene.materials.len() > capabilities.max_materials {
        return Err(format!(
            "scene has {} materials but renderer supports at most {}",
            scene.materials.len(),
            capabilities.max_materials
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
    for light in &scene.lights {
        light
            .validate_physical()
            .map_err(|error| format!("light '{}' is invalid: {error}", light.name))?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        Light, LightKind, Material, MaterialClass, MaterialId, Object, ObjectKind, Scene,
    };
    use crate::math::Vec3;

    fn base_scene() -> Scene {
        Scene {
            id: "test",
            objects: vec![Object {
                name: "probe",
                kind: ObjectKind::Sphere {
                    center: Vec3::new(0.0, 0.0, 0.0),
                    radius: 1.0,
                },
                material_id: MaterialId(0),
            }],
            materials: vec![Material {
                name: "opaque",
                class: MaterialClass::Opaque,
                albedo: Vec3::new(0.7, 0.6, 0.5),
                emission: Vec3::new(0.0, 0.0, 0.0),
                roughness: 0.4,
                metallic: 0.1,
                transmission: 0.0,
                ior: 1.45,
                absorption: Vec3::new(0.0, 0.0, 0.0),
            }],
            lights: vec![Light {
                name: "sun",
                kind: LightKind::Directional {
                    direction: Vec3::new(0.4, -1.0, 0.2),
                    color: Vec3::new(1.0, 0.95, 0.9),
                    intensity: 1.0,
                },
            }],
        }
    }

    fn base_capabilities() -> RendererCapabilities {
        RendererCapabilities {
            supports_reflection: true,
            supports_refraction: true,
            max_objects: 8,
            max_materials: 1,
            max_lights: 4,
            supported_scene_ids: &[],
        }
    }

    #[test]
    fn rejects_material_count_above_capabilities() {
        let mut scene = base_scene();
        scene.materials.push(scene.materials[0]);
        let err = validate_scene_against_capabilities(&scene, base_capabilities())
            .expect_err("scene with too many materials should fail validation");
        assert!(err.contains("renderer supports at most"));
    }

    #[test]
    fn rejects_physically_invalid_light() {
        let mut scene = base_scene();
        match &mut scene.lights[0].kind {
            LightKind::Directional { intensity, .. } => {
                *intensity = f32::NAN;
            }
        }
        let err = validate_scene_against_capabilities(&scene, base_capabilities())
            .expect_err("scene with invalid light should fail validation");
        assert!(err.contains("light 'sun' is invalid"));
    }
}
