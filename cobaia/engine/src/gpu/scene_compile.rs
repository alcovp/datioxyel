use crate::domain::{LightKind, Material, ObjectKind, Scene};
use crate::render::capabilities::{GPU_MAX_LIGHTS, GPU_MAX_MATERIALS, GPU_MAX_OBJECTS};

const OBJECT_KIND_PLANE: f32 = 0.0;
const OBJECT_KIND_MENGER: f32 = 1.0;
const OBJECT_KIND_SPHERE: f32 = 2.0;
const OBJECT_KIND_PARALLELEPIPED: f32 = 3.0;
const OBJECT_KIND_CYLINDER: f32 = 4.0;
const OBJECT_KIND_PYRAMID: f32 = 5.0;
const OBJECT_KIND_CAPSULE: f32 = 6.0;
const OBJECT_KIND_FRUSTUM: f32 = 7.0;
const OBJECT_KIND_TORUS: f32 = 8.0;
const OBJECT_KIND_ROUNDED_BOX: f32 = 9.0;
const OBJECT_KIND_ELLIPSOID: f32 = 10.0;

#[derive(Clone, Copy, Debug)]
pub(super) struct CompiledGpuScene {
    pub(super) object_count: u32,
    pub(super) material_count: u32,
    pub(super) light_count: u32,
    pub(super) object_meta: [[f32; 4]; GPU_MAX_OBJECTS],
    pub(super) object_data0: [[f32; 4]; GPU_MAX_OBJECTS],
    pub(super) object_data1: [[f32; 4]; GPU_MAX_OBJECTS],
    pub(super) material_albedo_roughness: [[f32; 4]; GPU_MAX_MATERIALS],
    pub(super) material_emission_metallic: [[f32; 4]; GPU_MAX_MATERIALS],
    pub(super) material_optics: [[f32; 4]; GPU_MAX_MATERIALS],
    pub(super) material_absorption: [[f32; 4]; GPU_MAX_MATERIALS],
    pub(super) light_direction: [[f32; 4]; GPU_MAX_LIGHTS],
    pub(super) light_color_intensity: [[f32; 4]; GPU_MAX_LIGHTS],
}

pub(super) fn compile_scene(scene: &Scene) -> Result<CompiledGpuScene, String> {
    if scene.objects.is_empty() {
        return Err("scene must contain at least one object".into());
    }
    if scene.materials.is_empty() {
        return Err("scene must contain at least one material".into());
    }
    if scene.objects.len() > GPU_MAX_OBJECTS {
        return Err(format!(
            "scene has {} objects but GPU backend supports at most {}",
            scene.objects.len(),
            GPU_MAX_OBJECTS
        ));
    }
    if scene.materials.len() > GPU_MAX_MATERIALS {
        return Err(format!(
            "scene has {} materials but GPU backend supports at most {}",
            scene.materials.len(),
            GPU_MAX_MATERIALS
        ));
    }
    if scene.lights.len() > GPU_MAX_LIGHTS {
        return Err(format!(
            "scene has {} lights but GPU backend supports at most {}",
            scene.lights.len(),
            GPU_MAX_LIGHTS
        ));
    }

    let mut object_meta = [[0.0; 4]; GPU_MAX_OBJECTS];
    let mut object_data0 = [[0.0; 4]; GPU_MAX_OBJECTS];
    let mut object_data1 = [[0.0; 4]; GPU_MAX_OBJECTS];

    for (index, object) in scene.objects.iter().enumerate() {
        let material_id = object.material_id.0;
        let _ = material_by_id(scene, object.name, material_id)?;
        object_meta[index][1] = material_id as f32;

        match object.kind {
            ObjectKind::InfinitePlane { y } => {
                object_meta[index][0] = OBJECT_KIND_PLANE;
                object_data0[index] = [y, 0.0, 0.0, 0.0];
            }
            ObjectKind::Parallelepiped {
                center,
                half_extents,
            } => {
                validate_positive_finite_vec3(
                    object.name,
                    "parallelepiped half extents",
                    half_extents,
                )?;
                object_meta[index][0] = OBJECT_KIND_PARALLELEPIPED;
                object_data0[index] = [center.x, center.y, center.z, half_extents.x];
                object_data1[index] = [half_extents.y, half_extents.z, 0.0, 0.0];
            }
            ObjectKind::Menger {
                center,
                scale,
                iterations,
            } => {
                validate_positive_finite_scalar(object.name, "Menger scale", scale)?;
                if iterations == 0 {
                    return Err(format!(
                        "object '{}' has zero Menger iterations",
                        object.name
                    ));
                }
                object_meta[index][0] = OBJECT_KIND_MENGER;
                object_data0[index] = [center.x, center.y, center.z, scale];
                object_data1[index] = [iterations as f32, 0.0, 0.0, 0.0];
            }
            ObjectKind::Sphere { center, radius } => {
                validate_positive_finite_scalar(object.name, "sphere radius", radius)?;
                object_meta[index][0] = OBJECT_KIND_SPHERE;
                object_data0[index] = [center.x, center.y, center.z, radius];
            }
            ObjectKind::Cylinder {
                center,
                radius,
                half_height,
            } => {
                validate_positive_finite_scalar(object.name, "cylinder radius", radius)?;
                validate_positive_finite_scalar(object.name, "cylinder half height", half_height)?;
                object_meta[index][0] = OBJECT_KIND_CYLINDER;
                object_data0[index] = [center.x, center.y, center.z, radius];
                object_data1[index] = [half_height, 0.0, 0.0, 0.0];
            }
            ObjectKind::Capsule {
                center,
                radius,
                half_height,
            } => {
                validate_positive_finite_scalar(object.name, "capsule radius", radius)?;
                validate_positive_finite_scalar(object.name, "capsule half height", half_height)?;
                object_meta[index][0] = OBJECT_KIND_CAPSULE;
                object_data0[index] = [center.x, center.y, center.z, radius];
                object_data1[index] = [half_height, 0.0, 0.0, 0.0];
            }
            ObjectKind::Frustum {
                center,
                half_height,
                radius_bottom,
                radius_top,
            } => {
                validate_positive_finite_scalar(object.name, "frustum half height", half_height)?;
                validate_non_negative_finite_scalar(
                    object.name,
                    "frustum radius_bottom",
                    radius_bottom,
                )?;
                validate_non_negative_finite_scalar(object.name, "frustum radius_top", radius_top)?;
                if radius_bottom <= 0.0 && radius_top <= 0.0 {
                    return Err(format!(
                        "object '{}' must have frustum radius_bottom > 0 or radius_top > 0",
                        object.name
                    ));
                }
                object_meta[index][0] = OBJECT_KIND_FRUSTUM;
                object_data0[index] = [center.x, center.y, center.z, half_height];
                object_data1[index] = [radius_bottom, radius_top, 0.0, 0.0];
            }
            ObjectKind::Torus {
                center,
                major_radius,
                minor_radius,
            } => {
                validate_positive_finite_scalar(object.name, "torus major radius", major_radius)?;
                validate_positive_finite_scalar(object.name, "torus minor radius", minor_radius)?;
                object_meta[index][0] = OBJECT_KIND_TORUS;
                object_data0[index] = [center.x, center.y, center.z, major_radius];
                object_data1[index] = [minor_radius, 0.0, 0.0, 0.0];
            }
            ObjectKind::RoundedBox {
                center,
                half_extents,
                radius,
            } => {
                validate_positive_finite_vec3(
                    object.name,
                    "rounded box half extents",
                    half_extents,
                )?;
                validate_non_negative_finite_scalar(object.name, "rounded box radius", radius)?;
                let min_half_extent = half_extents.x.min(half_extents.y).min(half_extents.z);
                if radius >= min_half_extent {
                    return Err(format!(
                        "object '{}' has rounded box radius ({radius}) that must be < min half extent ({min_half_extent})",
                        object.name
                    ));
                }
                object_meta[index][0] = OBJECT_KIND_ROUNDED_BOX;
                object_data0[index] = [center.x, center.y, center.z, half_extents.x];
                object_data1[index] = [half_extents.y, half_extents.z, radius, 0.0];
            }
            ObjectKind::Ellipsoid { center, radii } => {
                validate_positive_finite_vec3(object.name, "ellipsoid radii", radii)?;
                object_meta[index][0] = OBJECT_KIND_ELLIPSOID;
                object_data0[index] = [center.x, center.y, center.z, radii.x];
                object_data1[index] = [radii.y, radii.z, 0.0, 0.0];
            }
            ObjectKind::Pyramid {
                center,
                half_extent,
                height,
            } => {
                validate_positive_finite_scalar(object.name, "pyramid half extent", half_extent)?;
                validate_positive_finite_scalar(object.name, "pyramid height", height)?;
                object_meta[index][0] = OBJECT_KIND_PYRAMID;
                object_data0[index] = [center.x, center.y, center.z, half_extent];
                object_data1[index] = [height, 0.0, 0.0, 0.0];
            }
        }
    }

    let mut material_albedo_roughness = [[0.0; 4]; GPU_MAX_MATERIALS];
    let mut material_emission_metallic = [[0.0; 4]; GPU_MAX_MATERIALS];
    let mut material_optics = [[0.0; 4]; GPU_MAX_MATERIALS];
    let mut material_absorption = [[0.0; 4]; GPU_MAX_MATERIALS];
    for (index, material) in scene.materials.iter().enumerate() {
        material_albedo_roughness[index] = [
            material.albedo.x,
            material.albedo.y,
            material.albedo.z,
            material.roughness,
        ];
        material_emission_metallic[index] = [
            material.emission.x,
            material.emission.y,
            material.emission.z,
            material.metallic,
        ];
        material_optics[index] = [material.ior, material.transmission, 0.0, 0.0];
        material_absorption[index] = [
            material.absorption.x,
            material.absorption.y,
            material.absorption.z,
            0.0,
        ];
    }

    let mut light_direction = [[0.0; 4]; GPU_MAX_LIGHTS];
    let mut light_color_intensity = [[0.0; 4]; GPU_MAX_LIGHTS];
    let mut light_count: usize = 0;
    for light in &scene.lights {
        light
            .validate_physical()
            .map_err(|error| format!("light '{}' is invalid: {error}", light.name))?;
        match light.kind {
            LightKind::Directional {
                direction,
                color,
                intensity,
            } => {
                if light_count >= GPU_MAX_LIGHTS {
                    return Err(format!(
                        "scene has more than {} supported lights",
                        GPU_MAX_LIGHTS
                    ));
                }
                let slot = light_count;
                let normalized = direction.normalize();
                light_direction[slot] = [normalized.x, normalized.y, normalized.z, 0.0];
                light_color_intensity[slot] = [color.x, color.y, color.z, intensity];
                light_count += 1;
            }
        }
    }

    Ok(CompiledGpuScene {
        object_count: scene.objects.len() as u32,
        material_count: scene.materials.len() as u32,
        light_count: light_count as u32,
        object_meta,
        object_data0,
        object_data1,
        material_albedo_roughness,
        material_emission_metallic,
        material_optics,
        material_absorption,
        light_direction,
        light_color_intensity,
    })
}

fn material_by_id<'a>(
    scene: &'a Scene,
    object_name: &str,
    material_id: usize,
) -> Result<&'a Material, String> {
    scene.materials.get(material_id).ok_or_else(|| {
        format!("object '{object_name}' references missing material id {material_id}")
    })
}

fn validate_positive_finite_scalar(
    object_name: &str,
    field: &str,
    value: f32,
) -> Result<(), String> {
    if !value.is_finite() || value <= 0.0 {
        return Err(format!(
            "object '{object_name}' has non-positive or non-finite {field} ({value})"
        ));
    }
    Ok(())
}

fn validate_non_negative_finite_scalar(
    object_name: &str,
    field: &str,
    value: f32,
) -> Result<(), String> {
    if !value.is_finite() || value < 0.0 {
        return Err(format!(
            "object '{object_name}' has negative or non-finite {field} ({value})"
        ));
    }
    Ok(())
}

fn validate_positive_finite_vec3(
    object_name: &str,
    field: &str,
    value: crate::math::Vec3,
) -> Result<(), String> {
    if !(value.x.is_finite()
        && value.y.is_finite()
        && value.z.is_finite()
        && value.x > 0.0
        && value.y > 0.0
        && value.z > 0.0)
    {
        return Err(format!(
            "object '{object_name}' has non-positive or non-finite {field} ({}, {}, {})",
            value.x, value.y, value.z
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{MaterialClass, MaterialId, Object};
    use crate::math::Vec3;

    fn test_material() -> Material {
        Material {
            name: "opaque",
            class: MaterialClass::Opaque,
            albedo: Vec3::new(0.7, 0.7, 0.7),
            emission: Vec3::new(0.0, 0.0, 0.0),
            roughness: 0.5,
            metallic: 0.0,
            transmission: 0.0,
            ior: 1.45,
            absorption: Vec3::new(0.0, 0.0, 0.0),
        }
    }

    fn scene_with_objects(objects: Vec<Object>) -> Scene {
        Scene {
            id: "test",
            objects,
            materials: vec![test_material()],
            lights: vec![],
        }
    }

    #[test]
    fn compiles_new_primitive_kinds() {
        let scene = scene_with_objects(vec![
            Object {
                name: "box",
                kind: ObjectKind::Parallelepiped {
                    center: Vec3::new(0.0, 0.0, 0.0),
                    half_extents: Vec3::new(1.0, 2.0, 0.5),
                },
                material_id: MaterialId(0),
            },
            Object {
                name: "cylinder",
                kind: ObjectKind::Cylinder {
                    center: Vec3::new(0.0, 0.0, 0.0),
                    radius: 0.8,
                    half_height: 1.2,
                },
                material_id: MaterialId(0),
            },
            Object {
                name: "capsule",
                kind: ObjectKind::Capsule {
                    center: Vec3::new(0.0, 0.0, 0.0),
                    radius: 0.4,
                    half_height: 1.0,
                },
                material_id: MaterialId(0),
            },
            Object {
                name: "frustum",
                kind: ObjectKind::Frustum {
                    center: Vec3::new(0.0, 0.0, 0.0),
                    half_height: 1.2,
                    radius_bottom: 0.9,
                    radius_top: 0.2,
                },
                material_id: MaterialId(0),
            },
            Object {
                name: "torus",
                kind: ObjectKind::Torus {
                    center: Vec3::new(0.0, 0.0, 0.0),
                    major_radius: 1.1,
                    minor_radius: 0.3,
                },
                material_id: MaterialId(0),
            },
            Object {
                name: "rounded_box",
                kind: ObjectKind::RoundedBox {
                    center: Vec3::new(0.0, 0.0, 0.0),
                    half_extents: Vec3::new(1.0, 0.8, 0.6),
                    radius: 0.15,
                },
                material_id: MaterialId(0),
            },
            Object {
                name: "ellipsoid",
                kind: ObjectKind::Ellipsoid {
                    center: Vec3::new(0.0, 0.0, 0.0),
                    radii: Vec3::new(1.2, 0.7, 0.9),
                },
                material_id: MaterialId(0),
            },
            Object {
                name: "pyramid",
                kind: ObjectKind::Pyramid {
                    center: Vec3::new(0.0, 0.0, 0.0),
                    half_extent: 1.0,
                    height: 2.0,
                },
                material_id: MaterialId(0),
            },
        ]);

        let compiled = compile_scene(&scene).expect("new primitive kinds should compile");
        assert_eq!(compiled.object_count, 8);
        assert_eq!(compiled.object_meta[0][0], OBJECT_KIND_PARALLELEPIPED);
        assert_eq!(compiled.object_meta[1][0], OBJECT_KIND_CYLINDER);
        assert_eq!(compiled.object_meta[2][0], OBJECT_KIND_CAPSULE);
        assert_eq!(compiled.object_meta[3][0], OBJECT_KIND_FRUSTUM);
        assert_eq!(compiled.object_meta[4][0], OBJECT_KIND_TORUS);
        assert_eq!(compiled.object_meta[5][0], OBJECT_KIND_ROUNDED_BOX);
        assert_eq!(compiled.object_meta[6][0], OBJECT_KIND_ELLIPSOID);
        assert_eq!(compiled.object_meta[7][0], OBJECT_KIND_PYRAMID);
    }

    #[test]
    fn rejects_invalid_parallelepiped_half_extents() {
        let scene = scene_with_objects(vec![Object {
            name: "bad_box",
            kind: ObjectKind::Parallelepiped {
                center: Vec3::new(0.0, 0.0, 0.0),
                half_extents: Vec3::new(1.0, 0.0, 0.5),
            },
            material_id: MaterialId(0),
        }]);

        let error = compile_scene(&scene).expect_err("invalid box must fail");
        assert!(error.contains("parallelepiped half extents"));
    }

    #[test]
    fn rejects_invalid_cylinder_height() {
        let scene = scene_with_objects(vec![Object {
            name: "bad_cylinder",
            kind: ObjectKind::Cylinder {
                center: Vec3::new(0.0, 0.0, 0.0),
                radius: 0.8,
                half_height: 0.0,
            },
            material_id: MaterialId(0),
        }]);

        let error = compile_scene(&scene).expect_err("invalid cylinder must fail");
        assert!(error.contains("cylinder half height"));
    }

    #[test]
    fn rejects_frustum_with_zero_radii() {
        let scene = scene_with_objects(vec![Object {
            name: "bad_frustum",
            kind: ObjectKind::Frustum {
                center: Vec3::new(0.0, 0.0, 0.0),
                half_height: 1.0,
                radius_bottom: 0.0,
                radius_top: 0.0,
            },
            material_id: MaterialId(0),
        }]);

        let error = compile_scene(&scene).expect_err("invalid frustum must fail");
        assert!(error.contains("radius_bottom > 0 or radius_top > 0"));
    }

    #[test]
    fn rejects_rounded_box_radius_too_large() {
        let scene = scene_with_objects(vec![Object {
            name: "bad_rounded_box",
            kind: ObjectKind::RoundedBox {
                center: Vec3::new(0.0, 0.0, 0.0),
                half_extents: Vec3::new(0.5, 0.5, 0.5),
                radius: 0.5,
            },
            material_id: MaterialId(0),
        }]);

        let error = compile_scene(&scene).expect_err("invalid rounded box must fail");
        assert!(error.contains("rounded box radius"));
    }
}
