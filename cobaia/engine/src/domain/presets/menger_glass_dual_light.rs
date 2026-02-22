use crate::domain::{
    Light, LightKind, Material, MaterialClass, MaterialId, Object, ObjectKind, Scene,
};
use crate::math::Vec3;

pub const SCENE_ID: &str = "menger_glass_dual_light";

pub fn build() -> Scene {
    let floor_y = -1.05;
    let sponge_scale = 0.9;
    let cube_height = sponge_scale * 2.0;
    let sphere_radius = cube_height / 3.2;
    let sponge_center = Vec3::new(0.0, floor_y + sponge_scale, 0.0);
    let glass_sphere_center = Vec3::new(
        sponge_scale + sphere_radius + 0.18,
        floor_y + sphere_radius,
        0.0,
    );
    let mirror_sphere_center = Vec3::new(
        -(sponge_scale + sphere_radius + 0.22),
        floor_y + sphere_radius * 1.05,
        0.12,
    );

    let floor_material = MaterialId(0);
    let sponge_material = MaterialId(1);
    let glass_material = MaterialId(2);
    let mirror_material = MaterialId(3);

    Scene {
        id: SCENE_ID,
        materials: vec![
            Material {
                name: "floor_matte",
                class: MaterialClass::Floor,
                albedo: Vec3::new(0.94, 0.94, 0.93),
                emission: Vec3::new(0.0, 0.0, 0.0),
                roughness: 0.95,
                metallic: 0.0,
                transmission: 0.0,
                ior: 1.45,
                absorption: Vec3::new(0.0, 0.0, 0.0),
            },
            Material {
                name: "menger_opaque_red",
                class: MaterialClass::Opaque,
                albedo: Vec3::new(0.9, 0.09, 0.08),
                emission: Vec3::new(0.0, 0.0, 0.0),
                roughness: 0.68,
                metallic: 0.0,
                transmission: 0.0,
                ior: 1.45,
                absorption: Vec3::new(0.0, 0.0, 0.0),
            },
            Material {
                name: "glass_probe",
                class: MaterialClass::Glass,
                albedo: Vec3::new(1.0, 1.0, 1.0),
                emission: Vec3::new(0.0, 0.0, 0.0),
                roughness: 0.01,
                metallic: 0.0,
                transmission: 0.97,
                ior: 1.52,
                absorption: Vec3::new(0.07, 0.03, 0.015),
            },
            Material {
                name: "mirror_probe",
                class: MaterialClass::Mirror,
                albedo: Vec3::new(0.98, 0.98, 1.0),
                emission: Vec3::new(0.0, 0.0, 0.0),
                roughness: 0.02,
                metallic: 1.0,
                transmission: 0.0,
                ior: 1.45,
                absorption: Vec3::new(0.0, 0.0, 0.0),
            },
        ],
        objects: vec![
            Object {
                name: "floor",
                kind: ObjectKind::InfinitePlane { y: floor_y },
                material_id: floor_material,
            },
            Object {
                name: "menger_sponge",
                kind: ObjectKind::Menger {
                    center: sponge_center,
                    scale: sponge_scale,
                    iterations: 6,
                },
                material_id: sponge_material,
            },
            Object {
                name: "glass_probe_sphere",
                kind: ObjectKind::Sphere {
                    center: glass_sphere_center,
                    radius: sphere_radius,
                },
                material_id: glass_material,
            },
            Object {
                name: "mirror_probe_sphere",
                kind: ObjectKind::Sphere {
                    center: mirror_sphere_center,
                    radius: sphere_radius * 0.95,
                },
                material_id: mirror_material,
            },
        ],
        lights: vec![
            Light {
                name: "sun_key",
                kind: LightKind::Directional {
                    direction: Vec3::new(0.78, -1.0, 0.55).normalize(),
                    color: Vec3::new(1.0, 0.96, 0.9),
                    intensity: 1.0,
                },
            },
            Light {
                name: "sky_fill",
                kind: LightKind::Directional {
                    direction: Vec3::new(-0.35, -1.0, -0.42).normalize(),
                    color: Vec3::new(0.74, 0.86, 1.0),
                    intensity: 0.35,
                },
            },
        ],
    }
}
