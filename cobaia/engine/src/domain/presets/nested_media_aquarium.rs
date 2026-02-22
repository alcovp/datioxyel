use crate::domain::{
    Light, LightKind, Material, MaterialClass, MaterialId, Object, ObjectKind, Scene,
};
use crate::math::Vec3;

pub const SCENE_ID: &str = "nested_media_aquarium";

pub fn build() -> Scene {
    let floor_y = -1.05;
    let vessel_center = Vec3::new(0.0, 0.15, 0.0);
    let glass_radius = 0.95;
    let water_radius = 0.82;
    let bubble_center = Vec3::new(-0.24, 0.28, 0.12);
    let bubble_radius = 0.22;
    let probe_center = Vec3::new(1.35, -0.28, -0.32);
    let probe_radius = 0.44;
    let mirror_probe_center = Vec3::new(-1.25, -0.42, -0.38);
    let mirror_probe_radius = 0.38;

    let floor_material = MaterialId(0);
    let glass_shell_material = MaterialId(1);
    let water_material = MaterialId(2);
    let bubble_material = MaterialId(3);
    let probe_material = MaterialId(4);
    let mirror_probe_material = MaterialId(5);

    Scene {
        id: SCENE_ID,
        materials: vec![
            Material {
                name: "floor_matte",
                class: MaterialClass::Floor,
                albedo: Vec3::new(0.93, 0.94, 0.96),
                emission: Vec3::new(0.0, 0.0, 0.0),
                roughness: 0.94,
                metallic: 0.0,
                transmission: 0.0,
                ior: 1.45,
                absorption: Vec3::new(0.0, 0.0, 0.0),
            },
            Material {
                name: "glass_shell",
                class: MaterialClass::Glass,
                albedo: Vec3::new(1.0, 1.0, 1.0),
                emission: Vec3::new(0.0, 0.0, 0.0),
                roughness: 0.01,
                metallic: 0.0,
                transmission: 0.985,
                ior: 1.52,
                absorption: Vec3::new(0.012, 0.006, 0.003),
            },
            Material {
                name: "water_volume",
                class: MaterialClass::Glass,
                albedo: Vec3::new(0.98, 0.995, 1.0),
                emission: Vec3::new(0.0, 0.0, 0.0),
                roughness: 0.004,
                metallic: 0.0,
                transmission: 0.995,
                ior: 1.333,
                absorption: Vec3::new(0.09, 0.03, 0.012),
            },
            Material {
                name: "air_bubble",
                class: MaterialClass::Glass,
                albedo: Vec3::new(1.0, 1.0, 1.0),
                emission: Vec3::new(0.0, 0.0, 0.0),
                roughness: 0.0,
                metallic: 0.0,
                transmission: 0.998,
                ior: 1.0003,
                absorption: Vec3::new(0.0, 0.0, 0.0),
            },
            Material {
                name: "ceramic_probe",
                class: MaterialClass::Opaque,
                albedo: Vec3::new(0.94, 0.36, 0.14),
                emission: Vec3::new(0.0, 0.0, 0.0),
                roughness: 0.42,
                metallic: 0.0,
                transmission: 0.0,
                ior: 1.45,
                absorption: Vec3::new(0.0, 0.0, 0.0),
            },
            Material {
                name: "mirror_probe",
                class: MaterialClass::Mirror,
                albedo: Vec3::new(0.98, 0.99, 1.0),
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
                name: "glass_shell",
                kind: ObjectKind::Sphere {
                    center: vessel_center,
                    radius: glass_radius,
                },
                material_id: glass_shell_material,
            },
            Object {
                name: "water_core",
                kind: ObjectKind::Sphere {
                    center: vessel_center,
                    radius: water_radius,
                },
                material_id: water_material,
            },
            Object {
                name: "air_bubble",
                kind: ObjectKind::Sphere {
                    center: bubble_center,
                    radius: bubble_radius,
                },
                material_id: bubble_material,
            },
            Object {
                name: "ceramic_probe",
                kind: ObjectKind::Sphere {
                    center: probe_center,
                    radius: probe_radius,
                },
                material_id: probe_material,
            },
            Object {
                name: "mirror_probe",
                kind: ObjectKind::Sphere {
                    center: mirror_probe_center,
                    radius: mirror_probe_radius,
                },
                material_id: mirror_probe_material,
            },
        ],
        lights: vec![
            Light {
                name: "sun_key",
                kind: LightKind::Directional {
                    direction: Vec3::new(0.85, -1.0, 0.45).normalize(),
                    color: Vec3::new(1.0, 0.96, 0.9),
                    intensity: 1.05,
                },
            },
            Light {
                name: "sky_fill",
                kind: LightKind::Directional {
                    direction: Vec3::new(-0.32, -1.0, -0.58).normalize(),
                    color: Vec3::new(0.68, 0.82, 1.0),
                    intensity: 0.32,
                },
            },
        ],
    }
}
