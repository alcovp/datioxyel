use crate::domain::{
    Light, LightKind, Material, MaterialClass, MaterialId, Object, ObjectKind, Scene,
};
use crate::math::Vec3;

pub const SCENE_ID: &str = "primitives_gallery";

pub fn build() -> Scene {
    let floor_y = -1.05;

    let floor_material = MaterialId(0);
    let stone_material = MaterialId(1);
    let glass_material = MaterialId(2);
    let mirror_material = MaterialId(3);

    Scene {
        id: SCENE_ID,
        materials: vec![
            Material {
                name: "floor_matte",
                class: MaterialClass::Floor,
                albedo: Vec3::new(0.93, 0.94, 0.96),
                emission: Vec3::new(0.0, 0.0, 0.0),
                roughness: 0.96,
                metallic: 0.0,
                transmission: 0.0,
                ior: 1.45,
                absorption: Vec3::new(0.0, 0.0, 0.0),
            },
            Material {
                name: "stone_opaque",
                class: MaterialClass::Opaque,
                albedo: Vec3::new(0.72, 0.66, 0.58),
                emission: Vec3::new(0.0, 0.0, 0.0),
                roughness: 0.72,
                metallic: 0.0,
                transmission: 0.0,
                ior: 1.45,
                absorption: Vec3::new(0.0, 0.0, 0.0),
            },
            Material {
                name: "gallery_glass",
                class: MaterialClass::Glass,
                albedo: Vec3::new(1.0, 1.0, 1.0),
                emission: Vec3::new(0.0, 0.0, 0.0),
                roughness: 0.01,
                metallic: 0.0,
                transmission: 0.985,
                ior: 1.5,
                absorption: Vec3::new(0.05, 0.02, 0.01),
            },
            Material {
                name: "gallery_mirror",
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
                name: "box_block",
                kind: ObjectKind::Parallelepiped {
                    center: Vec3::new(-2.3, -0.35, 0.0),
                    half_extents: Vec3::new(0.42, 0.70, 0.34),
                },
                material_id: stone_material,
            },
            Object {
                name: "cylinder_column",
                kind: ObjectKind::Cylinder {
                    center: Vec3::new(-1.35, -0.40, -0.12),
                    radius: 0.35,
                    half_height: 0.65,
                },
                material_id: stone_material,
            },
            Object {
                name: "capsule_pillar",
                kind: ObjectKind::Capsule {
                    center: Vec3::new(-0.25, -0.22, 0.02),
                    radius: 0.28,
                    half_height: 0.55,
                },
                material_id: stone_material,
            },
            Object {
                name: "frustum_tower",
                kind: ObjectKind::Frustum {
                    center: Vec3::new(0.95, -0.40, -0.08),
                    half_height: 0.65,
                    radius_bottom: 0.42,
                    radius_top: 0.16,
                },
                material_id: stone_material,
            },
            Object {
                name: "torus_ring",
                kind: ObjectKind::Torus {
                    center: Vec3::new(2.15, -0.86, 0.02),
                    major_radius: 0.52,
                    minor_radius: 0.19,
                },
                material_id: mirror_material,
            },
            Object {
                name: "ellipsoid_glass",
                kind: ObjectKind::Ellipsoid {
                    center: Vec3::new(0.45, -0.40, 1.10),
                    radii: Vec3::new(0.52, 0.65, 0.34),
                },
                material_id: glass_material,
            },
            Object {
                name: "pyramid_spire",
                kind: ObjectKind::Pyramid {
                    center: Vec3::new(1.95, -0.50, 1.05),
                    half_extent: 0.45,
                    height: 1.10,
                },
                material_id: stone_material,
            },
        ],
        lights: vec![
            Light {
                name: "sun_key",
                kind: LightKind::Directional {
                    direction: Vec3::new(0.82, -1.0, 0.43).normalize(),
                    color: Vec3::new(1.0, 0.96, 0.9),
                    intensity: 1.0,
                },
            },
            Light {
                name: "sky_fill",
                kind: LightKind::Directional {
                    direction: Vec3::new(-0.28, -1.0, -0.56).normalize(),
                    color: Vec3::new(0.70, 0.84, 1.0),
                    intensity: 0.30,
                },
            },
        ],
    }
}
