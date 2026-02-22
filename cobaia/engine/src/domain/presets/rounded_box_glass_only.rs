use crate::domain::{
    Light, LightKind, Material, MaterialClass, MaterialId, Object, ObjectKind, Scene,
};
use crate::math::Vec3;

pub const SCENE_ID: &str = "rounded_box_glass_only";

pub fn build() -> Scene {
    let glass_material = MaterialId(0);

    Scene {
        id: SCENE_ID,
        materials: vec![Material {
            name: "gallery_glass",
            class: MaterialClass::Glass,
            albedo: Vec3::new(1.0, 1.0, 1.0),
            emission: Vec3::new(0.0, 0.0, 0.0),
            roughness: 0.01,
            metallic: 0.0,
            transmission: 0.985,
            ior: 1.5,
            absorption: Vec3::new(0.05, 0.02, 0.01),
        }],
        objects: vec![Object {
            name: "rounded_box_glass",
            kind: ObjectKind::RoundedBox {
                center: Vec3::new(-0.95, -0.60, 1.05),
                half_extents: Vec3::new(0.46, 0.45, 0.40),
                radius: 0.12,
            },
            material_id: glass_material,
        }],
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
