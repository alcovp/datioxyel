use crate::domain::{
    Light, LightKind, Material, MaterialClass, MaterialId, Object, ObjectKind, Scene,
};
use crate::math::Vec3;

pub const SCENE_ID: &str = "menger_glass_on_plane";

pub fn build() -> Scene {
    let floor_y = -1.05;
    let sponge_scale = 0.9;
    let cube_height = sponge_scale * 2.0;
    let mirror_sphere_radius = cube_height / 3.0;
    let mirror_gap = 0.18;
    let sponge_center = Vec3::new(0.0, floor_y + sponge_scale, 0.0);
    let mirror_sphere_center = Vec3::new(
        sponge_scale + mirror_sphere_radius + mirror_gap,
        floor_y + mirror_sphere_radius,
        0.0,
    );

    let floor_material = MaterialId(0);
    let sponge_material = MaterialId(1);
    let sphere_material = MaterialId(2);

    Scene {
        id: SCENE_ID,
        materials: vec![
            Material {
                name: "floor_matte",
                class: MaterialClass::Floor,
                albedo: Vec3::new(0.94, 0.94, 0.93),
            },
            Material {
                name: "menger_opaque_red",
                class: MaterialClass::Opaque,
                albedo: Vec3::new(0.9, 0.09, 0.08),
            },
            Material {
                name: "mirror_sphere",
                class: MaterialClass::Mirror,
                albedo: Vec3::new(1.0, 1.0, 1.0),
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
                name: "mirror_probe_sphere",
                kind: ObjectKind::Sphere {
                    center: mirror_sphere_center,
                    radius: mirror_sphere_radius,
                },
                material_id: sphere_material,
            },
        ],
        lights: vec![Light {
            name: "sun",
            kind: LightKind::Directional {
                direction: Vec3::new(0.78, -1.0, 0.55).normalize(),
                color: Vec3::new(1.0, 0.96, 0.9),
                intensity: 1.0,
            },
        }],
    }
}
