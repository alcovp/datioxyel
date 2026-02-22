use crate::math::Vec3;

use super::material::MaterialId;

#[derive(Clone, Copy, Debug)]
pub enum ObjectKind {
    InfinitePlane {
        y: f32,
    },
    Menger {
        center: Vec3,
        scale: f32,
        iterations: u32,
    },
    Sphere {
        center: Vec3,
        radius: f32,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct Object {
    pub name: &'static str,
    pub kind: ObjectKind,
    pub material_id: MaterialId,
}
