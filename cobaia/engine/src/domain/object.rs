use crate::math::Vec3;

use super::material::MaterialId;

#[derive(Clone, Copy, Debug)]
pub enum ObjectKind {
    InfinitePlane {
        y: f32,
    },
    Parallelepiped {
        center: Vec3,
        half_extents: Vec3,
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
    Cylinder {
        center: Vec3,
        radius: f32,
        half_height: f32,
    },
    Capsule {
        center: Vec3,
        radius: f32,
        half_height: f32,
    },
    Frustum {
        center: Vec3,
        half_height: f32,
        radius_bottom: f32,
        radius_top: f32,
    },
    Torus {
        center: Vec3,
        major_radius: f32,
        minor_radius: f32,
    },
    RoundedBox {
        center: Vec3,
        half_extents: Vec3,
        radius: f32,
    },
    Ellipsoid {
        center: Vec3,
        radii: Vec3,
    },
    Pyramid {
        center: Vec3,
        half_extent: f32,
        height: f32,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct Object {
    pub name: &'static str,
    pub kind: ObjectKind,
    pub material_id: MaterialId,
}
