use crate::math::Vec3;

#[derive(Clone, Copy, Debug)]
pub enum LightKind {
    Directional {
        direction: Vec3,
        color: Vec3,
        intensity: f32,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct Light {
    pub name: &'static str,
    pub kind: LightKind,
}
