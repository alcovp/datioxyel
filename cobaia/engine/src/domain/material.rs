use crate::math::Vec3;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MaterialId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaterialClass {
    Floor,
    Opaque,
    Glass,
    Mirror,
}

#[derive(Clone, Copy, Debug)]
pub struct Material {
    pub name: &'static str,
    pub class: MaterialClass,
    pub albedo: Vec3,
}
