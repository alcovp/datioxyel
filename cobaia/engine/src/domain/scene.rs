use super::{Light, Material, MaterialId, Object};

#[derive(Clone, Debug)]
pub struct Scene {
    pub id: &'static str,
    pub objects: Vec<Object>,
    pub materials: Vec<Material>,
    pub lights: Vec<Light>,
}

impl Scene {
    pub fn material(&self, id: MaterialId) -> Option<&Material> {
        self.materials.get(id.0)
    }
}
