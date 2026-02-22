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
    pub emission: Vec3,
    pub roughness: f32,
    pub metallic: f32,
    pub transmission: f32,
    pub ior: f32,
    pub absorption: Vec3,
}

impl Material {
    pub fn validate_physical(&self) -> Result<(), String> {
        validate_vec3_finite(self.albedo, "albedo")?;
        validate_vec3_finite(self.emission, "emission")?;
        validate_vec3_finite(self.absorption, "absorption")?;

        validate_vec3_unit_interval(self.albedo, "albedo")?;
        validate_vec3_non_negative(self.emission, "emission")?;
        validate_vec3_non_negative(self.absorption, "absorption")?;

        validate_unit_interval(self.roughness, "roughness")?;
        validate_unit_interval(self.metallic, "metallic")?;
        validate_unit_interval(self.transmission, "transmission")?;

        if !self.ior.is_finite() || self.ior < 1.0 {
            return Err(format!("ior must be finite and >= 1.0, got {}", self.ior));
        }

        match self.class {
            MaterialClass::Glass => {
                if self.transmission <= 0.0 {
                    return Err("glass material must have transmission > 0".into());
                }
                if self.ior <= 1.0 {
                    return Err("glass material must have ior > 1.0".into());
                }
            }
            MaterialClass::Floor | MaterialClass::Opaque | MaterialClass::Mirror => {
                if self.transmission > 0.0 {
                    return Err(format!(
                        "{:?} material must have transmission = 0",
                        self.class
                    ));
                }
            }
        }

        Ok(())
    }
}

fn validate_unit_interval(value: f32, field: &str) -> Result<(), String> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(format!("{field} must be finite and in [0, 1], got {value}"));
    }
    Ok(())
}

fn validate_vec3_finite(value: Vec3, field: &str) -> Result<(), String> {
    if !value.x.is_finite() || !value.y.is_finite() || !value.z.is_finite() {
        return Err(format!(
            "{field} components must be finite, got ({}, {}, {})",
            value.x, value.y, value.z
        ));
    }
    Ok(())
}

fn validate_vec3_unit_interval(value: Vec3, field: &str) -> Result<(), String> {
    validate_unit_interval(value.x, &format!("{field}.x"))?;
    validate_unit_interval(value.y, &format!("{field}.y"))?;
    validate_unit_interval(value.z, &format!("{field}.z"))?;
    Ok(())
}

fn validate_vec3_non_negative(value: Vec3, field: &str) -> Result<(), String> {
    if value.x < 0.0 || value.y < 0.0 || value.z < 0.0 {
        return Err(format!(
            "{field} components must be >= 0, got ({}, {}, {})",
            value.x, value.y, value.z
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_material(class: MaterialClass) -> Material {
        Material {
            name: "test",
            class,
            albedo: Vec3::new(0.7, 0.6, 0.5),
            emission: Vec3::new(0.0, 0.0, 0.0),
            roughness: 0.5,
            metallic: 0.0,
            transmission: 0.0,
            ior: 1.45,
            absorption: Vec3::new(0.0, 0.0, 0.0),
        }
    }

    #[test]
    fn validates_opaque_material() {
        let material = base_material(MaterialClass::Opaque);
        assert!(material.validate_physical().is_ok());
    }

    #[test]
    fn rejects_non_glass_transmission() {
        let mut material = base_material(MaterialClass::Mirror);
        material.transmission = 0.2;
        assert!(material.validate_physical().is_err());
    }

    #[test]
    fn validates_glass_constraints() {
        let mut material = base_material(MaterialClass::Glass);
        assert!(material.validate_physical().is_err());

        material.transmission = 0.95;
        material.ior = 1.52;
        assert!(material.validate_physical().is_ok());
    }
}
