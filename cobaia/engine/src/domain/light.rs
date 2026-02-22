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

impl Light {
    pub fn validate_physical(&self) -> Result<(), String> {
        match self.kind {
            LightKind::Directional {
                direction,
                color,
                intensity,
            } => {
                validate_vec3_finite(direction, "direction")?;
                validate_vec3_finite(color, "color")?;
                validate_vec3_non_negative(color, "color")?;
                if direction.length() < 0.0001 {
                    return Err("direction vector length must be > 0".into());
                }
                if !intensity.is_finite() || intensity <= 0.0 {
                    return Err(format!("intensity must be finite and > 0, got {intensity}"));
                }
            }
        }
        Ok(())
    }
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

    fn directional_light() -> Light {
        Light {
            name: "test",
            kind: LightKind::Directional {
                direction: Vec3::new(0.5, -1.0, 0.1),
                color: Vec3::new(1.0, 0.9, 0.8),
                intensity: 1.0,
            },
        }
    }

    #[test]
    fn validates_directional_light() {
        let light = directional_light();
        assert!(light.validate_physical().is_ok());
    }

    #[test]
    fn rejects_non_finite_intensity() {
        let mut light = directional_light();
        match &mut light.kind {
            LightKind::Directional { intensity, .. } => {
                *intensity = f32::NAN;
            }
        }
        assert!(light.validate_physical().is_err());
    }

    #[test]
    fn rejects_negative_color_component() {
        let mut light = directional_light();
        match &mut light.kind {
            LightKind::Directional { color, .. } => {
                *color = Vec3::new(-0.1, 0.8, 0.9);
            }
        }
        assert!(light.validate_physical().is_err());
    }
}
