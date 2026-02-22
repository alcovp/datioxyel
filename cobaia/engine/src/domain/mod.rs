pub mod light;
pub mod material;
pub mod object;
pub mod presets;
pub mod scene;

pub use light::{Light, LightKind};
pub use material::{Material, MaterialClass, MaterialId};
pub use object::{Object, ObjectKind};
pub use scene::Scene;
