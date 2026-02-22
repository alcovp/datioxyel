mod menger_glass_on_plane;

use crate::domain::Scene;

pub fn build_scene(scene_id: &str) -> Result<Scene, String> {
    if scene_id.eq_ignore_ascii_case(menger_glass_on_plane::SCENE_ID) {
        return Ok(menger_glass_on_plane::build());
    }

    Err(format!("unknown scene identifier: {scene_id}"))
}
