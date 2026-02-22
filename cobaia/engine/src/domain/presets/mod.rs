mod menger_glass_dual_light;
mod menger_glass_on_plane;
mod nested_media_aquarium;
mod primitives_gallery;
mod rounded_box_glass_only;

use crate::domain::Scene;

pub fn build_scene(scene_id: &str) -> Result<Scene, String> {
    if scene_id.eq_ignore_ascii_case(menger_glass_dual_light::SCENE_ID) {
        return Ok(menger_glass_dual_light::build());
    }
    if scene_id.eq_ignore_ascii_case(menger_glass_on_plane::SCENE_ID) {
        return Ok(menger_glass_on_plane::build());
    }
    if scene_id.eq_ignore_ascii_case(nested_media_aquarium::SCENE_ID) {
        return Ok(nested_media_aquarium::build());
    }
    if scene_id.eq_ignore_ascii_case(primitives_gallery::SCENE_ID) {
        return Ok(primitives_gallery::build());
    }
    if scene_id.eq_ignore_ascii_case(rounded_box_glass_only::SCENE_ID) {
        return Ok(rounded_box_glass_only::build());
    }

    Err(format!("unknown scene identifier: {scene_id}"))
}
