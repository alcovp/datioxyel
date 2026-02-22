use crate::math::Vec3;

#[derive(Clone, Copy)]
pub enum MaterialId {
    Floor,
    Glass,
    Mirror,
}

#[derive(Clone, Copy)]
pub struct SdfSample {
    pub distance: f32,
    pub material: MaterialId,
}

#[derive(Clone, Copy)]
pub struct HitRecord {
    pub t: f32,
    pub point: Vec3,
    pub normal: Vec3,
    pub material: MaterialId,
}

#[derive(Clone, Copy)]
pub struct Scene {
    pub floor_y: f32,
    pub sponge_center: Vec3,
    pub sponge_scale: f32,
    pub sponge_iterations: u32,
    pub mirror_sphere_center: Vec3,
    pub mirror_sphere_radius: f32,
    pub sun_direction: Vec3,
}

impl Scene {
    pub fn stage4_scene() -> Self {
        let floor_y = -1.05;
        let sponge_scale = 0.9;
        let cube_height = sponge_scale * 2.0;
        // Diameter is 2/3 of the cube height -> radius is cube_height / 3.
        let mirror_sphere_radius = cube_height / 3.0;
        let mirror_gap = 0.18;
        let mirror_sphere_center = Vec3::new(
            sponge_scale + mirror_sphere_radius + mirror_gap,
            floor_y + mirror_sphere_radius,
            0.0,
        );

        Self {
            floor_y,
            sponge_center: Vec3::new(0.0, floor_y + sponge_scale, 0.0),
            sponge_scale,
            sponge_iterations: 6,
            mirror_sphere_center,
            mirror_sphere_radius,
            // Direction of sunlight rays (from sun toward scene).
            // Tuned so the floor shadow is visible from the current camera angle.
            sun_direction: Vec3::new(0.78, -1.0, 0.55).normalize(),
        }
    }

    pub fn sample(self, p: Vec3) -> SdfSample {
        let floor_distance = p.y - self.floor_y;
        let local = (p - self.sponge_center) / self.sponge_scale;
        let sponge_distance = sd_menger(local, self.sponge_iterations) * self.sponge_scale;
        let mirror_sphere_distance =
            sd_sphere(p - self.mirror_sphere_center, self.mirror_sphere_radius);

        let mut closest = SdfSample {
            distance: floor_distance,
            material: MaterialId::Floor,
        };
        if sponge_distance < closest.distance {
            closest = SdfSample {
                distance: sponge_distance,
                material: MaterialId::Glass,
            };
        }
        if mirror_sphere_distance < closest.distance {
            closest = SdfSample {
                distance: mirror_sphere_distance,
                material: MaterialId::Mirror,
            };
        }

        closest
    }

    pub fn distance(self, p: Vec3) -> f32 {
        self.sample(p).distance
    }
}

fn sd_box(p: Vec3, half_extents: Vec3) -> f32 {
    let q = p.abs() - half_extents;
    let outside = q.max(Vec3::splat(0.0));
    outside.length() + q.max_component().min(0.0)
}

fn sd_sphere(p: Vec3, radius: f32) -> f32 {
    p.length() - radius
}

fn sd_menger(p: Vec3, iterations: u32) -> f32 {
    let mut distance = sd_box(p, Vec3::splat(1.0));
    let mut scale = 1.0;

    for _ in 0..iterations {
        let cell = (p * scale).rem_euclid(2.0) - Vec3::splat(1.0);
        scale *= 3.0;
        // Canonical Menger fold: absolute after the subtraction keeps the full
        // cross-shaped carve pattern on each subdivision level.
        let r = (Vec3::splat(1.0) - (cell.abs() * 3.0)).abs();

        let da = r.x.max(r.y);
        let db = r.y.max(r.z);
        let dc = r.x.max(r.z);
        let carved = (da.min(db).min(dc) - 1.0) / scale;
        distance = distance.max(carved);
    }

    distance
}
