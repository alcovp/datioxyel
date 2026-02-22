use image::{Rgb, RgbImage};
use rayon::prelude::*;

use crate::config::{vec3_from, DebugOptions, RenderFrameConfig};
use crate::math::{frosted_offset, reflect, refract, sample_jitter, schlick, Ray, Vec3};
use crate::scene::{HitRecord, MaterialId, Scene};

const MAX_MARCH_STEPS: u32 = 280;
const MAX_TRACE_DISTANCE: f32 = 42.0;
const HIT_EPSILON: f32 = 0.00035;
const NORMAL_EPSILON: f32 = 0.001;
const RAY_BIAS: f32 = 0.003;

struct Camera {
    origin: Vec3,
    lower_left_corner: Vec3,
    horizontal: Vec3,
    vertical: Vec3,
}

impl Camera {
    fn new(origin: Vec3, target: Vec3, up: Vec3, vfov_deg: f32, aspect_ratio: f32) -> Self {
        let theta = vfov_deg.to_radians();
        let h = (theta * 0.5).tan();
        let viewport_height = 2.0 * h;
        let viewport_width = aspect_ratio * viewport_height;

        let w = (origin - target).normalize();
        let u = up.cross(w).normalize();
        let v = w.cross(u);

        let horizontal = u * viewport_width;
        let vertical = v * viewport_height;
        let lower_left_corner = origin - (horizontal * 0.5) - (vertical * 0.5) - w;

        Self {
            origin,
            lower_left_corner,
            horizontal,
            vertical,
        }
    }

    fn get_ray(&self, u: f32, v: f32) -> Ray {
        let direction = (self.lower_left_corner + (self.horizontal * u) + (self.vertical * v)
            - self.origin)
            .normalize();
        Ray {
            origin: self.origin,
            direction,
        }
    }
}

pub fn render_cpu(config: &RenderFrameConfig, scene: Scene, debug: DebugOptions) -> RgbImage {
    let mut image = RgbImage::new(config.width, config.height);
    let aspect_ratio = config.width as f32 / config.height as f32;
    let camera = Camera::new(
        vec3_from(config.camera_origin),
        vec3_from(config.camera_target),
        Vec3::new(0.0, 1.0, 0.0),
        38.0,
        aspect_ratio,
    );
    let width = config.width as usize;
    let height = config.height as usize;
    let sample_count = config.samples_per_pixel.max(1) as u32;
    let width_f = config.width.max(1) as f32;
    let height_f = config.height.max(1) as f32;
    let mut color_buffer = vec![Vec3::splat(0.0); width * height];

    // Minimal parallelism stage: split work by scanlines.
    color_buffer
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            let y_u32 = y as u32;
            for (x, color_slot) in row.iter_mut().enumerate() {
                let x_u32 = x as u32;
                let mut accumulated = Vec3::splat(0.0);
                for sample_index in 0..sample_count {
                    let jitter_x = sample_jitter(x_u32, y_u32, sample_index, 0);
                    let jitter_y = sample_jitter(x_u32, y_u32, sample_index, 1);
                    let u = (x_u32 as f32 + jitter_x) / width_f;
                    let v = ((config.height - 1 - y_u32) as f32 + jitter_y) / height_f;
                    let ray = camera.get_ray(u, v);
                    accumulated = accumulated + trace_ray(ray, scene, config.max_depth, debug);
                }
                *color_slot = accumulated / sample_count as f32;
            }
        });

    for y in 0..height {
        for x in 0..width {
            let color = color_buffer[(y * width) + x];
            image.put_pixel(x as u32, y as u32, to_rgb(color));
        }
    }

    image
}

fn trace_ray(ray: Ray, scene: Scene, depth: u8, debug: DebugOptions) -> Vec3 {
    if depth == 0 {
        return background_color(ray.direction, scene);
    }

    let Some(hit) = ray_march(ray, scene) else {
        return background_color(ray.direction, scene);
    };

    match hit.material {
        MaterialId::Floor => shade_floor(hit, scene),
        MaterialId::Glass => {
            if debug.force_opaque_red_menger {
                shade_opaque_red(hit, ray, scene, depth, debug)
            } else {
                shade_glass(hit, ray, scene, depth, debug)
            }
        }
        MaterialId::Mirror => shade_transparent_sphere(hit, ray, scene, depth, debug),
    }
}

fn shade_floor(hit: HitRecord, scene: Scene) -> Vec3 {
    let light_dir = (-scene.sun_direction).normalize();
    let lambert = hit.normal.dot(light_dir).max(0.0);
    let shadow = soft_shadow(
        scene,
        hit.point + (hit.normal * (RAY_BIAS * 1.5)),
        light_dir,
        0.02,
        24.0,
        24.0,
    );
    let ambient = 0.08;
    let direct = lambert * shadow;
    let shade = ambient + (0.92 * direct);

    let base = Vec3::new(0.94, 0.94, 0.93);
    let hemi = 0.5 * (hit.normal.y + 1.0);
    let bounce = Vec3::new(0.08, 0.1, 0.12) * (0.03 * hemi);
    let distance_fade = (1.0 - (hit.t * 0.012)).clamp(0.7, 1.0);

    ((base * shade * distance_fade) + bounce).clamp01()
}

fn shade_glass(hit: HitRecord, ray: Ray, scene: Scene, depth: u8, debug: DebugOptions) -> Vec3 {
    let ior = 1.52;
    let haze = 0.028;

    let incident = ray.direction.normalize();
    let outward = hit.normal;
    let entering = incident.dot(outward) < 0.0;
    let shading_normal = if entering { outward } else { -outward };

    let eta_i = if entering { 1.0 } else { ior };
    let eta_t = if entering { ior } else { 1.0 };
    let eta = eta_i / eta_t;
    let cos_i = (-incident).dot(shading_normal).clamp(0.0, 1.0);

    let reflect_dir =
        (reflect(incident, shading_normal) + (frosted_offset(hit.point, 0.0) * haze)).normalize();
    let reflect_origin = hit.point + (shading_normal * RAY_BIAS);
    let reflected = trace_ray(
        Ray {
            origin: reflect_origin,
            direction: reflect_dir,
        },
        scene,
        depth.saturating_sub(1),
        debug,
    );

    let mut fresnel = schlick(cos_i, eta_i, eta_t);
    let mut transmitted = Vec3::splat(0.0);
    if let Some(refract_dir_raw) = refract(incident, shading_normal, eta) {
        let refract_dir =
            (refract_dir_raw + (frosted_offset(hit.point, 1.0) * (haze * 0.75))).normalize();
        let refract_origin = if entering {
            hit.point - (shading_normal * RAY_BIAS)
        } else {
            hit.point + (shading_normal * RAY_BIAS)
        };

        let refracted = trace_ray(
            Ray {
                origin: refract_origin,
                direction: refract_dir,
            },
            scene,
            depth.saturating_sub(1),
            debug,
        );

        let travel = 1.0 / cos_i.max(0.2);
        let absorption = Vec3::new(0.07, 0.035, 0.015) * (travel * 0.7);
        let transmittance = Vec3::new(
            (-absorption.x).exp(),
            (-absorption.y).exp(),
            (-absorption.z).exp(),
        );
        transmitted = refracted * transmittance;
    } else {
        fresnel = 1.0;
    }

    let light_dir = (-scene.sun_direction).normalize();
    let sun_reflect = reflect(-light_dir, shading_normal);
    let sun_spec = sun_reflect.dot(-incident).max(0.0).powf(70.0);
    let sun_shadow = soft_shadow(
        scene,
        hit.point + (shading_normal * (RAY_BIAS * 1.5)),
        light_dir,
        0.02,
        24.0,
        24.0,
    );
    let specular = Vec3::splat(sun_spec * sun_shadow * 0.28);

    let tint = Vec3::new(0.96, 0.99, 1.0);
    let blended = (reflected * fresnel) + ((transmitted * tint) * (1.0 - fresnel));
    (blended + specular + Vec3::new(0.01, 0.015, 0.02)).clamp01()
}

fn shade_transparent_sphere(
    hit: HitRecord,
    ray: Ray,
    scene: Scene,
    depth: u8,
    debug: DebugOptions,
) -> Vec3 {
    let incident = ray.direction.normalize();
    let surface_normal = hit.normal;
    let reflection_share = 0.5;
    let transparency_share = 0.5;

    let reflection_dir = reflect(incident, surface_normal).normalize();
    let reflected = trace_ray(
        Ray {
            origin: hit.point + (surface_normal * RAY_BIAS),
            direction: reflection_dir,
        },
        scene,
        depth.saturating_sub(1),
        debug,
    );

    let transmitted = trace_scene_skipping_sphere(
        Ray {
            origin: hit.point - (surface_normal * RAY_BIAS),
            direction: incident,
        },
        scene,
        depth.saturating_sub(1),
        debug,
    );

    let light_dir = (-scene.sun_direction).normalize();
    let sun_reflect = reflect(-light_dir, surface_normal);
    let sun_spec = sun_reflect.dot(-incident).max(0.0).powf(240.0);
    let sun_shadow = soft_shadow(
        scene,
        hit.point + (surface_normal * (RAY_BIAS * 1.5)),
        light_dir,
        0.02,
        24.0,
        36.0,
    );
    let highlight = Vec3::splat(sun_spec * sun_shadow * 0.06);

    ((reflected * reflection_share) + (transmitted * transparency_share) + highlight).clamp01()
}

fn shade_opaque_red(
    hit: HitRecord,
    ray: Ray,
    scene: Scene,
    depth: u8,
    debug: DebugOptions,
) -> Vec3 {
    let light_dir = (-scene.sun_direction).normalize();
    let lambert = hit.normal.dot(light_dir).max(0.0);
    let shadow = soft_shadow(
        scene,
        hit.point + (hit.normal * (RAY_BIAS * 1.5)),
        light_dir,
        0.02,
        24.0,
        22.0,
    );
    let ao = ambient_occlusion(
        scene,
        hit.point + (hit.normal * (RAY_BIAS * 2.0)),
        hit.normal,
    );
    let hemi = 0.5 * (hit.normal.y + 1.0);
    let ambient = 0.03 + (0.18 * hemi);
    let diffuse = lambert * shadow;

    let view = (-ray.direction).normalize();
    let half_vec = (light_dir + view).normalize();
    let spec = hit.normal.dot(half_vec).max(0.0).powf(64.0) * shadow;

    let base = Vec3::new(0.9, 0.09, 0.08);
    let lit = base * (ambient + (0.92 * diffuse));
    let sky_tint = Vec3::new(0.24, 0.32, 0.44) * (0.12 * hemi);
    let bounce_tint = Vec3::new(0.18, 0.04, 0.03) * (0.08 * (1.0 - hemi));
    let highlight = Vec3::splat(spec * 0.2);
    let distance_fade = (1.0 - (hit.t * 0.014)).clamp(0.72, 1.0);
    let base_shaded =
        ((((lit + sky_tint + bounce_tint) * ao) + highlight) * distance_fade).clamp01();

    if depth <= 1 {
        return base_shaded;
    }

    let reflectivity = 0.2;
    let reflected = trace_ray(
        Ray {
            origin: hit.point + (hit.normal * RAY_BIAS),
            direction: reflect(ray.direction.normalize(), hit.normal).normalize(),
        },
        scene,
        depth.saturating_sub(1),
        debug,
    );

    ((base_shaded * (1.0 - reflectivity)) + (reflected * reflectivity)).clamp01()
}

fn trace_scene_skipping_sphere(ray: Ray, scene: Scene, depth: u8, debug: DebugOptions) -> Vec3 {
    if depth == 0 {
        return background_color(ray.direction, scene);
    }

    let mut current_ray = ray;
    for _ in 0..6 {
        let Some(hit) = ray_march(current_ray, scene) else {
            return background_color(current_ray.direction, scene);
        };

        match hit.material {
            MaterialId::Floor => return shade_floor(hit, scene),
            MaterialId::Glass => {
                if debug.force_opaque_red_menger {
                    return shade_opaque_red(hit, current_ray, scene, depth, debug);
                }
                return shade_glass(hit, current_ray, scene, depth, debug);
            }
            MaterialId::Mirror => {
                current_ray = Ray {
                    origin: hit.point + (current_ray.direction * (RAY_BIAS * 2.0)),
                    direction: current_ray.direction,
                };
            }
        }
    }

    background_color(current_ray.direction, scene)
}

fn ray_march(ray: Ray, scene: Scene) -> Option<HitRecord> {
    let mut t = 0.0;
    for _ in 0..MAX_MARCH_STEPS {
        if t > MAX_TRACE_DISTANCE {
            return None;
        }

        let p = ray.at(t);
        let sample = scene.sample(p);
        if sample.distance.abs() < HIT_EPSILON {
            let normal = estimate_normal(scene, p);
            return Some(HitRecord {
                t,
                point: p,
                normal,
                material: sample.material,
            });
        }

        t += sample.distance.abs().max(0.0003);
    }
    None
}

fn soft_shadow(scene: Scene, origin: Vec3, direction: Vec3, min_t: f32, max_t: f32, k: f32) -> f32 {
    let mut attenuation: f32 = 1.0;
    let mut t = min_t;

    for _ in 0..96 {
        if t >= max_t {
            break;
        }

        let p = origin + (direction * t);
        let h = scene.distance(p);
        if h < (HIT_EPSILON * 0.9) {
            return 0.0;
        }

        attenuation = attenuation.min((k * h / t).clamp(0.0, 1.0));
        t += h.clamp(0.015, 0.45);
    }

    attenuation.clamp(0.0, 1.0)
}

fn ambient_occlusion(scene: Scene, origin: Vec3, normal: Vec3) -> f32 {
    let mut occlusion = 0.0;
    let mut weight = 1.0;
    let mut distance = 0.02;

    for _ in 0..6 {
        let sample_point = origin + (normal * distance);
        let sdf = scene.distance(sample_point);
        occlusion += ((distance - sdf).max(0.0)) * weight;
        weight *= 0.65;
        distance += 0.03;
    }

    (1.0 - (occlusion * 1.7)).clamp(0.0, 1.0)
}

fn estimate_normal(scene: Scene, p: Vec3) -> Vec3 {
    let e = NORMAL_EPSILON;
    let dx =
        scene.distance(p + Vec3::new(e, 0.0, 0.0)) - scene.distance(p - Vec3::new(e, 0.0, 0.0));
    let dy =
        scene.distance(p + Vec3::new(0.0, e, 0.0)) - scene.distance(p - Vec3::new(0.0, e, 0.0));
    let dz =
        scene.distance(p + Vec3::new(0.0, 0.0, e)) - scene.distance(p - Vec3::new(0.0, 0.0, e));
    Vec3::new(dx, dy, dz).normalize()
}

fn background_color(direction: Vec3, scene: Scene) -> Vec3 {
    let unit = direction.normalize();
    let t = 0.5 * (unit.y + 1.0);
    let top = Vec3::new(0.5, 0.71, 0.94);
    let bottom = Vec3::new(0.98, 0.99, 1.0);
    let base = (bottom * (1.0 - t)) + (top * t);

    let sun_alignment = unit.dot(-scene.sun_direction).max(0.0);
    let sun = Vec3::new(1.0, 0.96, 0.9) * sun_alignment.powf(420.0) * 6.0;
    (base + sun).clamp01()
}

fn to_rgb(color: Vec3) -> Rgb<u8> {
    let mapped = filmic_tone_map(color);
    let corrected = Vec3::new(
        mapped.x.powf(1.0 / 2.2),
        mapped.y.powf(1.0 / 2.2),
        mapped.z.powf(1.0 / 2.2),
    )
    .clamp01();
    let r = (corrected.x * 255.999) as u8;
    let g = (corrected.y * 255.999) as u8;
    let b = (corrected.z * 255.999) as u8;
    Rgb([r, g, b])
}

fn filmic_curve(x: f32) -> f32 {
    let clamped = x.max(0.0);
    let numerator = clamped * ((2.51 * clamped) + 0.03);
    let denominator = clamped * ((2.43 * clamped) + 0.59) + 0.14;
    (numerator / denominator).clamp(0.0, 1.0)
}

fn filmic_tone_map(color: Vec3) -> Vec3 {
    Vec3::new(
        filmic_curve(color.x),
        filmic_curve(color.y),
        filmic_curve(color.z),
    )
}
