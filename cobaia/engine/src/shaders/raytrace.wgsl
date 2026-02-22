struct Params {
    width: u32,
    height: u32,
    max_depth: u32,
    sponge_iterations: u32,
    samples_per_pixel: u32,
    _padding_u0: u32,
    _padding_u1: u32,
    _padding_u2: u32,
    camera_origin: vec4<f32>,
    camera_target: vec4<f32>,
    camera_up: vec4<f32>,
    sponge_center: vec4<f32>,
    sun_direction: vec4<f32>,
    sun_color_intensity: vec4<f32>,
    floor_base_color: vec4<f32>,
    menger_base_color: vec4<f32>,
    mirror_sphere_center: vec4<f32>,
    scene_scalars: vec4<f32>,
};

@group(0) @binding(0) var output_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> params: Params;

const MAX_STEPS: u32 = 280u;
const MAX_TRACE_DISTANCE: f32 = 42.0;
const HIT_EPSILON: f32 = 0.00035;
const NORMAL_EPSILON: f32 = 0.001;
const RAY_BIAS: f32 = 0.003;

struct Hit {
    t: f32,
    point: vec3<f32>,
    normal: vec3<f32>,
    material: f32,
};

fn rem_euclid3(v: vec3<f32>, rhs: f32) -> vec3<f32> {
    return v - floor(v / rhs) * rhs;
}

fn sd_box(p: vec3<f32>, half_extents: vec3<f32>) -> f32 {
    let q = abs(p) - half_extents;
    let outside = max(q, vec3<f32>(0.0));
    return length(outside) + min(max(max(q.x, q.y), q.z), 0.0);
}

fn sd_sphere(p: vec3<f32>, radius: f32) -> f32 {
    return length(p) - radius;
}

fn sd_menger(p: vec3<f32>) -> f32 {
    var distance = sd_box(p, vec3<f32>(1.0));
    var scale = 1.0;
    var i: u32 = 0u;
    loop {
        if (i >= params.sponge_iterations) {
            break;
        }
        let cell = rem_euclid3(p * scale, 2.0) - vec3<f32>(1.0);
        scale = scale * 3.0;
        let r = abs(vec3<f32>(1.0) - (abs(cell) * 3.0));
        let da = max(r.x, r.y);
        let db = max(r.y, r.z);
        let dc = max(r.x, r.z);
        let carved = (min(min(da, db), dc) - 1.0) / scale;
        distance = max(distance, carved);
        i = i + 1u;
    }
    return distance;
}

fn sample_scene(p: vec3<f32>) -> vec2<f32> {
    let floor_y = params.scene_scalars.x;
    let sponge_scale = params.scene_scalars.y;
    let mirror_sphere_radius = params.scene_scalars.z;
    let floor_distance = p.y - floor_y;
    let local = (p - params.sponge_center.xyz) / sponge_scale;
    let sponge_distance = sd_menger(local) * sponge_scale;
    let mirror_sphere_distance = sd_sphere(
        p - params.mirror_sphere_center.xyz,
        mirror_sphere_radius
    );

    var best = vec2<f32>(floor_distance, 0.0);
    if (sponge_distance < best.x) {
        best = vec2<f32>(sponge_distance, 1.0);
    }
    if (mirror_sphere_distance < best.x) {
        best = vec2<f32>(mirror_sphere_distance, 2.0);
    }
    return best;
}

fn scene_distance(p: vec3<f32>) -> f32 {
    return sample_scene(p).x;
}

fn estimate_normal(p: vec3<f32>) -> vec3<f32> {
    let e = NORMAL_EPSILON;
    let dx = scene_distance(p + vec3<f32>(e, 0.0, 0.0)) - scene_distance(p - vec3<f32>(e, 0.0, 0.0));
    let dy = scene_distance(p + vec3<f32>(0.0, e, 0.0)) - scene_distance(p - vec3<f32>(0.0, e, 0.0));
    let dz = scene_distance(p + vec3<f32>(0.0, 0.0, e)) - scene_distance(p - vec3<f32>(0.0, 0.0, e));
    return normalize(vec3<f32>(dx, dy, dz));
}

fn ray_march(origin: vec3<f32>, direction: vec3<f32>) -> Hit {
    var t = 0.0;
    var i: u32 = 0u;
    loop {
        if (i >= MAX_STEPS) {
            break;
        }
        if (t > MAX_TRACE_DISTANCE) {
            break;
        }

        let p = origin + (direction * t);
        let sample = sample_scene(p);
        if (abs(sample.x) < HIT_EPSILON) {
            return Hit(t, p, estimate_normal(p), sample.y);
        }

        t = t + max(abs(sample.x), 0.0003);
        i = i + 1u;
    }

    return Hit(-1.0, vec3<f32>(0.0), vec3<f32>(0.0, 1.0, 0.0), 0.0);
}

fn soft_shadow(origin: vec3<f32>, direction: vec3<f32>, min_t: f32, max_t: f32, k: f32) -> f32 {
    var attenuation = 1.0;
    var t = min_t;
    var i: u32 = 0u;
    loop {
        if (i >= 96u) {
            break;
        }
        if (t >= max_t) {
            break;
        }

        let p = origin + (direction * t);
        let h = scene_distance(p);
        if (h < (HIT_EPSILON * 0.9)) {
            return 0.0;
        }

        attenuation = min(attenuation, clamp(k * h / t, 0.0, 1.0));
        t = t + clamp(h, 0.015, 0.45);
        i = i + 1u;
    }
    return clamp(attenuation, 0.0, 1.0);
}

fn ambient_occlusion(origin: vec3<f32>, normal: vec3<f32>) -> f32 {
    var occlusion = 0.0;
    var weight = 1.0;
    var distance = 0.02;
    var i: u32 = 0u;
    loop {
        if (i >= 6u) {
            break;
        }
        let sample_point = origin + (normal * distance);
        let sdf = scene_distance(sample_point);
        occlusion = occlusion + max(distance - sdf, 0.0) * weight;
        weight = weight * 0.65;
        distance = distance + 0.03;
        i = i + 1u;
    }
    return clamp(1.0 - (occlusion * 1.7), 0.0, 1.0);
}

fn sun_radiance() -> vec3<f32> {
    return params.sun_color_intensity.xyz * params.sun_color_intensity.w;
}

fn background_color(direction: vec3<f32>) -> vec3<f32> {
    let unit = normalize(direction);
    let t = 0.5 * (unit.y + 1.0);
    let top = vec3<f32>(0.5, 0.71, 0.94);
    let bottom = vec3<f32>(0.98, 0.99, 1.0);
    let base = (bottom * (1.0 - t)) + (top * t);

    let sun_alignment = max(dot(unit, -params.sun_direction.xyz), 0.0);
    let sun = sun_radiance() * pow(sun_alignment, 420.0) * 6.0;
    return clamp(base + sun, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn shade_floor(hit: Hit) -> vec3<f32> {
    let light_dir = normalize(-params.sun_direction.xyz);
    let lambert = max(dot(hit.normal, light_dir), 0.0);
    let shadow = soft_shadow(
        hit.point + (hit.normal * (RAY_BIAS * 1.5)),
        light_dir,
        0.02,
        24.0,
        24.0
    );
    let ambient = 0.08;
    let direct = lambert * shadow;
    let sunlight = sun_radiance();

    let base = params.floor_base_color.xyz;
    let hemi = 0.5 * (hit.normal.y + 1.0);
    let bounce = vec3<f32>(0.08, 0.1, 0.12) * (0.03 * hemi);
    let distance_fade = clamp(1.0 - (hit.t * 0.012), 0.7, 1.0);
    let lit = ((base * ambient) + ((base * (0.92 * direct)) * sunlight)) * distance_fade;

    return clamp(lit + bounce, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn shade_opaque_red(hit: Hit, ray_dir: vec3<f32>) -> vec3<f32> {
    let light_dir = normalize(-params.sun_direction.xyz);
    let lambert = max(dot(hit.normal, light_dir), 0.0);
    let shadow = soft_shadow(
        hit.point + (hit.normal * (RAY_BIAS * 1.5)),
        light_dir,
        0.02,
        24.0,
        22.0
    );
    let ao = ambient_occlusion(hit.point + (hit.normal * (RAY_BIAS * 2.0)), hit.normal);
    let hemi = 0.5 * (hit.normal.y + 1.0);
    let ambient = 0.03 + (0.18 * hemi);
    let diffuse = lambert * shadow;

    let view = normalize(-ray_dir);
    let half_vec = normalize(light_dir + view);
    let spec = pow(max(dot(hit.normal, half_vec), 0.0), 64.0) * shadow;
    let sunlight = sun_radiance();

    let base = params.menger_base_color.xyz;
    let lit = (base * ambient) + ((base * (0.92 * diffuse)) * sunlight);
    let sky_tint = vec3<f32>(0.24, 0.32, 0.44) * (0.12 * hemi);
    let bounce_tint = vec3<f32>(0.18, 0.04, 0.03) * (0.08 * (1.0 - hemi));
    let highlight = sunlight * (spec * 0.2);
    let distance_fade = clamp(1.0 - (hit.t * 0.014), 0.72, 1.0);

    return clamp((((lit + sky_tint + bounce_tint) * ao) + highlight) * distance_fade, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn shade_sphere_lighting(hit: Hit, ray_dir: vec3<f32>) -> vec3<f32> {
    let light_dir = normalize(-params.sun_direction.xyz);
    let sun_reflect = reflect(-light_dir, hit.normal);
    let sun_spec = pow(max(dot(sun_reflect, normalize(-ray_dir)), 0.0), 240.0);
    let sun_shadow = soft_shadow(
        hit.point + (hit.normal * (RAY_BIAS * 1.5)),
        light_dir,
        0.02,
        24.0,
        36.0
    );
    let highlight = sun_radiance() * (sun_spec * sun_shadow * 0.06);
    return highlight;
}

fn trace_scene_skipping_sphere(origin_in: vec3<f32>, direction_in: vec3<f32>, max_skips: u32) -> vec3<f32> {
    var origin = origin_in;
    let direction = normalize(direction_in);
    var skips: u32 = 0u;
    loop {
        if (skips >= 6u || skips >= max_skips) {
            return background_color(direction);
        }

        let hit = ray_march(origin, direction);
        if (hit.t < 0.0) {
            return background_color(direction);
        }
        if (hit.material < 0.5) {
            return shade_floor(hit);
        }
        if (hit.material < 1.5) {
            return shade_opaque_red(hit, direction);
        }

        origin = hit.point + (direction * (RAY_BIAS * 2.0));
        skips = skips + 1u;
    }

    return background_color(direction);
}

fn trace_ray(origin_in: vec3<f32>, direction_in: vec3<f32>) -> vec3<f32> {
    var origin = origin_in;
    var direction = normalize(direction_in);
    var throughput = vec3<f32>(1.0);
    var accumulated = vec3<f32>(0.0);
    let max_bounces = max(params.max_depth, 1u);
    let cube_reflectivity = 0.2;
    let sphere_transparency = 0.5;
    let sphere_reflection = 1.0 - sphere_transparency;

    var bounce: u32 = 0u;
    loop {
        if (bounce >= max_bounces) {
            accumulated = accumulated + (throughput * background_color(direction));
            break;
        }

        let hit = ray_march(origin, direction);
        if (hit.t < 0.0) {
            accumulated = accumulated + (throughput * background_color(direction));
            break;
        }

        if (hit.material < 0.5) {
            accumulated = accumulated + (throughput * shade_floor(hit));
            break;
        }

        if (hit.material < 1.5) {
            let cube_base = shade_opaque_red(hit, direction);
            accumulated = accumulated + (throughput * cube_base * (1.0 - cube_reflectivity));
            throughput = throughput * cube_reflectivity;
            if (max(max(throughput.x, throughput.y), throughput.z) < 0.001) {
                break;
            }

            origin = hit.point + (hit.normal * RAY_BIAS);
            direction = normalize(reflect(direction, hit.normal));
            bounce = bounce + 1u;
            continue;
        }

        let transmitted = trace_scene_skipping_sphere(
            hit.point - (hit.normal * RAY_BIAS),
            direction,
            max_bounces - bounce
        );
        let sphere_lighting = shade_sphere_lighting(hit, direction);
        accumulated = accumulated + (throughput * ((transmitted * sphere_transparency) + sphere_lighting));
        throughput = throughput * sphere_reflection;

        if (max(max(throughput.x, throughput.y), throughput.z) < 0.001) {
            break;
        }

        origin = hit.point + (hit.normal * RAY_BIAS);
        direction = normalize(reflect(direction, hit.normal));
        bounce = bounce + 1u;
    }

    return accumulated;
}

fn hash_u32(value: u32) -> u32 {
    var v = value;
    v = v ^ (v >> 16u);
    v = v * 0x7feb352du;
    v = v ^ (v >> 15u);
    v = v * 0x846ca68bu;
    v = v ^ (v >> 16u);
    return v;
}

fn random01(seed: u32) -> f32 {
    return f32(hash_u32(seed)) / 4294967295.0;
}

fn sample_jitter(x: u32, y: u32, sample_index: u32, axis: u32) -> f32 {
    let seed = (x * 1973u) + (y * 9277u) + (sample_index * 26699u) + (axis * 104729u) ^ 0x68bc21ebu;
    return random01(seed);
}

fn filmic_tone_map(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    let safe_color = max(color, vec3<f32>(0.0));
    let mapped = (safe_color * (a * safe_color + b)) / (safe_color * (c * safe_color + d) + e);
    return clamp(mapped, vec3<f32>(0.0), vec3<f32>(1.0));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    let origin = params.camera_origin.xyz;
    let camera_target = params.camera_target.xyz;
    let width_f = max(f32(params.width), 1.0);
    let height_f = max(f32(params.height), 1.0);
    let aspect_ratio = width_f / height_f;

    let theta = radians(max(params.scene_scalars.w, 1.0));
    let h = tan(theta * 0.5);
    let viewport_height = 2.0 * h;
    let viewport_width = aspect_ratio * viewport_height;

    let w = normalize(origin - camera_target);
    var up = params.camera_up.xyz;
    if (length(up) < 0.0001) {
        up = vec3<f32>(0.0, 1.0, 0.0);
    } else {
        up = normalize(up);
    }
    if (abs(dot(up, w)) > 0.999) {
        up = vec3<f32>(0.0, 0.0, 1.0);
    }
    let right = normalize(cross(up, w));
    let up_cam = cross(w, right);

    let horizontal = right * viewport_width;
    let vertical = up_cam * viewport_height;
    let lower_left = origin - (horizontal * 0.5) - (vertical * 0.5) - w;
    let sample_count = max(params.samples_per_pixel, 1u);
    var color = vec3<f32>(0.0);
    var sample_index: u32 = 0u;
    loop {
        if (sample_index >= sample_count) {
            break;
        }
        let jitter_x = sample_jitter(gid.x, gid.y, sample_index, 0u);
        let jitter_y = sample_jitter(gid.x, gid.y, sample_index, 1u);
        let u = (f32(gid.x) + jitter_x) / width_f;
        let v = (f32((params.height - 1u) - gid.y) + jitter_y) / height_f;
        let direction = normalize(lower_left + (horizontal * u) + (vertical * v) - origin);
        color = color + trace_ray(origin, direction);
        sample_index = sample_index + 1u;
    }
    color = color / f32(sample_count);
    color = filmic_tone_map(color);
    color = pow(color, vec3<f32>(1.0 / 2.2));
    textureStore(output_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
