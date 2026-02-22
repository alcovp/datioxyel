const MAX_OBJECTS: u32 = __GPU_MAX_OBJECTS__;
const MAX_MATERIALS: u32 = __GPU_MAX_MATERIALS__;
const MAX_LIGHTS: u32 = __GPU_MAX_LIGHTS__;

const OBJECT_KIND_PLANE: u32 = 0u;
const OBJECT_KIND_MENGER: u32 = 1u;
const OBJECT_KIND_SPHERE: u32 = 2u;
const OBJECT_KIND_PARALLELEPIPED: u32 = 3u;
const OBJECT_KIND_CYLINDER: u32 = 4u;
const OBJECT_KIND_PYRAMID: u32 = 5u;
const OBJECT_KIND_CAPSULE: u32 = 6u;
const OBJECT_KIND_FRUSTUM: u32 = 7u;
const OBJECT_KIND_TORUS: u32 = 8u;
const OBJECT_KIND_ROUNDED_BOX: u32 = 9u;
const OBJECT_KIND_ELLIPSOID: u32 = 10u;

struct Params {
    width: u32,
    height: u32,
    max_depth: u32,
    samples_per_pixel: u32,
    object_count: u32,
    material_count: u32,
    light_count: u32,
    _padding_u0: u32,
    camera_origin: vec4<f32>,
    camera_target: vec4<f32>,
    camera_up: vec4<f32>,
    object_meta: array<vec4<f32>, MAX_OBJECTS>,
    object_data0: array<vec4<f32>, MAX_OBJECTS>,
    object_data1: array<vec4<f32>, MAX_OBJECTS>,
    material_albedo_roughness: array<vec4<f32>, MAX_MATERIALS>,
    material_emission_metallic: array<vec4<f32>, MAX_MATERIALS>,
    material_optics: array<vec4<f32>, MAX_MATERIALS>,
    material_absorption: array<vec4<f32>, MAX_MATERIALS>,
    light_direction: array<vec4<f32>, MAX_LIGHTS>,
    light_color_intensity: array<vec4<f32>, MAX_LIGHTS>,
    scene_scalars: vec4<f32>,
    render_tuning0: vec4<f32>,
    render_tuning1: vec4<f32>,
    render_tuning2: vec4<f32>,
    render_tuning3: vec4<f32>,
};

@group(0) @binding(0) var output_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> params: Params;

const MAX_STEPS: u32 = 280u;
const MAX_TRACE_DISTANCE: f32 = 42.0;
const HIT_EPSILON: f32 = 0.00035;
const NORMAL_EPSILON: f32 = 0.001;
const RAY_BIAS: f32 = 0.003;
const RR_MIN_SURVIVE: f32 = 0.05;
const RR_MAX_SURVIVE: f32 = 0.98;
const SHADOW_MAX_STEPS: u32 = 80u;
const AO_MAX_SAMPLES: u32 = 6u;
const AO_EARLY_EXIT: f32 = 0.72;
const MENGER_DETAIL_BAND_FACTOR: f32 = 0.14;
const MENGER_DETAIL_BAND_MIN: f32 = 0.035;
const INV_U32_SCALE: f32 = 2.3283064365386963e-10;

struct SceneSample {
    distance: f32,
    object_index: u32,
    material_index: u32,
    kind: u32,
};

struct Hit {
    t: f32,
    point: vec3<f32>,
    normal: vec3<f32>,
    object_index: u32,
    material_index: u32,
    kind: u32,
};

fn as_u32_rounded(value: f32) -> u32 {
    return u32(max(value, 0.0) + 0.5);
}

fn safe_material_index(index: u32) -> u32 {
    if (params.material_count == 0u) {
        return 0u;
    }
    return min(index, params.material_count - 1u);
}

fn light_count_capped() -> u32 {
    return min(params.light_count, MAX_LIGHTS);
}

fn material_albedo_roughness(index: u32) -> vec4<f32> {
    return params.material_albedo_roughness[safe_material_index(index)];
}

fn material_emission_metallic(index: u32) -> vec4<f32> {
    return params.material_emission_metallic[safe_material_index(index)];
}

fn material_ior(index: u32) -> f32 {
    return max(params.material_optics[safe_material_index(index)].x, 1.0);
}

fn material_transmission(index: u32) -> f32 {
    return clamp(params.material_optics[safe_material_index(index)].y, 0.0, 1.0);
}

fn material_absorption(index: u32) -> vec3<f32> {
    return max(params.material_absorption[safe_material_index(index)].xyz, vec3<f32>(0.0));
}

fn light_dir(index: u32) -> vec3<f32> {
    return normalize(-params.light_direction[index].xyz);
}

fn light_radiance(index: u32) -> vec3<f32> {
    return params.light_color_intensity[index].xyz * params.light_color_intensity[index].w;
}

fn runtime_march_max_steps() -> u32 {
    let value = as_u32_rounded(params.render_tuning0.x);
    return clamp(value, 1u, MAX_STEPS);
}

fn runtime_hit_epsilon_scale() -> f32 {
    return clamp(params.render_tuning0.y, 0.2, 5.0);
}

fn runtime_step_scale() -> f32 {
    return clamp(params.render_tuning0.z, 0.2, 3.0);
}

fn runtime_rr_start_bounce() -> u32 {
    return clamp(as_u32_rounded(params.render_tuning0.w), 0u, 32u);
}

fn runtime_shadow_max_steps() -> u32 {
    let value = as_u32_rounded(params.render_tuning1.x);
    return clamp(value, 1u, SHADOW_MAX_STEPS);
}

fn runtime_ao_samples() -> u32 {
    let value = as_u32_rounded(params.render_tuning1.y);
    return clamp(value, 1u, AO_MAX_SAMPLES);
}

fn runtime_sampling_mode() -> u32 {
    return min(as_u32_rounded(params.render_tuning1.z), 2u);
}

fn runtime_adaptive_min_samples_fraction() -> f32 {
    return clamp(params.render_tuning2.x, 0.05, 1.0);
}

fn runtime_adaptive_variance_threshold() -> f32 {
    return clamp(params.render_tuning2.y, 0.0, 0.5);
}

fn runtime_adaptive_check_interval() -> u32 {
    return clamp(as_u32_rounded(params.render_tuning2.z), 1u, 8u);
}

fn runtime_firefly_clamp_scale() -> f32 {
    return clamp(params.render_tuning2.w, 0.2, 3.0);
}

fn runtime_shadow_distance_scale() -> f32 {
    return clamp(params.render_tuning3.x, 0.4, 2.0);
}

fn runtime_shadow_min_step_scale() -> f32 {
    return clamp(params.render_tuning3.y, 0.5, 2.5);
}

fn runtime_ao_radius_scale() -> f32 {
    return clamp(params.render_tuning3.z, 0.4, 2.0);
}

fn runtime_ao_horizon_bias() -> f32 {
    return clamp(params.render_tuning3.w, 0.65, 1.35);
}

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

fn sd_capped_cylinder_y(p: vec3<f32>, radius: f32, half_height: f32) -> f32 {
    let q = vec2<f32>(length(p.xz) - radius, abs(p.y) - half_height);
    let outside = max(q, vec2<f32>(0.0));
    return length(outside) + min(max(q.x, q.y), 0.0);
}

fn sd_capsule_y(p: vec3<f32>, radius: f32, half_height: f32) -> f32 {
    let y_clamped = clamp(p.y, -half_height, half_height);
    let closest = vec3<f32>(0.0, y_clamped, 0.0);
    return length(p - closest) - radius;
}

fn sd_capped_cone_y(
    p: vec3<f32>,
    half_height: f32,
    radius_bottom: f32,
    radius_top: f32,
) -> f32 {
    let y01 = clamp((p.y + half_height) / max(2.0 * half_height, 0.0001), 0.0, 1.0);
    let radius_at_y = mix(radius_bottom, radius_top, y01);
    let radial = length(p.xz) - radius_at_y;
    let caps = abs(p.y) - half_height;
    let outside = vec2<f32>(max(radial, 0.0), max(caps, 0.0));
    return length(outside) + min(max(radial, caps), 0.0);
}

fn sd_torus_y(p: vec3<f32>, major_radius: f32, minor_radius: f32) -> f32 {
    let q = vec2<f32>(length(p.xz) - major_radius, p.y);
    return length(q) - minor_radius;
}

fn sd_rounded_box(p: vec3<f32>, half_extents: vec3<f32>, radius: f32) -> f32 {
    let inner = max(half_extents - vec3<f32>(radius), vec3<f32>(0.0));
    let q = abs(p) - inner;
    let outside = max(q, vec3<f32>(0.0));
    return length(outside) + min(max(max(q.x, q.y), q.z), 0.0) - radius;
}

// This is a conservative signed distance estimate for a square pyramid aligned to +Y.
// It stays safe for sphere tracing (never oversteps), at the cost of slightly shorter steps near edges.
fn sd_pyramid_y(p: vec3<f32>, half_extent: f32, height: f32) -> f32 {
    let h = max(height, 0.0001);
    let half_h = h * 0.5;
    let side_slope = half_extent / h;
    let side_norm = sqrt(1.0 + (side_slope * side_slope));
    let side_x = (abs(p.x) + (side_slope * p.y) - (side_slope * half_h)) / side_norm;
    let side_z = (abs(p.z) + (side_slope * p.y) - (side_slope * half_h)) / side_norm;
    let base = -(p.y + half_h);
    return max(base, max(side_x, side_z));
}

fn sd_menger(p: vec3<f32>, iterations: u32) -> f32 {
    var distance = sd_box(p, vec3<f32>(1.0));
    var scale = 1.0;
    var i: u32 = 0u;
    loop {
        if (i >= iterations) {
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

fn object_distance_lower_bound(kind: u32, p: vec3<f32>, data0: vec4<f32>, data1: vec4<f32>) -> f32 {
    if (kind == OBJECT_KIND_PLANE) {
        return p.y - data0.x;
    }
    if (kind == OBJECT_KIND_PARALLELEPIPED) {
        let half_extents = vec3<f32>(
            max(data0.w, 0.0001),
            max(data1.x, 0.0001),
            max(data1.y, 0.0001),
        );
        return sd_box(p - data0.xyz, half_extents);
    }
    if (kind == OBJECT_KIND_MENGER) {
        let scale = max(data0.w, 0.0001);
        let local = (p - data0.xyz) / scale;
        return sd_box(local, vec3<f32>(1.0)) * scale;
    }
    if (kind == OBJECT_KIND_SPHERE) {
        return sd_sphere(p - data0.xyz, data0.w);
    }
    if (kind == OBJECT_KIND_CYLINDER) {
        let half_height = max(data1.x, 0.0001);
        return sd_capped_cylinder_y(p - data0.xyz, max(data0.w, 0.0001), half_height);
    }
    if (kind == OBJECT_KIND_CAPSULE) {
        let half_height = max(data1.x, 0.0001);
        return sd_capsule_y(p - data0.xyz, max(data0.w, 0.0001), half_height);
    }
    if (kind == OBJECT_KIND_FRUSTUM) {
        let half_height = max(data0.w, 0.0001);
        let radius_bottom = max(data1.x, 0.0);
        let radius_top = max(data1.y, 0.0);
        return sd_capped_cone_y(p - data0.xyz, half_height, radius_bottom, radius_top);
    }
    if (kind == OBJECT_KIND_TORUS) {
        let major_radius = max(data0.w, 0.0001);
        let minor_radius = max(data1.x, 0.0001);
        return sd_torus_y(p - data0.xyz, major_radius, minor_radius);
    }
    if (kind == OBJECT_KIND_ROUNDED_BOX) {
        let half_extents = vec3<f32>(
            max(data0.w, 0.0001),
            max(data1.x, 0.0001),
            max(data1.y, 0.0001),
        );
        let radius = clamp(data1.z, 0.0, min(min(half_extents.x, half_extents.y), half_extents.z) - 0.0001);
        return sd_rounded_box(p - data0.xyz, half_extents, max(radius, 0.0));
    }
    if (kind == OBJECT_KIND_ELLIPSOID) {
        let radii = vec3<f32>(
            max(data0.w, 0.0001),
            max(data1.x, 0.0001),
            max(data1.y, 0.0001),
        );
        let bound_radius = max(max(radii.x, radii.y), radii.z);
        return sd_sphere(p - data0.xyz, bound_radius);
    }
    if (kind == OBJECT_KIND_PYRAMID) {
        let height = max(data1.x, 0.0001);
        return sd_pyramid_y(p - data0.xyz, max(data0.w, 0.0001), height);
    }
    return MAX_TRACE_DISTANCE + 1.0;
}

fn object_distance_refined(kind: u32, p: vec3<f32>, data0: vec4<f32>, data1: vec4<f32>) -> f32 {
    if (kind == OBJECT_KIND_MENGER) {
        let scale = max(data0.w, 0.0001);
        let local = (p - data0.xyz) / scale;
        let iterations = max(as_u32_rounded(data1.x), 1u);
        return sd_menger(local, iterations) * scale;
    }
    return object_distance_lower_bound(kind, p, data0, data1);
}

fn sample_scene(p: vec3<f32>) -> SceneSample {
    if (params.object_count == 0u) {
        return SceneSample(MAX_TRACE_DISTANCE + 1.0, 0u, 0u, OBJECT_KIND_PLANE);
    }

    var best = SceneSample(MAX_TRACE_DISTANCE + 1.0, 0u, 0u, OBJECT_KIND_PLANE);
    var i: u32 = 0u;
    loop {
        if (i >= MAX_OBJECTS || i >= params.object_count) {
            break;
        }

        let object_meta_value = params.object_meta[i];
        let kind = as_u32_rounded(object_meta_value.x);
        let material_index = as_u32_rounded(object_meta_value.y);
        let data0 = params.object_data0[i];
        let data1 = params.object_data1[i];

        // Broad-phase culling: use conservative lower bounds to skip expensive DE eval.
        let lower_bound = object_distance_lower_bound(kind, p, data0, data1);
        if (lower_bound > best.distance) {
            i = i + 1u;
            continue;
        }

        var distance = lower_bound;
        if (kind == OBJECT_KIND_MENGER) {
            let detail_band = max(data0.w * MENGER_DETAIL_BAND_FACTOR, MENGER_DETAIL_BAND_MIN);
            if (lower_bound <= detail_band) {
                distance = object_distance_refined(kind, p, data0, data1);
            }
        }

        if (distance < best.distance) {
            best = SceneSample(distance, i, material_index, kind);
        }

        i = i + 1u;
    }

    return best;
}

fn scene_distance(p: vec3<f32>) -> f32 {
    return sample_scene(p).distance;
}

fn march_hit_epsilon(distance_along_ray: f32) -> f32 {
    let scale = runtime_hit_epsilon_scale();
    let adaptive = min(sqrt(max(distance_along_ray, 0.0)) * 0.00007, 0.0011);
    return (HIT_EPSILON + adaptive) * scale;
}

fn march_step_size(sdf_distance: f32, distance_along_ray: f32) -> f32 {
    let step_scale = runtime_step_scale();
    let abs_distance = abs(sdf_distance);
    let far_boost = (1.0 + min(distance_along_ray * 0.025, 1.2)) * step_scale;
    let near_surface_safety = mix(0.72, 1.0, clamp(abs_distance * 18.0, 0.0, 1.0));
    let min_step = 0.0002 + min(distance_along_ray * 0.00004, 0.002);
    let max_step = 0.55 + min(distance_along_ray * 0.08, 5.5);
    let candidate = abs_distance * far_boost * near_surface_safety;
    return clamp(candidate, min_step, max_step);
}

fn normal_epsilon(distance_along_ray: f32) -> f32 {
    let scale = runtime_hit_epsilon_scale();
    let adaptive = NORMAL_EPSILON + min(distance_along_ray * 0.00005, 0.0012);
    return clamp(adaptive * (0.75 + (0.25 * scale)), 0.0005, 0.0035);
}

fn estimate_normal(p: vec3<f32>, distance_along_ray: f32) -> vec3<f32> {
    let e = normal_epsilon(distance_along_ray);
    let k1 = vec3<f32>(1.0, -1.0, -1.0);
    let k2 = vec3<f32>(-1.0, -1.0, 1.0);
    let k3 = vec3<f32>(-1.0, 1.0, -1.0);
    let k4 = vec3<f32>(1.0, 1.0, 1.0);
    let n =
        (k1 * scene_distance(p + (k1 * e))) +
        (k2 * scene_distance(p + (k2 * e))) +
        (k3 * scene_distance(p + (k3 * e))) +
        (k4 * scene_distance(p + (k4 * e)));
    return normalize(n);
}

fn ray_march(origin: vec3<f32>, direction: vec3<f32>) -> Hit {
    let max_steps = runtime_march_max_steps();
    var t = 0.0;
    var i: u32 = 0u;
    loop {
        if (i >= max_steps) {
            break;
        }
        if (t > MAX_TRACE_DISTANCE) {
            break;
        }

        let p = origin + (direction * t);
        let sample = sample_scene(p);
        if (abs(sample.distance) < march_hit_epsilon(t)) {
            return Hit(t, p, estimate_normal(p, t), sample.object_index, sample.material_index, sample.kind);
        }

        let remaining = MAX_TRACE_DISTANCE - t;
        if (sample.distance > remaining) {
            break;
        }

        t = t + march_step_size(sample.distance, t);
        i = i + 1u;
    }

    return Hit(-1.0, vec3<f32>(0.0), vec3<f32>(0.0, 1.0, 0.0), 0u, 0u, OBJECT_KIND_PLANE);
}

fn soft_shadow(
    origin: vec3<f32>,
    direction: vec3<f32>,
    min_t: f32,
    max_t: f32,
    roughness: f32,
    penumbra_k: f32
) -> f32 {
    let max_steps = runtime_shadow_max_steps();
    let max_distance = max_t * runtime_shadow_distance_scale();
    let min_step = (0.006 + (0.01 * roughness)) * runtime_shadow_min_step_scale();
    let max_step = mix(0.38, 0.68, roughness);
    let hit_threshold = max(HIT_EPSILON * 0.65, 0.00012);
    var attenuation = 1.0;
    var t = min_t;
    var i: u32 = 0u;
    loop {
        if (i >= max_steps) {
            break;
        }
        if (t >= max_distance) {
            break;
        }

        let p = origin + (direction * t);
        let h = scene_distance(p);
        if (h < hit_threshold) {
            return 0.0;
        }

        attenuation = min(attenuation, clamp(penumbra_k * h / max(t, min_t), 0.0, 1.0));
        if (attenuation <= 0.002) {
            return 0.0;
        }
        let travel = clamp(t / max(max_distance, 0.001), 0.0, 1.0);
        let far_boost = mix(1.04, 1.35, travel);
        let step_size = clamp(h * far_boost, min_step, max_step);
        t = t + step_size;
        if (t > (max_distance * 0.62) && attenuation > 0.996 && h > (max_step * 1.7)) {
            break;
        }
        i = i + 1u;
    }
    return clamp(attenuation, 0.0, 1.0);
}

fn ambient_occlusion(origin: vec3<f32>, normal: vec3<f32>, roughness: f32) -> f32 {
    let max_samples = runtime_ao_samples();
    let radius = mix(0.16, 0.3, 1.0 - roughness) * runtime_ao_radius_scale();
    let horizon_bias = runtime_ao_horizon_bias();
    let tangent = tangent_from_axis(normal);
    let bitangent = cross(normal, tangent);

    var occlusion = 0.0;
    var weight_sum = 0.0;
    var distance = 0.016;
    var i: u32 = 0u;
    loop {
        if (i >= max_samples) {
            break;
        }
        let sample_ratio = (f32(i) + 0.5) / f32(max_samples);
        let radial = sqrt(clamp(sample_ratio, 0.0, 1.0));
        let phi = 2.39996323 * f32(i);
        let tangent_dir = (tangent * cos(phi)) + (bitangent * sin(phi));
        let hemi_dir = normalize(normal + (tangent_dir * radial * 0.85));
        let sample_point = origin + (hemi_dir * distance);
        let sdf = scene_distance(sample_point);
        if (sdf <= 0.0) {
            return 0.0;
        }
        let expected_clearance = distance * horizon_bias;
        let blocker = max(expected_clearance - sdf, 0.0) / max(distance, 0.001);
        let weight = exp(-1.2 * f32(i));
        occlusion = occlusion + (blocker * weight);
        weight_sum = weight_sum + weight;
        if (occlusion >= (AO_EARLY_EXIT * 1.25)) {
            break;
        }
        if (i >= 2u && sdf > (distance * 2.6)) {
            break;
        }
        let ring_step = max(radius / max(f32(max_samples), 1.0), 0.018);
        distance = distance + (ring_step * (1.0 + (0.24 * f32(i))));
        i = i + 1u;
    }
    let normalized_occlusion = occlusion / max(weight_sum, 0.0001);
    return clamp(1.0 - (normalized_occlusion * 1.12), 0.0, 1.0);
}

fn background_color(direction: vec3<f32>) -> vec3<f32> {
    let unit = normalize(direction);
    let t = 0.5 * (unit.y + 1.0);
    let top = vec3<f32>(0.5, 0.71, 0.94);
    let bottom = vec3<f32>(0.98, 0.99, 1.0);
    let base = (bottom * (1.0 - t)) + (top * t);

    var sun = vec3<f32>(0.0);
    var i: u32 = 0u;
    let count = light_count_capped();
    loop {
        if (i >= count) {
            break;
        }
        let alignment = max(dot(unit, light_dir(i)), 0.0);
        sun = sun + (light_radiance(i) * pow(alignment, 420.0) * 6.0);
        i = i + 1u;
    }
    return clamp(base + sun, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn fresnel_schlick_dielectric(cos_theta: f32, eta_i: f32, eta_t: f32) -> f32 {
    let eta_sum = max(eta_i + eta_t, 0.0001);
    let r0 = pow((eta_i - eta_t) / eta_sum, 2.0);
    let m = 1.0 - clamp(cos_theta, 0.0, 1.0);
    let m2 = m * m;
    let m5 = m2 * m2 * m;
    return clamp(r0 + ((1.0 - r0) * m5), 0.0, 1.0);
}

fn beer_lambert(absorption: vec3<f32>, distance: f32) -> vec3<f32> {
    return exp(-absorption * max(distance, 0.0));
}

fn normalized_specular_lobe(cos_theta: f32, exponent: f32) -> f32 {
    let n_dot_h = clamp(cos_theta, 0.0, 1.0);
    let normalization = (exponent + 8.0) * 0.039788735772973836;
    return pow(n_dot_h, exponent) * normalization;
}

fn shade_lit_surface(hit: Hit, ray_dir: vec3<f32>, material_index: u32) -> vec3<f32> {
    let albedo_roughness = material_albedo_roughness(material_index);
    let emission_metallic = material_emission_metallic(material_index);
    let albedo = clamp(albedo_roughness.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
    let roughness = clamp(albedo_roughness.w, 0.0, 1.0);
    let metallic = clamp(emission_metallic.w, 0.0, 1.0);
    let emission = max(emission_metallic.xyz, vec3<f32>(0.0));

    let view = normalize(-ray_dir);
    let shininess = 12.0 + ((1.0 - roughness) * 220.0);
    let f0 = (vec3<f32>(0.04) * (1.0 - metallic)) + (albedo * metallic);

    var direct_diffuse = vec3<f32>(0.0);
    var direct_specular = vec3<f32>(0.0);
    var i: u32 = 0u;
    let light_count = light_count_capped();
    loop {
        if (i >= light_count) {
            break;
        }

        let ldir = light_dir(i);
        let radiance = light_radiance(i);
        let lambert = max(dot(hit.normal, ldir), 0.0);
        let shadow = soft_shadow(
            hit.point + (hit.normal * (RAY_BIAS * 1.5)),
            ldir,
            0.02,
            24.0,
            roughness,
            20.0 + ((1.0 - roughness) * 18.0)
        );
        let diffuse_factor = lambert * shadow * (1.0 - (0.75 * metallic));
        direct_diffuse = direct_diffuse + (albedo * diffuse_factor * radiance);

        let half_vec = normalize(ldir + view);
        let spec_shape = normalized_specular_lobe(dot(hit.normal, half_vec), shininess) * shadow;
        direct_specular = direct_specular + (radiance * f0 * spec_shape);

        i = i + 1u;
    }

    let ao_strength = select(1.0, 0.2, hit.kind == OBJECT_KIND_PLANE);
    let ao = mix(
        1.0,
        ambient_occlusion(hit.point + (hit.normal * (RAY_BIAS * 2.0)), hit.normal, roughness),
        ao_strength
    );
    let hemi = 0.5 * (hit.normal.y + 1.0);
    let ambient = 0.03 + ((0.16 * hemi) * (1.0 - (0.35 * metallic)));
    let sky_tint = vec3<f32>(0.24, 0.32, 0.44) * (0.09 * hemi) * (1.0 - (0.4 * roughness));
    let bounce_tint = albedo * (0.05 * (1.0 - hemi)) * (1.0 - metallic);
    let lit = (albedo * ambient) + direct_diffuse;
    let distance_fade = clamp(1.0 - (hit.t * (0.012 + (0.004 * roughness))), 0.7, 1.0);

    return max(
        (((lit + sky_tint + bounce_tint) * ao) + direct_specular + emission) * distance_fade,
        vec3<f32>(0.0)
    );
}

fn material_reflectivity(material_index: u32) -> f32 {
    let albedo_roughness = material_albedo_roughness(material_index);
    let emission_metallic = material_emission_metallic(material_index);
    let roughness = clamp(albedo_roughness.w, 0.0, 1.0);
    let metallic = clamp(emission_metallic.w, 0.0, 1.0);
    let dielectric = 0.02 + ((1.0 - roughness) * 0.18);
    let conductor = 0.18 + ((1.0 - roughness) * 0.62);
    return clamp((dielectric * (1.0 - metallic)) + (conductor * metallic), 0.0, 0.95);
}

fn shade_transmissive_lighting(hit: Hit, ray_dir: vec3<f32>, material_index: u32) -> vec3<f32> {
    let albedo_roughness = material_albedo_roughness(material_index);
    let emission_metallic = material_emission_metallic(material_index);
    let roughness = clamp(albedo_roughness.w, 0.0, 1.0);
    let metallic = clamp(emission_metallic.w, 0.0, 1.0);
    let albedo = clamp(albedo_roughness.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
    let emission = max(emission_metallic.xyz, vec3<f32>(0.0));
    let f0 = (vec3<f32>(0.04) * (1.0 - metallic)) + (albedo * metallic);

    var highlight = vec3<f32>(0.0);
    var i: u32 = 0u;
    let light_count = light_count_capped();
    loop {
        if (i >= light_count) {
            break;
        }

        let ldir = light_dir(i);
        let reflected_light = reflect(-ldir, hit.normal);
        let spec_power = 20.0 + ((1.0 - roughness) * 320.0);
        let spec = normalized_specular_lobe(dot(reflected_light, normalize(-ray_dir)), spec_power);
        let shadow = soft_shadow(
            hit.point + (hit.normal * (RAY_BIAS * 1.5)),
            ldir,
            0.02,
            24.0,
            roughness,
            36.0
        );
        let glass_spec_gain = 0.62 + ((1.0 - roughness) * 0.46);
        highlight = highlight + (light_radiance(i) * f0 * (spec * shadow * glass_spec_gain));

        i = i + 1u;
    }

    return highlight + emission;
}

fn firefly_luminance_cap(roughness: f32, transmission: f32, bounce: u32) -> f32 {
    let clamp_scale = runtime_firefly_clamp_scale();
    let smooth_factor = 1.0 - clamp(roughness, 0.0, 1.0);
    let base = 1.6 + (smooth_factor * 2.8) + (transmission * 2.4);
    let bounce_decay = mix(1.0, 0.68, clamp(f32(bounce) / 8.0, 0.0, 1.0));
    return max(base * clamp_scale * bounce_decay, 0.25);
}

fn clamp_firefly(color: vec3<f32>, luminance_cap: f32) -> vec3<f32> {
    let safe_color = max(color, vec3<f32>(0.0));
    let luminance = color_luminance(safe_color);
    if (luminance <= luminance_cap) {
        return safe_color;
    }
    return safe_color * (luminance_cap / max(luminance, 0.0001));
}

fn trace_scene_skipping_object(
    origin_in: vec3<f32>,
    direction_in: vec3<f32>,
    skip_object_index: u32,
    max_skips: u32
) -> vec3<f32> {
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

        if (hit.object_index == skip_object_index) {
            origin = hit.point + (direction * (RAY_BIAS * 2.0));
            skips = skips + 1u;
            continue;
        }

        let material_index = safe_material_index(hit.material_index);
        if (material_transmission(material_index) > 0.0) {
            return shade_transmissive_lighting(hit, direction, material_index);
        }
        return shade_lit_surface(hit, direction, material_index);
    }

    return background_color(direction);
}

fn trace_glass_transmission(
    surface_hit: Hit,
    incoming_dir: vec3<f32>,
    ior: f32,
    absorption: vec3<f32>,
    transmissive_object_index: u32,
    max_skips: u32,
    roughness: f32,
    path_seed: u32,
    bounce_seed: u32
) -> vec3<f32> {
    let front_face = dot(incoming_dir, surface_hit.normal) < 0.0;
    let interface_normal = select(-surface_hit.normal, surface_hit.normal, front_face);
    let eta_i = select(ior, 1.0, front_face);
    let eta_t = select(1.0, ior, front_face);
    let eta = eta_i / max(eta_t, 0.0001);
    let cos_theta = clamp(dot(-incoming_dir, interface_normal), 0.0, 1.0);
    let sin2_theta = max(0.0, 1.0 - (cos_theta * cos_theta));
    let total_internal_reflection = (eta * eta * sin2_theta) > 1.0;
    if (total_internal_reflection) {
        return vec3<f32>(0.0);
    }

    let rough_transmission = clamp(roughness * 0.75, 0.0, 0.95);
    let refracted = normalize(refract(incoming_dir, interface_normal, eta));
    var refracted_rough = sample_rough_direction(
        refracted,
        rough_transmission,
        path_seed,
        bounce_seed,
        31u
    );
    if (front_face && dot(refracted_rough, surface_hit.normal) >= 0.0) {
        refracted_rough = refracted;
    }
    if (!front_face && dot(refracted_rough, surface_hit.normal) <= 0.0) {
        refracted_rough = refracted;
    }

    if (!front_face) {
        let outside_origin = surface_hit.point + (surface_hit.normal * (RAY_BIAS * 2.0));
        let exit_fresnel = fresnel_schlick_dielectric(cos_theta, eta_i, eta_t);
        let outside_color = trace_scene_skipping_object(
            outside_origin,
            refracted_rough,
            transmissive_object_index,
            max_skips
        ) * (1.0 - exit_fresnel);
        return clamp_firefly(
            outside_color,
            firefly_luminance_cap(roughness, 1.0, bounce_seed)
        );
    }

    var inside_origin = surface_hit.point - (surface_hit.normal * (RAY_BIAS * 2.0));
    var inside_dir = refracted_rough;
    var travel_distance = 0.0;
    var internal_bounce: u32 = 0u;
    loop {
        if (internal_bounce >= 6u) {
            break;
        }

        let inside_hit = ray_march(inside_origin, inside_dir);
        if (inside_hit.t < 0.0) {
            let attenuation = beer_lambert(absorption, travel_distance);
            return clamp_firefly(
                background_color(inside_dir) * attenuation,
                firefly_luminance_cap(roughness, 1.0, bounce_seed)
            );
        }

        if (inside_hit.object_index != transmissive_object_index) {
            let attenuation = beer_lambert(absorption, travel_distance + max(inside_hit.t, 0.0));
            let fallback_origin = inside_hit.point + (inside_dir * (RAY_BIAS * 2.0));
            let outside_color = trace_scene_skipping_object(
                fallback_origin,
                inside_dir,
                transmissive_object_index,
                max_skips
            ) * attenuation;
            return clamp_firefly(
                outside_color,
                firefly_luminance_cap(roughness, 1.0, bounce_seed)
            );
        }

        travel_distance = travel_distance + max(inside_hit.t, 0.0);
        let outward_normal = inside_hit.normal;
        let exit_normal = -outward_normal;
        let exit_eta = ior / 1.0;
        let cos_exit = clamp(dot(-inside_dir, exit_normal), 0.0, 1.0);
        let sin2_exit = max(0.0, 1.0 - (cos_exit * cos_exit));

        if ((exit_eta * exit_eta * sin2_exit) > 1.0) {
            let ideal_internal_reflect = normalize(reflect(inside_dir, exit_normal));
            var internal_reflect = sample_rough_direction(
                ideal_internal_reflect,
                rough_transmission,
                path_seed,
                bounce_seed + internal_bounce + 1u,
                41u
            );
            if (dot(internal_reflect, outward_normal) >= 0.0) {
                internal_reflect = ideal_internal_reflect;
            }
            inside_dir = internal_reflect;
            inside_origin = inside_hit.point - (outward_normal * (RAY_BIAS * 2.0));
            internal_bounce = internal_bounce + 1u;
            continue;
        }

        let exit_fresnel = fresnel_schlick_dielectric(cos_exit, ior, 1.0);
        let attenuation = beer_lambert(absorption, travel_distance);
        let refracted_out = normalize(refract(inside_dir, exit_normal, exit_eta));
        var refracted_out_rough = sample_rough_direction(
            refracted_out,
            rough_transmission,
            path_seed,
            bounce_seed + internal_bounce + 1u,
            51u
        );
        if (dot(refracted_out_rough, outward_normal) <= 0.0) {
            refracted_out_rough = refracted_out;
        }
        let outside_origin = inside_hit.point + (outward_normal * (RAY_BIAS * 2.0));
        let outside_color = trace_scene_skipping_object(
            outside_origin,
            refracted_out_rough,
            transmissive_object_index,
            max_skips
        );
        return clamp_firefly(
            outside_color * attenuation * (1.0 - exit_fresnel),
            firefly_luminance_cap(roughness, 1.0, bounce_seed)
        );
    }

    return vec3<f32>(0.0);
}

fn trace_ray(origin_in: vec3<f32>, direction_in: vec3<f32>, path_seed: u32) -> vec3<f32> {
    let rr_start_bounce = runtime_rr_start_bounce();
    var origin = origin_in;
    var direction = normalize(direction_in);
    var throughput = vec3<f32>(1.0);
    var accumulated = vec3<f32>(0.0);
    let max_bounces = max(params.max_depth, 1u);

    var bounce: u32 = 0u;
    loop {
        if (bounce >= max_bounces) {
            let env_contribution = clamp_firefly(
                background_color(direction),
                firefly_luminance_cap(0.0, 0.0, bounce)
            );
            accumulated = accumulated + (throughput * env_contribution);
            break;
        }

        let hit = ray_march(origin, direction);
        if (hit.t < 0.0) {
            let env_contribution = clamp_firefly(
                background_color(direction),
                firefly_luminance_cap(0.0, 0.0, bounce)
            );
            accumulated = accumulated + (throughput * env_contribution);
            break;
        }

        let material_index = safe_material_index(hit.material_index);
        let roughness = clamp(material_albedo_roughness(material_index).w, 0.0, 1.0);
        let transmission = material_transmission(material_index);

        if (transmission <= 0.0) {
            let shaded = shade_lit_surface(hit, direction, material_index);
            let reflectivity = material_reflectivity(material_index);
            let diffuse_contribution = clamp_firefly(
                shaded * (1.0 - reflectivity),
                firefly_luminance_cap(roughness, 0.0, bounce)
            );
            accumulated = accumulated + (throughput * diffuse_contribution);
            throughput = throughput * reflectivity;
            if (max(max(throughput.x, throughput.y), throughput.z) < 0.001) {
                break;
            }
            let rr_start = rr_start_for_material(rr_start_bounce, roughness, transmission);
            if (bounce >= rr_start) {
                let survive_prob = russian_roulette_probability(
                    throughput,
                    roughness,
                    transmission,
                    bounce,
                    rr_start
                );
                let rr = path_random(path_seed, bounce, 61u);
                if (rr > survive_prob) {
                    break;
                }
                throughput = throughput / survive_prob;
            }

            let ideal_reflect = normalize(reflect(direction, hit.normal));
            var reflected_dir = sample_rough_direction(
                ideal_reflect,
                roughness,
                path_seed,
                bounce,
                11u
            );
            if (dot(reflected_dir, hit.normal) <= 0.0) {
                reflected_dir = ideal_reflect;
            }
            origin = hit.point + (hit.normal * RAY_BIAS);
            direction = reflected_dir;
            bounce = bounce + 1u;
            continue;
        }

        let ior = material_ior(material_index);
        let absorption = material_absorption(material_index);
        let front_face = dot(direction, hit.normal) < 0.0;
        let interface_normal = select(-hit.normal, hit.normal, front_face);
        let eta_i = select(ior, 1.0, front_face);
        let eta_t = select(1.0, ior, front_face);
        let cos_theta = clamp(dot(-direction, interface_normal), 0.0, 1.0);
        let fresnel = fresnel_schlick_dielectric(cos_theta, eta_i, eta_t);

        let transmission_color = trace_glass_transmission(
            hit,
            direction,
            ior,
            absorption,
            hit.object_index,
            max_bounces - bounce,
            roughness,
            path_seed,
            bounce
        );
        let transmission_weight_raw = transmission * (1.0 - fresnel);
        let reflection_weight_raw = (1.0 - transmission) + (transmission * fresnel);
        let weight_sum = max(transmission_weight_raw + reflection_weight_raw, 0.0001);
        let transmission_weight = transmission_weight_raw / weight_sum;
        let reflection_weight = reflection_weight_raw / weight_sum;
        let glass_luminance_cap = firefly_luminance_cap(roughness, transmission, bounce);
        let transmission_contribution = clamp_firefly(
            transmission_color * transmission_weight,
            glass_luminance_cap
        );
        accumulated = accumulated + (throughput * transmission_contribution);

        let reflective_lighting = shade_transmissive_lighting(hit, direction, material_index);
        let reflection_contribution = clamp_firefly(
            reflective_lighting * reflection_weight,
            glass_luminance_cap
        );
        accumulated = accumulated + (throughput * reflection_contribution);
        throughput = throughput * reflection_weight;

        if (max(max(throughput.x, throughput.y), throughput.z) < 0.001) {
            break;
        }
        let rr_start = rr_start_for_material(rr_start_bounce, roughness, transmission);
        if (bounce >= rr_start) {
            let survive_prob = russian_roulette_probability(
                throughput,
                roughness,
                transmission,
                bounce,
                rr_start
            );
            let rr = path_random(path_seed, bounce, 71u);
            if (rr > survive_prob) {
                break;
            }
            throughput = throughput / survive_prob;
        }

        let ideal_reflect = normalize(reflect(direction, hit.normal));
        var reflected_dir = sample_rough_direction(
            ideal_reflect,
            min(roughness * 0.6, 0.8),
            path_seed,
            bounce,
            21u
        );
        if (dot(reflected_dir, hit.normal) <= 0.0) {
            reflected_dir = ideal_reflect;
        }
        origin = hit.point + (hit.normal * RAY_BIAS);
        direction = reflected_dir;
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

fn path_random(path_seed: u32, bounce: u32, dimension: u32) -> f32 {
    let mixed =
        path_seed + (bounce * 0x9e3779b9u) + (dimension * 0x85ebca6bu) + 0xa511e9b3u;
    return random01(mixed);
}

fn tangent_from_axis(axis: vec3<f32>) -> vec3<f32> {
    let helper = select(
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(1.0, 0.0, 0.0),
        abs(axis.y) > 0.9
    );
    return normalize(cross(helper, axis));
}

fn sample_direction_around_axis(axis: vec3<f32>, xi1: f32, xi2: f32) -> vec3<f32> {
    let n = normalize(axis);
    let t = tangent_from_axis(n);
    let b = cross(n, t);
    let r = sqrt(clamp(xi1, 0.0, 0.999999));
    let phi = 6.28318530718 * xi2;
    let h = vec3<f32>(r * cos(phi), r * sin(phi), sqrt(max(1.0 - (r * r), 0.0)));
    return normalize((t * h.x) + (b * h.y) + (n * h.z));
}

fn sample_rough_direction(
    axis: vec3<f32>,
    roughness: f32,
    path_seed: u32,
    bounce: u32,
    dimension: u32
) -> vec3<f32> {
    let base = normalize(axis);
    let spread = clamp(roughness * roughness, 0.0, 1.0);
    if (spread <= 0.0001) {
        return base;
    }
    let xi1 = path_random(path_seed, bounce, dimension);
    let xi2 = path_random(path_seed, bounce, dimension + 1u);
    let lobe = sample_direction_around_axis(base, xi1, xi2);
    return normalize(mix(base, lobe, spread));
}

fn rr_start_for_material(base_start: u32, roughness: f32, transmission: f32) -> u32 {
    let glossy_bonus = select(0u, 1u, roughness < 0.24);
    let transmissive_bonus = select(0u, 1u, transmission > 0.02);
    let diffuse_penalty = select(0u, 1u, (roughness > 0.82) && (transmission <= 0.02));
    let adjusted = base_start + glossy_bonus + transmissive_bonus;
    return clamp(adjusted - diffuse_penalty, 0u, 32u);
}

fn russian_roulette_probability(
    throughput: vec3<f32>,
    roughness: f32,
    transmission: f32,
    bounce: u32,
    rr_start: u32
) -> f32 {
    let luminance = dot(throughput, vec3<f32>(0.2126, 0.7152, 0.0722));
    let peak = max(max(throughput.x, throughput.y), throughput.z);
    let energy_blend = mix(luminance, peak, 0.28);
    let specular_bias = (1.0 - roughness) * 0.16;
    let transmissive_bias = transmission * 0.24;
    let energy = energy_blend * (1.0 + specular_bias + transmissive_bias);

    var rr_age: u32 = 0u;
    if (bounce > rr_start) {
        rr_age = bounce - rr_start;
    }
    let warmup = clamp(f32(rr_age) / 3.0, 0.0, 1.0);
    let min_survive = mix(0.42, RR_MIN_SURVIVE, warmup);
    let max_survive = mix(
        RR_MAX_SURVIVE,
        0.94,
        clamp((transmission * 0.8) + ((1.0 - roughness) * 0.2), 0.0, 1.0)
    );

    return clamp(energy, min_survive, max_survive);
}

fn radical_inverse_vdc(bits_in: u32) -> f32 {
    var bits = bits_in;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x00ff00ffu) << 8u) | ((bits & 0xff00ff00u) >> 8u);
    bits = ((bits & 0x0f0f0f0fu) << 4u) | ((bits & 0xf0f0f0f0u) >> 4u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xccccccccu) >> 2u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xaaaaaaaau) >> 1u);
    return f32(bits) * INV_U32_SCALE;
}

fn halton_scrambled(index: u32, base: u32, scramble_seed: u32) -> f32 {
    var i = index;
    var f = 1.0;
    var result = 0.0;
    var digit_level: u32 = 0u;
    loop {
        if (i == 0u) {
            break;
        }
        f = f / f32(base);
        let digit = i % base;
        let scramble = hash_u32(scramble_seed ^ (digit_level * 0x9e3779b9u));
        let permuted_digit = (digit + (scramble % base)) % base;
        result = result + (f * f32(permuted_digit));
        i = i / base;
        digit_level = digit_level + 1u;
    }
    return result;
}

fn sobol_component(index: u32, axis: u32, scramble_seed: u32) -> f32 {
    let dimension_index = select(index, index ^ (index >> 1u), axis == 1u);
    let scrambled = dimension_index ^ hash_u32(scramble_seed ^ (axis * 0x85ebca6bu));
    return radical_inverse_vdc(scrambled);
}

fn cranley_patterson_rotation(x: u32, y: u32, axis: u32) -> f32 {
    let seed = (x * 0x9e3779b9u) ^ (y * 0xc2b2ae35u) ^ (axis * 0x27d4eb2fu) ^ 0xa511e9b3u;
    return random01(seed);
}

fn gcd_u32(a_in: u32, b_in: u32) -> u32 {
    var a = a_in;
    var b = b_in;
    loop {
        if (b == 0u) {
            break;
        }
        let next = a % b;
        a = b;
        b = next;
    }
    return a;
}

fn permutation_stride(sample_count: u32, seed: u32) -> u32 {
    if (sample_count <= 1u) {
        return 1u;
    }

    var stride = (seed % max(sample_count - 1u, 1u)) + 1u;
    if ((stride & 1u) == 0u) {
        stride = stride + 1u;
        if (stride >= sample_count) {
            stride = stride - sample_count;
            if (stride == 0u) {
                stride = 1u;
            }
        }
    }

    var tries: u32 = 0u;
    loop {
        if (tries >= 24u || gcd_u32(stride, sample_count) == 1u) {
            break;
        }
        stride = stride + 2u;
        if (stride >= sample_count) {
            stride = (stride % sample_count) + 1u;
        }
        tries = tries + 1u;
    }

    if (gcd_u32(stride, sample_count) != 1u) {
        return 1u;
    }
    return stride;
}

fn permuted_sample_index(sample_index: u32, sample_count: u32, pixel_seed: u32) -> u32 {
    if (sample_count <= 1u) {
        return 0u;
    }
    let stride = permutation_stride(sample_count, pixel_seed ^ 0x7feb352du);
    let offset = hash_u32(pixel_seed ^ 0x846ca68bu) % sample_count;
    return (sample_index * stride + offset) % sample_count;
}

fn color_luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn adaptive_min_samples(sample_count: u32) -> u32 {
    if (sample_count <= 1u) {
        return 1u;
    }
    let scaled = as_u32_rounded(f32(sample_count) * runtime_adaptive_min_samples_fraction());
    let min_samples = max(scaled, 2u);
    return clamp(min_samples, 2u, sample_count);
}

fn sample_jitter(x: u32, y: u32, sequence_index: u32, axis: u32, pixel_seed: u32) -> f32 {
    let mode = runtime_sampling_mode();
    if (mode == 0u) {
        let seed =
            (x * 1973u) +
            (y * 9277u) +
            (sequence_index * 26699u) +
            (axis * 104729u) ^
            0x68bc21ebu;
        return random01(seed);
    }

    let rotation = cranley_patterson_rotation(x, y, axis);
    let index = sequence_index + 1u;
    if (mode == 1u) {
        let base = select(2u, 3u, axis == 1u);
        let scramble_seed = pixel_seed ^ (axis * 0x68bc21ebu);
        let sequence = halton_scrambled(index, base, scramble_seed);
        return fract(sequence + rotation);
    }

    let sobol = sobol_component(index, axis, pixel_seed ^ 0x02e5be93u);
    return fract(sobol + rotation);
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

    let theta = radians(max(params.scene_scalars.x, 1.0));
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
    let pixel_seed = hash_u32((gid.x * 73856093u) ^ (gid.y * 19349663u) ^ 0x9e3779b9u);
    let strata_x_count = max(u32(sqrt(f32(sample_count))), 1u);
    let strata_y_count = (sample_count + strata_x_count - 1u) / strata_x_count;
    let adaptive_threshold = runtime_adaptive_variance_threshold();
    let adaptive_enabled = (sample_count >= 3u) && (adaptive_threshold > 0.0);
    let adaptive_min_spp = adaptive_min_samples(sample_count);
    let adaptive_check_interval = runtime_adaptive_check_interval();

    var color = vec3<f32>(0.0);
    var luminance_mean = 0.0;
    var luminance_m2 = 0.0;
    var used_samples: u32 = 0u;
    var sample_index: u32 = 0u;
    loop {
        if (sample_index >= sample_count) {
            break;
        }

        let sequence_index = permuted_sample_index(sample_index, sample_count, pixel_seed);
        let strata_x = sequence_index % strata_x_count;
        let strata_y = sequence_index / strata_x_count;
        let jitter_x = (f32(strata_x) + sample_jitter(gid.x, gid.y, sequence_index, 0u, pixel_seed)) / f32(strata_x_count);
        let jitter_y = (f32(strata_y) + sample_jitter(gid.x, gid.y, sequence_index, 1u, pixel_seed)) / f32(strata_y_count);

        let u = (f32(gid.x) + jitter_x) / width_f;
        let v = (f32((params.height - 1u) - gid.y) + jitter_y) / height_f;
        let direction = normalize(lower_left + (horizontal * u) + (vertical * v) - origin);
        let path_seed = hash_u32(
            (gid.x * 73856093u) ^
            (gid.y * 19349663u) ^
            (sequence_index * 83492791u) ^
            (sample_index * 1597334677u) ^
            0x9e3779b9u
        );
        let sample_color = trace_ray(origin, direction, path_seed);
        color = color + sample_color;

        let current_sample = sample_index + 1u;
        used_samples = current_sample;
        let lum = color_luminance(sample_color);
        let sample_count_f = f32(current_sample);
        let delta = lum - luminance_mean;
        luminance_mean = luminance_mean + (delta / sample_count_f);
        let delta2 = lum - luminance_mean;
        luminance_m2 = luminance_m2 + (delta * delta2);

        if (adaptive_enabled && current_sample >= adaptive_min_spp && current_sample < sample_count) {
            let after_min = current_sample - adaptive_min_spp;
            if ((after_min % adaptive_check_interval) == 0u) {
                let variance = luminance_m2 / max(sample_count_f - 1.0, 1.0);
                let std_dev = sqrt(max(variance, 0.0));
                let relative_noise = std_dev / max(abs(luminance_mean), 0.03);
                if (relative_noise <= adaptive_threshold) {
                    break;
                }
            }
        }

        sample_index = sample_index + 1u;
    }
    color = color / f32(max(used_samples, 1u));
    color = filmic_tone_map(color);
    color = pow(color, vec3<f32>(1.0 / 2.2));
    textureStore(
        output_tex,
        vec2<i32>(i32(gid.x), i32(gid.y)),
        vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0)
    );
}
