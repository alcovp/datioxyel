use image::{Rgb, RgbImage};
use rayon::prelude::*;
use serde::Deserialize;
use std::borrow::Cow;
use std::io::{self, Read};
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::path::Path;
use std::time::Instant;

const MAX_MARCH_STEPS: u32 = 280;
const MAX_TRACE_DISTANCE: f32 = 42.0;
const HIT_EPSILON: f32 = 0.00035;
const NORMAL_EPSILON: f32 = 0.001;
const RAY_BIAS: f32 = 0.003;
const TEST_FORCE_OPAQUE_RED_MENGER: bool = true;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RenderFrameConfig {
    width: u32,
    height: u32,
    output_path: String,
    max_depth: u8,
    #[serde(default = "default_samples_per_pixel")]
    samples_per_pixel: u16,
    scene: String,
    renderer_mode: String,
    camera_origin: [f32; 3],
    camera_target: [f32; 3],
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RenderBatchConfig {
    frames: Vec<RenderFrameConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum IncomingConfig {
    Single(RenderFrameConfig),
    Batch(RenderBatchConfig),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RenderMode {
    Cpu,
    Gpu,
}

impl RenderMode {
    fn parse(value: &str) -> Self {
        if value.eq_ignore_ascii_case("gpu") {
            Self::Gpu
        } else {
            Self::Cpu
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Gpu => "GPU",
        }
    }
}

const fn default_samples_per_pixel() -> u16 {
    1
}

#[derive(Clone, Copy, Debug)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3 {
    const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    const fn splat(v: f32) -> Self {
        Self::new(v, v, v)
    }

    fn dot(self, rhs: Self) -> f32 {
        (self.x * rhs.x) + (self.y * rhs.y) + (self.z * rhs.z)
    }

    fn cross(self, rhs: Self) -> Self {
        Self::new(
            (self.y * rhs.z) - (self.z * rhs.y),
            (self.z * rhs.x) - (self.x * rhs.z),
            (self.x * rhs.y) - (self.y * rhs.x),
        )
    }

    fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    fn normalize(self) -> Self {
        let len = self.length();
        if len == 0.0 {
            return self;
        }
        self / len
    }

    fn abs(self) -> Self {
        Self::new(self.x.abs(), self.y.abs(), self.z.abs())
    }

    fn max(self, rhs: Self) -> Self {
        Self::new(self.x.max(rhs.x), self.y.max(rhs.y), self.z.max(rhs.z))
    }

    fn rem_euclid(self, rhs: f32) -> Self {
        Self::new(
            self.x.rem_euclid(rhs),
            self.y.rem_euclid(rhs),
            self.z.rem_euclid(rhs),
        )
    }

    fn max_component(self) -> f32 {
        self.x.max(self.y).max(self.z)
    }

    fn clamp01(self) -> Self {
        Self::new(
            self.x.clamp(0.0, 1.0),
            self.y.clamp(0.0, 1.0),
            self.z.clamp(0.0, 1.0),
        )
    }
}

impl Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Mul<f32> for Vec3 {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl Mul<Vec3> for Vec3 {
    type Output = Self;
    fn mul(self, rhs: Vec3) -> Self::Output {
        Self::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    }
}

impl Div<f32> for Vec3 {
    type Output = Self;
    fn div(self, rhs: f32) -> Self::Output {
        Self::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

impl Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y, -self.z)
    }
}

#[derive(Clone, Copy)]
struct Ray {
    origin: Vec3,
    direction: Vec3,
}

impl Ray {
    fn at(self, t: f32) -> Vec3 {
        self.origin + (self.direction * t)
    }
}

#[derive(Clone, Copy)]
enum MaterialId {
    Floor,
    Glass,
    Mirror,
}

#[derive(Clone, Copy)]
struct SdfSample {
    distance: f32,
    material: MaterialId,
}

#[derive(Clone, Copy)]
struct HitRecord {
    t: f32,
    point: Vec3,
    normal: Vec3,
    material: MaterialId,
}

#[derive(Clone, Copy)]
struct Scene {
    floor_y: f32,
    sponge_center: Vec3,
    sponge_scale: f32,
    sponge_iterations: u32,
    mirror_sphere_center: Vec3,
    mirror_sphere_radius: f32,
    sun_direction: Vec3,
}

impl Scene {
    fn stage4_scene() -> Self {
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

    fn sample(self, p: Vec3) -> SdfSample {
        let floor_distance = p.y - self.floor_y;
        let local = (p - self.sponge_center) / self.sponge_scale;
        let sponge_distance = sd_menger(local, self.sponge_iterations) * self.sponge_scale;
        let mirror_sphere_distance = sd_sphere(
            p - self.mirror_sphere_center,
            self.mirror_sphere_radius,
        );

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

    fn distance(self, p: Vec3) -> f32 {
        self.sample(p).distance
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParams {
    width: u32,
    height: u32,
    max_depth: u32,
    sponge_iterations: u32,
    camera_origin: [f32; 4],
    camera_target: [f32; 4],
    sun_direction: [f32; 4],
    mirror_sphere_center: [f32; 4],
    floor_y: f32,
    sponge_scale: f32,
    mirror_sphere_radius: f32,
    samples_per_pixel: u32,
}

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut raw = String::new();
    io::stdin().read_to_string(&mut raw)?;

    let incoming: IncomingConfig = serde_json::from_str(&raw)?;
    let frames = match incoming {
        IncomingConfig::Single(frame) => vec![frame],
        IncomingConfig::Batch(batch) => batch.frames,
    };
    if frames.is_empty() {
        return Err("frames array must not be empty".into());
    }

    let scene = Scene::stage4_scene();
    let total = frames.len();
    let mut gpu_renderer: Option<GpuRenderer> = None;
    let mut gpu_init_error: Option<String> = None;
    for (index, frame) in frames.iter().enumerate() {
        validate_config(frame)?;
        let mode = RenderMode::parse(&frame.renderer_mode);
        let started = Instant::now();
        let image = match mode {
            RenderMode::Cpu => render_cpu(frame, scene),
            RenderMode::Gpu => {
                if gpu_renderer.is_none() && gpu_init_error.is_none() {
                    match pollster::block_on(GpuRenderer::new()) {
                        Ok(renderer) => {
                            gpu_renderer = Some(renderer);
                        }
                        Err(error) => {
                            gpu_init_error = Some(error);
                        }
                    }
                }

                if let Some(renderer) = gpu_renderer.as_mut() {
                    match renderer.render_frame(frame, scene) {
                        Ok(image) => image,
                        Err(error) => {
                            eprintln!("GPU render failed: {error}. Falling back to CPU.");
                            render_cpu(frame, scene)
                        }
                    }
                } else {
                    if let Some(error) = &gpu_init_error {
                        eprintln!("GPU initialization failed: {error}. Falling back to CPU.");
                    }
                    render_cpu(frame, scene)
                }
            }
        };
        let elapsed_ms = started.elapsed().as_millis();
        image.save(&frame.output_path)?;

        println!(
            "[{}/{}] Rendered stage 4 frame [{}] in {} ms: {}",
            index + 1,
            total,
            mode.as_str(),
            elapsed_ms,
            frame.output_path
        );
    }

    Ok(())
}

fn validate_config(config: &RenderFrameConfig) -> Result<(), Box<dyn std::error::Error>> {
    if config.width == 0 || config.height == 0 {
        return Err("width and height must be positive".into());
    }

    let output_parent = Path::new(&config.output_path)
        .parent()
        .ok_or("outputPath must include a parent directory")?;

    if !output_parent.exists() {
        return Err(format!(
            "output directory does not exist: {}",
            output_parent.display()
        )
        .into());
    }

    if config.max_depth == 0 {
        return Err("maxDepth must be at least 1".into());
    }

    if config.samples_per_pixel == 0 {
        return Err("samplesPerPixel must be at least 1".into());
    }

    if config.scene.trim().is_empty() {
        return Err("scene must be a non-empty identifier".into());
    }

    if config.renderer_mode.trim().is_empty() {
        return Err("rendererMode must be a non-empty string".into());
    }

    if !is_finite_vec3(config.camera_origin) || !is_finite_vec3(config.camera_target) {
        return Err("camera vectors must contain finite values".into());
    }

    let camera_origin = vec3_from(config.camera_origin);
    let camera_target = vec3_from(config.camera_target);
    if (camera_origin - camera_target).length() < 0.0001 {
        return Err("cameraOrigin must differ from cameraTarget".into());
    }

    Ok(())
}

fn vec3_from(value: [f32; 3]) -> Vec3 {
    Vec3::new(value[0], value[1], value[2])
}

fn is_finite_vec3(value: [f32; 3]) -> bool {
    value[0].is_finite() && value[1].is_finite() && value[2].is_finite()
}

fn render_cpu(config: &RenderFrameConfig, scene: Scene) -> RgbImage {
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
                    accumulated = accumulated + trace_ray(ray, scene, config.max_depth);
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

struct GpuFrameResources {
    width: u32,
    height: u32,
    padded_bytes_per_row: u32,
    output_texture: wgpu::Texture,
    _output_view: wgpu::TextureView,
    output_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

struct GpuRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
    frame_resources: Option<GpuFrameResources>,
}

impl GpuRenderer {
    async fn new() -> Result<Self, String> {
        if !TEST_FORCE_OPAQUE_RED_MENGER {
            return Err(
                "GPU mode currently supports the opaque-red shading path; set TEST_FORCE_OPAQUE_RED_MENGER=true".to_string(),
            );
        }

        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| "No compatible GPU adapter available".to_string())?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("cobaia-gpu-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                },
                None,
            )
            .await
            .map_err(|error| format!("request_device failed: {error}"))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cobaia-compute-shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(GPU_SHADER_WGSL)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cobaia-bind-group-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cobaia-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cobaia-compute-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cobaia-params-buffer"),
            size: std::mem::size_of::<GpuParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            params_buffer,
            frame_resources: None,
        })
    }

    fn render_frame(
        &mut self,
        config: &RenderFrameConfig,
        scene: Scene,
    ) -> Result<RgbImage, String> {
        self.ensure_frame_resources(config.width, config.height);
        let frame = self
            .frame_resources
            .as_ref()
            .ok_or_else(|| "GPU frame resources are not initialized".to_string())?;

        let gpu_params = GpuParams {
            width: config.width,
            height: config.height,
            max_depth: config.max_depth as u32,
            sponge_iterations: scene.sponge_iterations,
            camera_origin: [
                config.camera_origin[0],
                config.camera_origin[1],
                config.camera_origin[2],
                0.0,
            ],
            camera_target: [
                config.camera_target[0],
                config.camera_target[1],
                config.camera_target[2],
                0.0,
            ],
            sun_direction: [
                scene.sun_direction.x,
                scene.sun_direction.y,
                scene.sun_direction.z,
                0.0,
            ],
            mirror_sphere_center: [
                scene.mirror_sphere_center.x,
                scene.mirror_sphere_center.y,
                scene.mirror_sphere_center.z,
                0.0,
            ],
            floor_y: scene.floor_y,
            sponge_scale: scene.sponge_scale,
            mirror_sphere_radius: scene.mirror_sphere_radius,
            samples_per_pixel: config.samples_per_pixel.max(1) as u32,
        };
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&gpu_params));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cobaia-command-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cobaia-compute-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &frame.bind_group, &[]);
            let groups_x = (config.width + 7) / 8;
            let groups_y = (config.height + 7) / 8;
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &frame.output_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &frame.output_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(frame.padded_bytes_per_row),
                    rows_per_image: Some(config.height),
                },
            },
            wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(Some(encoder.finish()));

        let slice = frame.output_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .map_err(|_| "Failed to receive GPU readback status".to_string())?
            .map_err(|error| format!("GPU readback map failed: {error}"))?;

        let bytes_per_pixel = 4usize;
        let data = slice.get_mapped_range();
        let mut image = RgbImage::new(config.width, config.height);
        for y in 0..config.height as usize {
            let row_start = y * frame.padded_bytes_per_row as usize;
            for x in 0..config.width as usize {
                let pixel_start = row_start + (x * bytes_per_pixel);
                let r = data[pixel_start];
                let g = data[pixel_start + 1];
                let b = data[pixel_start + 2];
                image.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
            }
        }
        drop(data);
        frame.output_buffer.unmap();

        Ok(image)
    }

    fn ensure_frame_resources(&mut self, width: u32, height: u32) {
        let needs_rebuild = match &self.frame_resources {
            Some(resources) => resources.width != width || resources.height != height,
            None => true,
        };
        if !needs_rebuild {
            return;
        }

        let output_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cobaia-output-texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let unpadded_bytes_per_row = width * 4u32;
        let padded_bytes_per_row = ((unpadded_bytes_per_row + 255) / 256) * 256;
        let output_buffer_size = (padded_bytes_per_row * height) as u64;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cobaia-readback-buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cobaia-bind-group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        });

        self.frame_resources = Some(GpuFrameResources {
            width,
            height,
            padded_bytes_per_row,
            output_texture,
            _output_view: output_view,
            output_buffer,
            bind_group,
        });
    }
}

fn trace_ray(ray: Ray, scene: Scene, depth: u8) -> Vec3 {
    if depth == 0 {
        return background_color(ray.direction, scene);
    }

    let Some(hit) = ray_march(ray, scene) else {
        return background_color(ray.direction, scene);
    };

    match hit.material {
        MaterialId::Floor => shade_floor(hit, scene),
        MaterialId::Glass => {
            if TEST_FORCE_OPAQUE_RED_MENGER {
                shade_opaque_red(hit, ray, scene, depth)
            } else {
                shade_glass(hit, ray, scene, depth)
            }
        }
        MaterialId::Mirror => shade_transparent_sphere(hit, ray, scene, depth),
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

fn shade_glass(hit: HitRecord, ray: Ray, scene: Scene, depth: u8) -> Vec3 {
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

fn shade_transparent_sphere(hit: HitRecord, ray: Ray, scene: Scene, depth: u8) -> Vec3 {
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
    );

    let transmitted = trace_scene_skipping_sphere(
        Ray {
            origin: hit.point - (surface_normal * RAY_BIAS),
            direction: incident,
        },
        scene,
        depth.saturating_sub(1),
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

fn shade_opaque_red(hit: HitRecord, ray: Ray, scene: Scene, depth: u8) -> Vec3 {
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
    let base_shaded = ((((lit + sky_tint + bounce_tint) * ao) + highlight) * distance_fade).clamp01();

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
    );

    ((base_shaded * (1.0 - reflectivity)) + (reflected * reflectivity)).clamp01()
}

fn trace_scene_skipping_sphere(ray: Ray, scene: Scene, depth: u8) -> Vec3 {
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
                if TEST_FORCE_OPAQUE_RED_MENGER {
                    return shade_opaque_red(hit, current_ray, scene, depth);
                }
                return shade_glass(hit, current_ray, scene, depth);
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

fn reflect(direction: Vec3, normal: Vec3) -> Vec3 {
    direction - (normal * (2.0 * direction.dot(normal)))
}

fn refract(direction: Vec3, normal: Vec3, eta: f32) -> Option<Vec3> {
    let cos_i = (-direction).dot(normal).clamp(-1.0, 1.0);
    let k = 1.0 - (eta * eta * (1.0 - (cos_i * cos_i)));
    if k < 0.0 {
        None
    } else {
        Some((direction * eta) + (normal * ((eta * cos_i) - k.sqrt())))
    }
}

fn schlick(cosine: f32, eta_i: f32, eta_t: f32) -> f32 {
    let r0 = ((eta_i - eta_t) / (eta_i + eta_t)).powi(2);
    r0 + ((1.0 - r0) * (1.0 - cosine).powf(5.0))
}

fn fract(v: f32) -> f32 {
    v - v.floor()
}

fn hash31(p: Vec3) -> f32 {
    let n = p.dot(Vec3::new(127.1, 311.7, 74.7));
    fract(n.sin() * 43758.5453)
}

fn frosted_offset(p: Vec3, seed: f32) -> Vec3 {
    let s = seed * 13.37;
    let j = Vec3::new(
        (hash31(p + Vec3::new(0.17 + s, 4.7, 9.2)) * 2.0) - 1.0,
        (hash31(p + Vec3::new(5.31, 1.2 + s, 3.4)) * 2.0) - 1.0,
        (hash31(p + Vec3::new(7.83, 8.1, 2.6 + s)) * 2.0) - 1.0,
    );
    j.normalize()
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

fn hash_u32(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^= value >> 16;
    value
}

fn random01(seed: u32) -> f32 {
    hash_u32(seed) as f32 / u32::MAX as f32
}

fn sample_jitter(x: u32, y: u32, sample: u32, axis: u32) -> f32 {
    let seed = x
        .wrapping_mul(1973)
        .wrapping_add(y.wrapping_mul(9277))
        .wrapping_add(sample.wrapping_mul(26699))
        .wrapping_add(axis.wrapping_mul(104_729))
        ^ 0x68bc_21eb;
    random01(seed)
}

const GPU_SHADER_WGSL: &str = r#"
struct Params {
    width: u32,
    height: u32,
    max_depth: u32,
    sponge_iterations: u32,
    camera_origin: vec4<f32>,
    camera_target: vec4<f32>,
    sun_direction: vec4<f32>,
    mirror_sphere_center: vec4<f32>,
    floor_y: f32,
    sponge_scale: f32,
    mirror_sphere_radius: f32,
    samples_per_pixel: u32,
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
    let floor_distance = p.y - params.floor_y;
    let sponge_center = vec3<f32>(0.0, params.floor_y + params.sponge_scale, 0.0);
    let local = (p - sponge_center) / params.sponge_scale;
    let sponge_distance = sd_menger(local) * params.sponge_scale;
    let mirror_sphere_distance = sd_sphere(
        p - params.mirror_sphere_center.xyz,
        params.mirror_sphere_radius
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

fn background_color(direction: vec3<f32>) -> vec3<f32> {
    let unit = normalize(direction);
    let t = 0.5 * (unit.y + 1.0);
    let top = vec3<f32>(0.5, 0.71, 0.94);
    let bottom = vec3<f32>(0.98, 0.99, 1.0);
    let base = (bottom * (1.0 - t)) + (top * t);

    let sun_alignment = max(dot(unit, -params.sun_direction.xyz), 0.0);
    let sun = vec3<f32>(1.0, 0.96, 0.9) * pow(sun_alignment, 420.0) * 6.0;
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
    let shade = ambient + (0.92 * direct);

    let base = vec3<f32>(0.94, 0.94, 0.93);
    let hemi = 0.5 * (hit.normal.y + 1.0);
    let bounce = vec3<f32>(0.08, 0.1, 0.12) * (0.03 * hemi);
    let distance_fade = clamp(1.0 - (hit.t * 0.012), 0.7, 1.0);

    return clamp((base * shade * distance_fade) + bounce, vec3<f32>(0.0), vec3<f32>(1.0));
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

    let base = vec3<f32>(0.9, 0.09, 0.08);
    let lit = base * (ambient + (0.92 * diffuse));
    let sky_tint = vec3<f32>(0.24, 0.32, 0.44) * (0.12 * hemi);
    let bounce_tint = vec3<f32>(0.18, 0.04, 0.03) * (0.08 * (1.0 - hemi));
    let highlight = vec3<f32>(spec * 0.2);
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
    let highlight = vec3<f32>(sun_spec * sun_shadow * 0.06);
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

    let theta = radians(38.0);
    let h = tan(theta * 0.5);
    let viewport_height = 2.0 * h;
    let viewport_width = aspect_ratio * viewport_height;

    let w = normalize(origin - camera_target);
    var up = vec3<f32>(0.0, 1.0, 0.0);
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
"#;
