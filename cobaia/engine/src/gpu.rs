use image::{Rgb, RgbImage};

use crate::domain::{LightKind, Material, MaterialClass, ObjectKind, Scene};
use crate::math::Vec3;
use crate::render::{RenderSettings, View};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParams {
    width: u32,
    height: u32,
    max_depth: u32,
    sponge_iterations: u32,
    samples_per_pixel: u32,
    _padding_u0: u32,
    _padding_u1: u32,
    _padding_u2: u32,
    camera_origin: [f32; 4],
    camera_target: [f32; 4],
    camera_up: [f32; 4],
    sponge_center: [f32; 4],
    sun_direction: [f32; 4],
    sun_color_intensity: [f32; 4],
    floor_base_color: [f32; 4],
    menger_base_color: [f32; 4],
    mirror_sphere_center: [f32; 4],
    scene_scalars: [f32; 4],
}

#[derive(Clone, Copy, Debug)]
struct CompiledGpuScene {
    floor_y: f32,
    sponge_center: Vec3,
    sponge_scale: f32,
    sponge_iterations: u32,
    mirror_sphere_center: Vec3,
    mirror_sphere_radius: f32,
    floor_color: Vec3,
    menger_color: Vec3,
    sun_direction: Vec3,
    sun_color: Vec3,
    sun_intensity: f32,
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

pub struct GpuRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
    frame_resources: Option<GpuFrameResources>,
}

impl GpuRenderer {
    pub async fn new() -> Result<Self, String> {
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
            source: wgpu::ShaderSource::Wgsl(GPU_SHADER_WGSL.into()),
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

    pub fn render_frame(
        &mut self,
        settings: &RenderSettings,
        view: &View,
        scene: &Scene,
    ) -> Result<RgbImage, String> {
        let compiled_scene = compile_scene(scene)?;

        self.ensure_frame_resources(settings.width, settings.height);
        let frame = self
            .frame_resources
            .as_ref()
            .ok_or_else(|| "GPU frame resources are not initialized".to_string())?;

        let gpu_params = GpuParams {
            width: settings.width,
            height: settings.height,
            max_depth: settings.max_depth,
            sponge_iterations: compiled_scene.sponge_iterations,
            samples_per_pixel: settings.samples_per_pixel,
            _padding_u0: 0,
            _padding_u1: 0,
            _padding_u2: 0,
            camera_origin: [view.origin.x, view.origin.y, view.origin.z, 0.0],
            camera_target: [view.target.x, view.target.y, view.target.z, 0.0],
            camera_up: [view.up.x, view.up.y, view.up.z, 0.0],
            sponge_center: [
                compiled_scene.sponge_center.x,
                compiled_scene.sponge_center.y,
                compiled_scene.sponge_center.z,
                0.0,
            ],
            sun_direction: [
                compiled_scene.sun_direction.x,
                compiled_scene.sun_direction.y,
                compiled_scene.sun_direction.z,
                0.0,
            ],
            sun_color_intensity: [
                compiled_scene.sun_color.x,
                compiled_scene.sun_color.y,
                compiled_scene.sun_color.z,
                compiled_scene.sun_intensity,
            ],
            floor_base_color: [
                compiled_scene.floor_color.x,
                compiled_scene.floor_color.y,
                compiled_scene.floor_color.z,
                0.0,
            ],
            menger_base_color: [
                compiled_scene.menger_color.x,
                compiled_scene.menger_color.y,
                compiled_scene.menger_color.z,
                0.0,
            ],
            mirror_sphere_center: [
                compiled_scene.mirror_sphere_center.x,
                compiled_scene.mirror_sphere_center.y,
                compiled_scene.mirror_sphere_center.z,
                0.0,
            ],
            scene_scalars: [
                compiled_scene.floor_y,
                compiled_scene.sponge_scale,
                compiled_scene.mirror_sphere_radius,
                view.vertical_fov_deg,
            ],
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
            let groups_x = (settings.width + 7) / 8;
            let groups_y = (settings.height + 7) / 8;
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
                    rows_per_image: Some(settings.height),
                },
            },
            wgpu::Extent3d {
                width: settings.width,
                height: settings.height,
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
        let mut image = RgbImage::new(settings.width, settings.height);
        for y in 0..settings.height as usize {
            let row_start = y * frame.padded_bytes_per_row as usize;
            for x in 0..settings.width as usize {
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

fn compile_scene(scene: &Scene) -> Result<CompiledGpuScene, String> {
    let mut floor_y: Option<f32> = None;
    let mut floor_color: Option<Vec3> = None;
    let mut sponge_center: Option<Vec3> = None;
    let mut sponge_scale: Option<f32> = None;
    let mut sponge_iterations: Option<u32> = None;
    let mut menger_color: Option<Vec3> = None;
    let mut mirror_sphere_center: Option<Vec3> = None;
    let mut mirror_sphere_radius: Option<f32> = None;

    for object in &scene.objects {
        match object.kind {
            ObjectKind::InfinitePlane { y } => {
                let material = material_with_class(
                    scene,
                    object.name,
                    object.material_id.0,
                    MaterialClass::Floor,
                )?;
                if floor_y.replace(y).is_some() {
                    return Err("scene has multiple floor planes; only one is supported".into());
                }
                floor_color = Some(material.albedo);
            }
            ObjectKind::Menger {
                center,
                scale,
                iterations,
            } => {
                let material = material_with_class(
                    scene,
                    object.name,
                    object.material_id.0,
                    MaterialClass::Opaque,
                )?;
                if sponge_center.replace(center).is_some() {
                    return Err("scene has multiple Menger objects; only one is supported".into());
                }
                sponge_scale = Some(scale);
                sponge_iterations = Some(iterations);
                menger_color = Some(material.albedo);
            }
            ObjectKind::Sphere { center, radius } => {
                let _ = material_with_class(
                    scene,
                    object.name,
                    object.material_id.0,
                    MaterialClass::Mirror,
                )?;
                if mirror_sphere_center.replace(center).is_some() {
                    return Err("scene has multiple spheres; only one is supported".into());
                }
                mirror_sphere_radius = Some(radius);
            }
        }
    }

    let mut sun_direction: Option<Vec3> = None;
    let mut sun_color: Option<Vec3> = None;
    let mut sun_intensity: Option<f32> = None;
    for light in &scene.lights {
        match light.kind {
            LightKind::Directional {
                direction,
                color,
                intensity,
            } => {
                if intensity <= 0.0 {
                    return Err(format!(
                        "light '{}' has non-positive intensity ({intensity})",
                        light.name
                    ));
                }
                if sun_direction.replace(direction.normalize()).is_some() {
                    return Err(
                        "scene has multiple directional lights; only one is supported".into(),
                    );
                }
                sun_color = Some(color);
                sun_intensity = Some(intensity);
            }
        }
    }

    let floor_y = floor_y.ok_or_else(|| "scene must include one floor plane".to_string())?;
    let floor_color =
        floor_color.ok_or_else(|| "scene must include a floor material color".to_string())?;
    let sponge_center =
        sponge_center.ok_or_else(|| "scene must include one Menger object".to_string())?;
    let sponge_scale = sponge_scale.ok_or_else(|| "scene must include Menger scale".to_string())?;
    if sponge_scale <= 0.0 {
        return Err(format!("Menger scale must be positive, got {sponge_scale}"));
    }
    let sponge_iterations =
        sponge_iterations.ok_or_else(|| "scene must include Menger iterations".to_string())?;
    if sponge_iterations == 0 {
        return Err("Menger iterations must be greater than zero".into());
    }
    let menger_color =
        menger_color.ok_or_else(|| "scene must include a Menger material color".to_string())?;
    let mirror_sphere_center =
        mirror_sphere_center.ok_or_else(|| "scene must include one sphere".to_string())?;
    let mirror_sphere_radius =
        mirror_sphere_radius.ok_or_else(|| "scene must include sphere radius".to_string())?;
    if mirror_sphere_radius <= 0.0 {
        return Err(format!(
            "sphere radius must be positive, got {mirror_sphere_radius}"
        ));
    }
    let sun_direction =
        sun_direction.ok_or_else(|| "scene must include one directional light".to_string())?;
    if sun_direction.length() < 0.0001 {
        return Err("directional light direction must be non-zero".into());
    }

    let sun_color =
        sun_color.ok_or_else(|| "scene must include directional light color".to_string())?;
    let sun_intensity = sun_intensity
        .ok_or_else(|| "scene must include directional light intensity".to_string())?;

    Ok(CompiledGpuScene {
        floor_y,
        sponge_center,
        sponge_scale,
        sponge_iterations,
        mirror_sphere_center,
        mirror_sphere_radius,
        floor_color,
        menger_color,
        sun_direction,
        sun_color,
        sun_intensity,
    })
}

fn material_with_class<'a>(
    scene: &'a Scene,
    object_name: &str,
    material_id: usize,
    expected: MaterialClass,
) -> Result<&'a Material, String> {
    let material = scene.materials.get(material_id).ok_or_else(|| {
        format!("object '{object_name}' references missing material id {material_id}")
    })?;
    if material.class != expected {
        return Err(format!(
            "object '{}' expects {:?} material class but got {:?} ({})",
            object_name, expected, material.class, material.name
        ));
    }
    Ok(material)
}

const GPU_SHADER_WGSL: &str = include_str!("shaders/raytrace.wgsl");
