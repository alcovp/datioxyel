use image::{Rgb, RgbImage};

use crate::domain::Scene;
use crate::render::capabilities::{GPU_MAX_LIGHTS, GPU_MAX_MATERIALS, GPU_MAX_OBJECTS};
use crate::render::{RenderSettings, View};

mod scene_compile;
mod shader_source;

use scene_compile::compile_scene;
use shader_source::build_gpu_shader_wgsl;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParams {
    width: u32,
    height: u32,
    max_depth: u32,
    samples_per_pixel: u32,
    object_count: u32,
    material_count: u32,
    light_count: u32,
    _padding_u0: u32,
    camera_origin: [f32; 4],
    camera_target: [f32; 4],
    camera_up: [f32; 4],
    object_meta: [[f32; 4]; GPU_MAX_OBJECTS],
    object_data0: [[f32; 4]; GPU_MAX_OBJECTS],
    object_data1: [[f32; 4]; GPU_MAX_OBJECTS],
    material_albedo_roughness: [[f32; 4]; GPU_MAX_MATERIALS],
    material_emission_metallic: [[f32; 4]; GPU_MAX_MATERIALS],
    material_optics: [[f32; 4]; GPU_MAX_MATERIALS],
    material_absorption: [[f32; 4]; GPU_MAX_MATERIALS],
    light_direction: [[f32; 4]; GPU_MAX_LIGHTS],
    light_color_intensity: [[f32; 4]; GPU_MAX_LIGHTS],
    scene_scalars: [f32; 4],
    render_tuning0: [f32; 4],
    render_tuning1: [f32; 4],
    render_tuning2: [f32; 4],
    render_tuning3: [f32; 4],
}

struct GpuFrameResources {
    bind_group: wgpu::BindGroup,
    output_buffer: wgpu::Buffer,
    output_texture: wgpu::Texture,
    _output_view: wgpu::TextureView,
    width: u32,
    height: u32,
    padded_bytes_per_row: u32,
}

pub struct GpuRenderer {
    // Keep dependent GPU resources before queue/device so they drop first.
    frame_resources: Option<GpuFrameResources>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
    queue: wgpu::Queue,
    device: wgpu::Device,
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

        let shader_source = build_gpu_shader_wgsl();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cobaia-compute-shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
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
            frame_resources: None,
            pipeline,
            bind_group_layout,
            params_buffer,
            queue,
            device,
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
            samples_per_pixel: settings.samples_per_pixel,
            object_count: compiled_scene.object_count,
            material_count: compiled_scene.material_count,
            light_count: compiled_scene.light_count,
            _padding_u0: 0,
            camera_origin: [view.origin.x, view.origin.y, view.origin.z, 0.0],
            camera_target: [view.target.x, view.target.y, view.target.z, 0.0],
            camera_up: [view.up.x, view.up.y, view.up.z, 0.0],
            object_meta: compiled_scene.object_meta,
            object_data0: compiled_scene.object_data0,
            object_data1: compiled_scene.object_data1,
            material_albedo_roughness: compiled_scene.material_albedo_roughness,
            material_emission_metallic: compiled_scene.material_emission_metallic,
            material_optics: compiled_scene.material_optics,
            material_absorption: compiled_scene.material_absorption,
            light_direction: compiled_scene.light_direction,
            light_color_intensity: compiled_scene.light_color_intensity,
            scene_scalars: [view.vertical_fov_deg, 0.0, 0.0, 0.0],
            render_tuning0: [
                settings.tuning.march_max_steps as f32,
                settings.tuning.hit_epsilon_scale,
                settings.tuning.step_scale,
                settings.tuning.rr_start_bounce as f32,
            ],
            render_tuning1: [
                settings.tuning.shadow_max_steps as f32,
                settings.tuning.ao_samples as f32,
                settings.tuning.sampling_mode as u32 as f32,
                0.0,
            ],
            render_tuning2: [
                settings.tuning.adaptive_min_samples_fraction,
                settings.tuning.adaptive_variance_threshold,
                settings.tuning.adaptive_check_interval as f32,
                settings.tuning.firefly_clamp_scale,
            ],
            render_tuning3: [
                settings.tuning.shadow_distance_scale,
                settings.tuning.shadow_min_step_scale,
                settings.tuning.ao_radius_scale,
                settings.tuning.ao_horizon_bias,
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
            bind_group,
            output_buffer,
            output_texture,
            _output_view: output_view,
            width,
            height,
            padded_bytes_per_row,
        });
    }
}

impl Drop for GpuRenderer {
    fn drop(&mut self) {
        // Explicitly drop per-frame resources first, then drain pending GPU work.
        self.frame_resources = None;
        self.device.poll(wgpu::Maintain::Wait);
    }
}
