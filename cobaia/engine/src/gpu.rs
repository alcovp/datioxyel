use image::{Rgb, RgbImage};

use crate::config::{DebugOptions, RenderFrameConfig};
use crate::scene::Scene;

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
    pub async fn new(debug: DebugOptions) -> Result<Self, String> {
        if !debug.force_opaque_red_menger {
            return Err(
                "GPU mode currently supports the opaque-red shading path; set COBAIA_FORCE_OPAQUE_RED_MENGER=true"
                    .to_string(),
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

const GPU_SHADER_WGSL: &str = include_str!("shaders/raytrace.wgsl");
