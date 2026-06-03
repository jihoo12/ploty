use std::sync::Arc;

use glam::Mat4;
use wgpu::util::DeviceExt;
use winit::{dpi::PhysicalSize, window::Window};

use super::{
    camera::Camera,
    config::PlotConfig,
    data::PlotData,
    geometry::create_full_grid_data,
    mesh::{merge_meshes, Mesh},
    vertex::Vertex,
};

const SHADER_SOURCE: &str = r#"
    struct Camera { view_proj: mat4x4<f32> }
    @group(0) @binding(0) var<uniform> camera: Camera;

    struct VertexIn  { @location(0) pos: vec4<f32>, @location(1) col: vec4<f32> }
    struct VertexOut { @builtin(position) pos: vec4<f32>, @location(0) col: vec4<f32> }

    @vertex
    fn vs_main(in: VertexIn) -> VertexOut {
        return VertexOut(camera.view_proj * in.pos, in.col);
    }

    @fragment
    fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
        return in.col;
    }
"#;

// ---------------------------------------------------------------------------
// GpuMesh — GPU 버퍼 쌍
// ---------------------------------------------------------------------------

struct GpuMesh {
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    index_count: u32,
}

impl GpuMesh {
    fn upload(device: &wgpu::Device, mesh: &Mesh) -> Self {
        let upload = |data: &[u8], usage: wgpu::BufferUsages| {
            let contents = if data.is_empty() { &[0u8; 4] } else { data };
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents,
                usage,
            })
        };
        Self {
            vertex_buf: upload(bytemuck::cast_slice(&mesh.vertices), wgpu::BufferUsages::VERTEX),
            index_buf: upload(bytemuck::cast_slice(&mesh.indices), wgpu::BufferUsages::INDEX),
            index_count: mesh.indices.len() as u32,
        }
    }
}

// ---------------------------------------------------------------------------
// App — wgpu 렌더러
// ---------------------------------------------------------------------------

pub struct App<'a> {
    /// 궤도 카메라. 입력 이벤트를 직접 전달하세요.
    pub camera: Camera,

    // wgpu 핵심 객체 (내부 전용)
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,

    // 파이프라인
    line_pipeline: wgpu::RenderPipeline,
    point_pipeline: wgpu::RenderPipeline,

    // 카메라 유니폼
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    depth_view: wgpu::TextureView,

    // GPU 메시
    grid: GpuMesh,
    graph: GpuMesh,
    scatter: GpuMesh,

    // 렌더 설정
    background_color: [f64; 4],
}

impl<'a> App<'a> {
    pub async fn new(window: Arc<Window>, data: PlotData) -> Self {
        let size = window.inner_size();
        let plot_config = &data.config;

        // ── wgpu 초기화 ──────────────────────────────────────────────────────
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats[0];
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // ── 셰이더 ───────────────────────────────────────────────────────────
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Plot Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
        });

        // ── 카메라 유니폼 ────────────────────────────────────────────────────
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera Buffer"),
            size: std::mem::size_of::<Mat4>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Camera BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera Bind Group"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        // ── 파이프라인 ───────────────────────────────────────────────────────
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[Some(&bgl)],
            ..Default::default()
        });
        let line_pipeline = Self::build_pipeline(
            &device, &shader, &pipeline_layout, format,
            wgpu::PrimitiveTopology::LineList, "Line Pipeline",
        );
        let point_pipeline = Self::build_pipeline(
            &device, &shader, &pipeline_layout, format,
            wgpu::PrimitiveTopology::PointList, "Point Pipeline",
        );

        // ── GPU 메시 업로드 ──────────────────────────────────────────────────
        let grid_mesh = create_full_grid_data(plot_config.grid_size, plot_config.grid_divisions);
        let background_color = plot_config.background_color;

        let grid    = GpuMesh::upload(&device, &grid_mesh);
        let graph   = GpuMesh::upload(&device, &merge_meshes(data.graphs));
        let scatter = GpuMesh::upload(&device, &merge_meshes(data.scatters));

        let depth_view = Self::make_depth_view(&device, size.width, size.height);

        Self {
            camera: Camera::new(),
            surface,
            device,
            queue,
            config,
            size,
            line_pipeline,
            point_pipeline,
            camera_buffer,
            camera_bind_group,
            depth_view,
            grid,
            graph,
            scatter,
            background_color,
        }
    }

    // ① resize 캡슐화 ─────────────────────────────────────────────────────────
    /// 윈도우 크기 변경 시 호출합니다. surface·depth 버퍼를 내부에서 재구성합니다.
    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        self.size = new_size;
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
        self.depth_view = Self::make_depth_view(&self.device, new_size.width, new_size.height);
    }

    // ── 내부 헬퍼 ────────────────────────────────────────────────────────────

    fn make_depth_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
        device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("Depth Texture"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn build_pipeline(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        layout: &wgpu::PipelineLayout,
        format: wgpu::TextureFormat,
        topology: wgpu::PrimitiveTopology,
        label: &str,
    ) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState { topology, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
    }

    fn draw_mesh<'rp>(&'rp self, rp: &mut wgpu::RenderPass<'rp>, mesh: &'rp GpuMesh) {
        if mesh.index_count == 0 {
            return;
        }
        rp.set_vertex_buffer(0, mesh.vertex_buf.slice(..));
        rp.set_index_buffer(mesh.index_buf.slice(..), wgpu::IndexFormat::Uint32);
        rp.draw_indexed(0..mesh.index_count, 0, 0..1);
    }

    // ── 퍼블릭 프레임 API ────────────────────────────────────────────────────

    /// 카메라 행렬을 GPU 버퍼에 씁니다. 매 프레임 `render()` 전에 호출합니다.
    pub fn update(&mut self) {
        let aspect = self.size.width as f32 / self.size.height as f32;
        let view_proj = self.camera.view_proj_matrix(aspect);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&view_proj.to_cols_array()),
        );
    }

    pub fn render(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) => t,
            wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            other => {
                eprintln!("Surface texture error: {:?}", other);
                return Ok(());
            }
        };

        let view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        {
            let [r, g, b, a] = self.background_color;
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r, g, b, a }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            rp.set_bind_group(0, &self.camera_bind_group, &[]);

            rp.set_pipeline(&self.line_pipeline);
            self.draw_mesh(&mut rp, &self.grid);
            self.draw_mesh(&mut rp, &self.graph);

            if self.scatter.index_count > 0 {
                rp.set_pipeline(&self.point_pipeline);
                self.draw_mesh(&mut rp, &self.scatter);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        surface_texture.present();
        Ok(())
    }
}