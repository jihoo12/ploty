use std::sync::Arc;
use std::time::Instant;

use glam::Mat4;
use glyphon::{
    Attrs, Buffer as GlyphBuffer, Cache as GlyphCache, Color as GlyphColor, Family, FontSystem,
    Metrics, Resolution, Shaping, SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer,
    Viewport,
};
use wgpu::util::DeviceExt;
use winit::{dpi::PhysicalSize, window::Window};

use super::{
    camera::Camera,
    config::LegendEntry,
    data::{AnimatedGraph, PlotData},
    geometry::{create_full_grid_data, plot_wireframe},
    mesh::{merge_meshes, Mesh},
    vertex::Vertex,
};

// ---------------------------------------------------------------------------
// WGSL 셰이더
// ---------------------------------------------------------------------------

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
            vertex_buf: upload(
                bytemuck::cast_slice(&mesh.vertices),
                wgpu::BufferUsages::VERTEX,
            ),
            index_buf: upload(
                bytemuck::cast_slice(&mesh.indices),
                wgpu::BufferUsages::INDEX,
            ),
            index_count: mesh.indices.len() as u32,
        }
    }

    /// 애니메이션용: 정점 수가 고정이라는 가정 하에 COPY_DST 버퍼로 생성합니다.
    fn upload_dynamic(device: &wgpu::Device, mesh: &Mesh) -> Self {
        let vdata: &[u8] = bytemuck::cast_slice(&mesh.vertices);
        let idata: &[u8] = bytemuck::cast_slice(&mesh.indices);
        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("anim vertex"),
            contents: vdata,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("anim index"),
            contents: idata,
            usage: wgpu::BufferUsages::INDEX,
        });
        Self {
            vertex_buf,
            index_buf,
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

    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,

    line_pipeline: wgpu::RenderPipeline,
    point_pipeline: wgpu::RenderPipeline,

    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    depth_view: wgpu::TextureView,

    grid: GpuMesh,
    graph: GpuMesh,
    scatter: GpuMesh,

    // 애니메이션
    animated_graphs: Vec<AnimatedGraph>,
    animated_gpu: Vec<GpuMesh>,
    start_time: Instant,

    // 범례 (glyphon)
    font_system: FontSystem,
    swash_cache: SwashCache,
    glyph_cache: GlyphCache,
    text_atlas: TextAtlas,
    text_renderer: TextRenderer,
    viewport: Viewport,
    /// (GlyphBuffer, 색상) 쌍
    legend_buffers: Vec<(GlyphBuffer, [f32; 3])>,

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
        // wgpu 29: bind_group_layouts 는 &[Option<&BindGroupLayout>]
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

        // ── 애니메이션 GPU 버퍼 초기 업로드 ─────────────────────────────────
        let mut animated_gpu = Vec::with_capacity(data.animated_graphs.len());
        for anim in &data.animated_graphs {
            let mesh = plot_wireframe(
                &anim.x_range,
                &anim.z_range,
                |x, z| (anim.func)(x, z, 0.0),
                anim.base_color,
            );
            animated_gpu.push(GpuMesh::upload_dynamic(&device, &mesh));
        }

        let depth_view = Self::make_depth_view(&device, size.width, size.height);

        // ── glyphon 초기화 ───────────────────────────────────────────────────
        let mut font_system = FontSystem::new();
        let swash_cache = SwashCache::new();
        let glyph_cache = GlyphCache::new(&device);
        let mut text_atlas = TextAtlas::new(&device, &queue, &glyph_cache, format);
        let text_renderer = TextRenderer::new(
            &mut text_atlas,
            &device,
            wgpu::MultisampleState::default(),
            Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
        );
        let viewport = Viewport::new(&device, &glyph_cache);

        let legend_buffers =
            Self::build_legend_buffers(&plot_config.legend, &mut font_system);

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
            animated_graphs: data.animated_graphs,
            animated_gpu,
            start_time: Instant::now(),
            font_system,
            swash_cache,
            glyph_cache,
            text_atlas,
            text_renderer,
            viewport,
            legend_buffers,
            background_color,
        }
    }

    // ── 범례 버퍼 생성 ────────────────────────────────────────────────────────

    fn build_legend_buffers(
        entries: &[LegendEntry],
        font_system: &mut FontSystem,
    ) -> Vec<(GlyphBuffer, [f32; 3])> {
        entries
            .iter()
            .map(|e| {
                let mut buf = GlyphBuffer::new(font_system, Metrics::new(18.0, 22.0));
                buf.set_size(font_system, Some(300.0), Some(30.0));
                buf.set_text(
                    font_system,
                    &e.label,
                    &Attrs::new().family(Family::SansSerif),
                    Shaping::Advanced,
                    None,
                );
                (buf, e.color)
            })
            .collect()
    }

    // ── resize ────────────────────────────────────────────────────────────────

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        self.size = new_size;
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
        self.depth_view =
            Self::make_depth_view(&self.device, new_size.width, new_size.height);
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
                // wgpu 29: buffers 는 &[Option<VertexBufferLayout>]
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

    /// 카메라 행렬 업로드 + 애니메이션 메시 갱신. render() 전에 호출합니다.
    pub fn update(&mut self) {
        let aspect = self.size.width as f32 / self.size.height as f32;
        let view_proj = self.camera.view_proj_matrix(aspect);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&view_proj.to_cols_array()),
        );

        // 애니메이션: 경과 시간 t 를 클로저에 넘겨 메시 재생성 → GPU 덮어쓰기
        let t = self.start_time.elapsed().as_secs_f32();
        for (i, anim) in self.animated_graphs.iter().enumerate() {
            let mesh = plot_wireframe(
                &anim.x_range,
                &anim.z_range,
                |x, z| (anim.func)(x, z, t),
                anim.base_color,
            );
            self.queue.write_buffer(
                &self.animated_gpu[i].vertex_buf,
                0,
                bytemuck::cast_slice(&mesh.vertices),
            );
        }
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

        // ── glyphon viewport 갱신 ─────────────────────────────────────────
        self.viewport.update(
            &self.queue,
            Resolution {
                width: self.size.width,
                height: self.size.height,
            },
        );

        // 범례 TextArea 목록: 우측 상단, 세로 나열
        // 텍스트 색상으로 계열 구분 (추가 geometry 없이 간결하게)
        let padding = 16.0f32;
        let row_h   = 28.0f32;
        let text_x  = self.size.width as f32 - 220.0;

        let text_areas: Vec<TextArea> = self
            .legend_buffers
            .iter()
            .enumerate()
            .map(|(i, (buf, color))| TextArea {
                buffer: buf,
                left: text_x,
                top: padding + i as f32 * row_h,
                scale: 1.0,
                bounds: TextBounds::default(),
                default_color: GlyphColor::rgb(
                    (color[0] * 255.0) as u8,
                    (color[1] * 255.0) as u8,
                    (color[2] * 255.0) as u8,
                ),
                custom_glyphs: &[],
            })
            .collect();

        if !text_areas.is_empty() {
            self.text_renderer
                .prepare(
                    &self.device,
                    &self.queue,
                    &mut self.font_system,
                    &mut self.text_atlas,
                    &self.viewport,
                    text_areas,
                    &mut self.swash_cache,
                )
                .unwrap();
        }

        // ── 렌더 커맨드 ──────────────────────────────────────────────────────
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

            // 정적 메시
            rp.set_pipeline(&self.line_pipeline);
            self.draw_mesh(&mut rp, &self.grid);
            self.draw_mesh(&mut rp, &self.graph);

            // 애니메이션 메시 (borrow 충돌 회피를 위해 인덱스로 순회)
            for i in 0..self.animated_gpu.len() {
                let gpu = &self.animated_gpu[i] as *const GpuMesh;
                // SAFETY: animated_gpu 는 rp 안에서 수정되지 않습니다.
                self.draw_mesh(&mut rp, unsafe { &*gpu });
            }

            // 산점도
            if self.scatter.index_count > 0 {
                rp.set_pipeline(&self.point_pipeline);
                self.draw_mesh(&mut rp, &self.scatter);
            }

            // 범례 텍스트 오버레이 (같은 렌더패스 내)
            if !self.legend_buffers.is_empty() {
                self.text_renderer
                    .render(&self.text_atlas, &self.viewport, &mut rp)
                    .unwrap();
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        surface_texture.present();

        // 매 프레임 atlas LRU 정리
        self.text_atlas.trim();

        Ok(())
    }
}