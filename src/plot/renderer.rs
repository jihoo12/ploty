use std::sync::Arc;
use std::time::Instant;

use glam::{Mat4, Vec3, Vec4};
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
    mesh::{Mesh, merge_meshes},
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
        // wgpu은 빈 버퍼를 허용하지 않으므로 최소 4바이트를 확보합니다.
        let make_buf = |data: &[u8], usage: wgpu::BufferUsages| {
            let contents = if data.is_empty() { &[0u8; 4] } else { data };
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents,
                usage,
            })
        };
        Self {
            vertex_buf: make_buf(
                bytemuck::cast_slice(&mesh.vertices),
                wgpu::BufferUsages::VERTEX,
            ),
            index_buf: make_buf(
                bytemuck::cast_slice(&mesh.indices),
                wgpu::BufferUsages::INDEX,
            ),
            index_count: mesh.indices.len() as u32,
        }
    }

    /// 애니메이션용: 정점 버퍼에 COPY_DST를 추가해 매 프레임 덮어씁니다.
    /// 인덱스는 변하지 않으므로 정적 버퍼를 사용합니다.
    fn upload_dynamic(device: &wgpu::Device, mesh: &Mesh) -> Self {
        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("anim vertex"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("anim index"),
            contents: bytemuck::cast_slice(&mesh.indices),
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
    /// 매 프레임 재사용되는 CPU-side 정점 스크래치 버퍼.
    /// 각 항목은 해당 인덱스의 animated_graph에 대응합니다.
    anim_scratch: Vec<Vec<Vertex>>,
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

    /// 배경색 RGBA — f32 [0, 1] 범위.
    background_color: [f32; 4],

    /// 직전 프레임의 뷰-프로젝션 행렬. render()에서 축 레이블 투영에 사용합니다.
    view_proj: Mat4,

    /// 축 눈금 레이블: (월드 좌표, GlyphBuffer) 쌍.
    /// X/Y/Z 축 각각 grid_divisions + 1개씩 저장합니다.
    axis_labels: Vec<(Vec3, GlyphBuffer)>,
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
                force_fallback_adapter: false,
                ..Default::default()
            })
            .await
            .expect("failed to find a GPU adapter");
        let info = adapter.get_info();

        println!("--- wgpu renderer info ---");
        println!("adapter name (GPU): {}", info.name);
        println!("backend API: {:?}", info.backend); // Vulkan, Gl, Dx12, Metal 등
        println!("driver name: {}", info.driver);
        println!("driver details: {}", info.driver_info);
        println!("--------------------------");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();

        let caps = surface.get_capabilities(&adapter);

        // sRGB 포맷을 우선 선택하고, 없으면 첫 번째 지원 포맷으로 폴백합니다.
        // caps.formats가 비어 있으면 이 플랫폼에서는 렌더링 불가능합니다.
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or_else(|| {
                *caps.formats.first().expect("no supported surface formats")
            });

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width,
            height: size.height,
            // AutoVsync: 가능하면 Mailbox, 없으면 Fifo로 자동 선택합니다.
            present_mode: wgpu::PresentMode::AutoVsync,
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
            &device,
            &shader,
            &pipeline_layout,
            format,
            wgpu::PrimitiveTopology::LineList,
            "Line Pipeline",
        );
        let point_pipeline = Self::build_pipeline(
            &device,
            &shader,
            &pipeline_layout,
            format,
            wgpu::PrimitiveTopology::PointList,
            "Point Pipeline",
        );

        // ── GPU 메시 업로드 ──────────────────────────────────────────────────
        let grid_mesh = create_full_grid_data(plot_config.grid_size, plot_config.grid_divisions);
        let background_color = plot_config.background_color;

        let grid    = GpuMesh::upload(&device, &grid_mesh);
        let graph   = GpuMesh::upload(&device, &merge_meshes(data.graphs));
        let scatter = GpuMesh::upload(&device, &merge_meshes(data.scatters));

        // ── 애니메이션 GPU 버퍼 + 스크래치 버퍼 초기 업로드 ─────────────────
        let mut anim_scratch: Vec<Vec<Vertex>> = Vec::with_capacity(data.animated_graphs.len());
        let animated_gpu: Vec<GpuMesh> = data
            .animated_graphs
            .iter()
            .map(|anim| {
                let mesh = plot_wireframe(
                    &anim.x_range,
                    &anim.z_range,
                    |x, z| (anim.func)(x, z, 0.0),
                    anim.base_color,
                );
                // 스크래치 버퍼를 정점 수에 맞게 미리 할당해 둡니다.
                anim_scratch.push(mesh.vertices.clone());
                GpuMesh::upload_dynamic(&device, &mesh)
            })
            .collect();

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
                // 텍스트는 깊이 버퍼에 쓰지 않고 항상 위에 그립니다.
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
        );
        let viewport = Viewport::new(&device, &glyph_cache);
        let legend_buffers = Self::build_legend_buffers(&plot_config.legend, &mut font_system);
        let axis_labels = if plot_config.show_axis_labels {
            Self::build_axis_label_buffers(
                plot_config.grid_size,
                plot_config.grid_divisions,
                &mut font_system,
            )
        } else {
            vec![]
        };

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
            anim_scratch,
            start_time: Instant::now(),
            font_system,
            swash_cache,
            glyph_cache,
            text_atlas,
            text_renderer,
            viewport,
            legend_buffers,
            background_color,
            view_proj: Mat4::IDENTITY,
            axis_labels,
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

    // ── 축 눈금 레이블 버퍼 생성 ──────────────────────────────────────────────

    /// 격자 분할 눈금에 맞춰 X / Z / Y 축 레이블을 만듭니다.
    ///
    /// 반환값: `(월드 좌표, GlyphBuffer)` 벡터.
    /// - X축: 바닥(y = −half), 앞면(z = −half) 모서리를 따라 배치
    /// - Z축: 바닥(y = −half), 왼쪽(x = −half) 모서리를 따라 배치
    /// - Y축: 왼쪽(x = −half), 뒷면(z = −half) 모서리를 따라 배치
    fn build_axis_label_buffers(
        grid_size: f32,
        divisions: usize,
        font_system: &mut FontSystem,
    ) -> Vec<(Vec3, GlyphBuffer)> {
        let half = grid_size / 2.0;
        let step = grid_size / divisions as f32;

        // 눈금 하나짜리 GlyphBuffer를 만드는 내부 헬퍼
        let make_buf = |font_system: &mut FontSystem, text: &str| {
            let mut buf = GlyphBuffer::new(font_system, Metrics::new(12.0, 16.0));
            buf.set_size(font_system, Some(70.0), Some(20.0));
            buf.set_text(
                font_system,
                text,
                &Attrs::new().family(Family::Monospace),
                Shaping::Basic,
                None,
            );
            buf
        };

        let tick_count = divisions + 1;
        let mut labels = Vec::with_capacity(tick_count * 3);

        for i in 0..tick_count {
            let v = -half + i as f32 * step;
            let text = format_tick(v);

            // X축: 바닥 앞쪽 모서리 (y = −half, z = −half)
            labels.push((Vec3::new(v, -half, -half), make_buf(font_system, &text)));
            // Z축: 바닥 왼쪽 모서리 (y = −half, x = −half)
            labels.push((Vec3::new(-half, -half, v), make_buf(font_system, &text)));
            // Y축: 왼쪽 뒤쪽 모서리 (x = −half, z = −half)
            labels.push((Vec3::new(-half, v, -half), make_buf(font_system, &text)));
        }

        labels
    }



    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return; // 최소화 시 무시
        }
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
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
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
            primitive: wgpu::PrimitiveState {
                topology,
                ..Default::default()
            },
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

    /// 렌더 패스에 단일 메시를 그립니다. 빈 메시는 건너뜁니다.
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
        self.view_proj = view_proj;
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&view_proj.to_cols_array()),
        );

        // 애니메이션 그래프가 없으면 즉시 반환합니다.
        if self.animated_graphs.is_empty() {
            return;
        }

        // 경과 시간 t를 한 번만 읽어 모든 애니메이션 그래프에 공유합니다.
        let t = self.start_time.elapsed().as_secs_f32();
        for ((anim, gpu), scratch) in self
            .animated_graphs
            .iter()
            .zip(self.animated_gpu.iter())
            .zip(self.anim_scratch.iter_mut())
        {
            // 스크래치 버퍼를 재사용해 매 프레임 Vec 할당을 피합니다.
            plot_wireframe_into(
                scratch,
                &anim.x_range,
                &anim.z_range,
                |x, z| (anim.func)(x, z, t),
                anim.base_color,
            );
            self.queue
                .write_buffer(&gpu.vertex_buf, 0, bytemuck::cast_slice(scratch));
        }
    }

    pub fn render(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) => t,
            wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            other => {
                eprintln!("surface texture error: {:?}", other);
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

        // 범례 TextArea: 우측 상단에 세로로 나열
        let padding = 16.0f32;
        let row_h   = 28.0f32;
        let text_x  = self.size.width as f32 - 220.0;

        let legend_areas: Vec<TextArea> = self
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

        // 축 눈금 TextArea: 뷰-프로젝션으로 3D 월드 좌표 → 2D 화면 좌표로 투영
        let w = self.size.width as f32;
        let h = self.size.height as f32;
        let axis_areas: Vec<TextArea> = self
            .axis_labels
            .iter()
            .filter_map(|(pos, buf)| {
                let clip = self.view_proj * Vec4::new(pos.x, pos.y, pos.z, 1.0);
                // 카메라 뒤쪽이거나 NDC 범위 밖이면 건너뜁니다.
                if clip.w <= 0.0 {
                    return None;
                }
                let ndc = clip / clip.w;
                if ndc.x < -1.1 || ndc.x > 1.1 || ndc.y < -1.1 || ndc.y > 1.1 {
                    return None;
                }
                // NDC → 픽셀 좌표 (Y축 반전)
                let sx = (ndc.x + 1.0) * 0.5 * w;
                let sy = (1.0 - ndc.y) * 0.5 * h;
                Some(TextArea {
                    buffer: buf,
                    left: sx,
                    top: sy,
                    scale: 1.0,
                    bounds: TextBounds::default(),
                    default_color: GlyphColor::rgb(140, 140, 155),
                    custom_glyphs: &[],
                })
            })
            .collect();

        let text_areas: Vec<TextArea> = legend_areas
            .into_iter()
            .chain(axis_areas)
            .collect();

        if !text_areas.is_empty() {
            self.text_renderer
                .prepare(
                    &self.device,
                    &self.queue,
                    &mut self.font_system,
                    &mut self.text_atlas,
                    &self.viewport,
                    text_areas.clone(),
                    &mut self.swash_cache,
                )?;
        }

        // ── 렌더 커맨드 ──────────────────────────────────────────────────────
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        {
            // f32 → f64 변환: wgpu Color 구조체는 f64를 사용합니다.
            let [r, g, b, a] = self.background_color.map(f64::from);
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
                        // 깊이 버퍼는 프레임 간에 공유되지 않으므로 저장 불필요
                        store: wgpu::StoreOp::Discard,
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

            // 애니메이션 메시
            for gpu in &self.animated_gpu {
                self.draw_mesh(&mut rp, gpu);
            }

            // 산점도
            if self.scatter.index_count > 0 {
                rp.set_pipeline(&self.point_pipeline);
                self.draw_mesh(&mut rp, &self.scatter);
            }

            // 텍스트 오버레이 (범례 + 축 눈금) — 같은 렌더패스 내
            if !text_areas.is_empty() {
                self.text_renderer
                    .render(&self.text_atlas, &self.viewport, &mut rp)?;
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        surface_texture.present();

        // 매 프레임 atlas LRU 정리
        self.text_atlas.trim();

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// 애니메이션 프레임 헬퍼 — 기존 Vec<Vertex>를 재사용해 할당을 피합니다.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 눈금 값 포매터
// ---------------------------------------------------------------------------

/// 격자 눈금 값을 간결한 문자열로 변환합니다.
///
/// - 정수면 소수점 없이 출력 (`6` → `"6"`)
/// - 소수점이 필요하면 한 자리까지 출력 (`6.5` → `"6.5"`)
fn format_tick(v: f32) -> String {
    if (v - v.round()).abs() < 1e-4 {
        format!("{}", v as i32)
    } else {
        format!("{:.1}", v)
    }
}
/// `out`의 길이는 `x_range.len() * z_range.len()`과 일치해야 합니다.
fn plot_wireframe_into(
    out: &mut Vec<Vertex>,
    x_range: &[f32],
    z_range: &[f32],
    y_func: impl Fn(f32, f32) -> f32,
    base_color: [f32; 3],
) {
    let rows = z_range.len();
    let cols = x_range.len();
    let expected = rows * cols;

    out.clear();
    out.reserve(expected);

    let (mut y_min, mut y_max) = (f32::MAX, f32::MIN);

    for &z in z_range {
        for &x in x_range {
            let y = y_func(x, z);
            if y < y_min { y_min = y; }
            if y > y_max { y_max = y; }
            out.push(Vertex {
                position: [x, y, z, 1.0],
                color: [0.0; 4],
            });
        }
    }

    let denom = (y_max - y_min).max(f32::EPSILON);
    let [cr, cg, cb] = base_color;
    for v in out.iter_mut() {
        let t = 0.4 + 0.6 * (v.position[1] - y_min) / denom;
        v.color = [cr * t, cg * t, cb * t, 1.0];
    }
}