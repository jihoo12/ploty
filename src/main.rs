use std::f32::consts::PI;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};
use glam::{Vec3, Mat4};

// ---------------------------------------------------------
// 1. WebGPU 깊이 보정 및 데이터 구조
// ---------------------------------------------------------
#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: Mat4 = Mat4::from_cols_array(&[
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
]);

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 4], // 16바이트 정렬 (x, y, z, 1.0)
    color: [f32; 4],    // 16바이트 정렬 (r, g, b, a)
}

impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { offset: 0, shader_location: 0, format: wgpu::VertexFormat::Float32x4 },
                wgpu::VertexAttribute { offset: 16, shader_location: 1, format: wgpu::VertexFormat::Float32x4 },
            ],
        }
    }
}

// ---------------------------------------------------------
// 2. 최적화된 데이터 생성 및 통합 함수
// ---------------------------------------------------------

/// 여러 그래프 데이터를 하나의 거대한 버퍼 데이터로 병합합니다. (핵심 최적화)
fn merge_graphs_data(graphs: Vec<(Vec<Vertex>, Vec<u32>)>) -> (Vec<Vertex>, Vec<u32>) {
    let total_v: usize = graphs.iter().map(|g| g.0.len()).sum();
    let total_i: usize = graphs.iter().map(|g| g.1.len()).sum();

    let mut merged_vertices = Vec::with_capacity(total_v);
    let mut merged_indices = Vec::with_capacity(total_i);

    for (mut vertices, indices) in graphs {
        let vertex_offset = merged_vertices.len() as u32; // 현재까지 쌓인 정점 수가 오프셋이 됨
        merged_vertices.append(&mut vertices);
        for idx in indices {
            merged_indices.push(idx + vertex_offset); // 인덱스 보정
        }
    }
    (merged_vertices, merged_indices)
}

fn create_full_grid_data(size: f32, divisions: usize) -> (Vec<Vertex>, Vec<u32>) {
    let step = size / divisions as f32;
    let start = -size / 2.0;
    let mut vertices = Vec::with_capacity((divisions + 1) * 12);
    let mut indices = Vec::with_capacity((divisions + 1) * 12);
    let mut curr_idx = 0;

    let mut add_line = |p1: [f32; 3], p2: [f32; 3], color: [f32; 3]| {
        vertices.push(Vertex { position: [p1[0], p1[1], p1[2], 1.0], color: [color[0], color[1], color[2], 1.0] });
        vertices.push(Vertex { position: [p2[0], p2[1], p2[2], 1.0], color: [color[0], color[1], color[2], 1.0] });
        indices.extend_from_slice(&[curr_idx, curr_idx + 1]);
        curr_idx += 2;
    };

    let c = [0.2, 0.2, 0.2];
    for i in 0..=divisions {
        let d = start + i as f32 * step;
        let end = -start;
        add_line([d, start, start], [d, start, end], c);
        add_line([start, start, d], [end, start, d], c);
        add_line([d, start, start], [d, end, start], c);
        add_line([start, d, start], [end, d, start], c);
        add_line([start, d, start], [start, d, end], c);
        add_line([start, start, d], [start, end, d], c);
    }
    (vertices, indices)
}

fn plot_wireframe(x_range: &[f32], z_range: &[f32], y_func: impl Fn(f32, f32) -> f32, base_color: [f32; 3]) -> (Vec<Vertex>, Vec<u32>) {
    let rows = z_range.len();
    let cols = x_range.len();
    let mut vertices = Vec::with_capacity(rows * cols);
    let (mut y_min, mut y_max) = (f32::MAX, f32::MIN);

    for &z in z_range {
        for &x in x_range {
            let y = y_func(x, z);
            y_min = y_min.min(y); y_max = y_max.max(y);
            vertices.push(Vertex { position: [x, y, z, 1.0], color: [0.0, 0.0, 0.0, 1.0] });
        }
    }

    let denom = if y_max != y_min { y_max - y_min } else { 1.0 };
    for v in &mut vertices {
        let norm = (v.position[1] - y_min) / denom;
        let it = 0.5 + 0.5 * norm;
        v.color = [base_color[0] * it, base_color[1] * it, base_color[2] * it, 1.0];
    }

    let mut indices = Vec::with_capacity(rows * cols * 4);
    for r in 0..rows {
        for c in 0..(cols - 1) {
            indices.extend_from_slice(&[(r * cols + c) as u32, (r * cols + c + 1) as u32]);
        }
    }
    for r in 0..(rows - 1) {
        for c in 0..cols {
            indices.extend_from_slice(&[(r * cols + c) as u32, ((r + 1) * cols + c) as u32]);
        }
    }
    (vertices, indices)
}

// ---------------------------------------------------------
// 3. App 구조체 및 렌더링 엔진
// ---------------------------------------------------------
struct App<'a> {
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    pipeline: wgpu::RenderPipeline,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    depth_view: wgpu::TextureView,
    
    // 통합된 리소스
    grid_res: (wgpu::Buffer, wgpu::Buffer, u32),
    graph_res: (wgpu::Buffer, wgpu::Buffer, u32),
    
    yaw: f32, pitch: f32, radius: f32,
    is_dragging: bool, last_pos: Option<(f64, f64)>,
}

impl<'a> App<'a> {
    async fn new(window: Arc<Window>, graphs: Vec<(Vec<Vertex>, Vec<u32>)>) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            ..Default::default()
        }).await.unwrap();

        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default()).await.unwrap();
        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats[0];
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format, width: size.width, height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(r#"
                struct Camera { view_proj: mat4x4<f32> };
                @group(0) @binding(0) var<uniform> camera: Camera;
                struct Vin { @location(0) pos: vec4<f32>, @location(1) col: vec4<f32> };
                struct Vout { @builtin(position) pos: vec4<f32>, @location(0) col: vec4<f32> };
                @vertex fn vs_main(in: Vin) -> Vout {
                    return Vout(camera.view_proj * in.pos, in.col);
                }
                @fragment fn fs_main(in: Vout) -> @location(0) vec4<f32> {
                    return in.col;
                }
            "#)),
        });

        // 그래프 통합 처리
        let (merged_v, merged_i) = merge_graphs_data(graphs);
        let graph_v_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&merged_v), usage: wgpu::BufferUsages::VERTEX });
        let graph_i_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&merged_i), usage: wgpu::BufferUsages::INDEX });

        let (gv, gi) = create_full_grid_data(10.0, 10);
        let gv_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&gv), usage: wgpu::BufferUsages::VERTEX });
        let gi_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&gi), usage: wgpu::BufferUsages::INDEX });

        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }], label: None,
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor { layout: &bgl, entries: &[wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() }], label: None });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None, layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bgl], immediate_size: 0 })),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"), buffers: &[Vertex::desc()], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_main"), targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL })], compilation_options: Default::default() }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::LineList, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState { format: wgpu::TextureFormat::Depth32Float, depth_write_enabled: true, depth_compare: wgpu::CompareFunction::Less, stencil: Default::default(), bias: Default::default() }),
            multisample: wgpu::MultisampleState::default(), multiview_mask: None, cache: None,
        });

        let depth_view = Self::create_depth_view(&device, size.width, size.height);

        Self {
            surface, device, queue, config, size, pipeline, camera_buffer, camera_bind_group, depth_view,
            grid_res: (gv_buf, gi_buf, gi.len() as u32),
            graph_res: (graph_v_buf, graph_i_buf, merged_i.len() as u32),
            yaw: -45.0f32.to_radians(), pitch: 25.0f32.to_radians(), radius: 15.0,
            is_dragging: false, last_pos: None,
        }
    }

    fn create_depth_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
        device.create_texture(&wgpu::TextureDescriptor {
            label: None, size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float, usage: wgpu::TextureUsages::RENDER_ATTACHMENT, view_formats: &[],
        }).create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn update(&mut self) {
        let aspect = self.size.width as f32 / self.size.height as f32;
        let proj = Mat4::perspective_rh(PI / 4.0, aspect, 0.1, 100.0);
        let eye = Vec3::new(self.radius * self.pitch.cos() * self.yaw.cos(), self.radius * self.pitch.sin(), self.radius * self.pitch.cos() * self.yaw.sin());
        let view_proj = OPENGL_TO_WGPU_MATRIX * proj * Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y);
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&view_proj.to_cols_array()));
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &view, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.01, g: 0.01, b: 0.02, a: 1.0 }), store: wgpu::StoreOp::Store }, depth_slice: None })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &self.depth_view, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None }),
                ..Default::default()
            });
            rp.set_pipeline(&self.pipeline);
            rp.set_bind_group(0, &self.camera_bind_group, &[]);
            
            // 1. 그리드 그리기 (1 Draw Call)
            rp.set_vertex_buffer(0, self.grid_res.0.slice(..));
            rp.set_index_buffer(self.grid_res.1.slice(..), wgpu::IndexFormat::Uint32);
            rp.draw_indexed(0..self.grid_res.2, 0, 0..1);
            
            // 2. 통합된 모든 그래프 한 번에 그리기 (1 Draw Call!)
            rp.set_vertex_buffer(0, self.graph_res.0.slice(..));
            rp.set_index_buffer(self.graph_res.1.slice(..), wgpu::IndexFormat::Uint32);
            rp.draw_indexed(0..self.graph_res.2, 0, 0..1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }
}

fn main() {
    let n = 80;
    let mut range = Vec::with_capacity(n);
    for i in 0..n { range.push(-5.0 + (i as f32 / (n - 1) as f32) * 10.0); }

    let mut graphs = Vec::new();
    // 여러 개의 복잡한 그래프 데이터 생성
    graphs.push(plot_wireframe(&range, &range, |x, z| (x*x + z*z).sqrt().sin(), [0.2, 0.5, 1.0]));
    graphs.push(plot_wireframe(&range, &range, |x, z| (x.cos() * z.sin()) * 2.0, [1.0, 0.3, 0.3]));
    graphs.push(plot_wireframe(&range, &range, |x, z| (x*0.5).exp().cos() + z.sin(), [0.3, 1.0, 0.3]));

    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(WindowBuilder::new().with_title("WGPU Batched Rendering").with_inner_size(winit::dpi::PhysicalSize::new(1000, 800)).build(&event_loop).unwrap());
    let mut app = pollster::block_on(App::new(window.clone(), graphs));

    event_loop.run(move |event, elwt| {
        match event {
            Event::WindowEvent { ref event, .. } => match event {
                WindowEvent::CloseRequested => elwt.exit(),
                WindowEvent::Resized(s) => {
                    app.size = *s; app.config.width = s.width; app.config.height = s.height;
                    app.surface.configure(&app.device, &app.config);
                    app.depth_view = App::create_depth_view(&app.device, s.width, s.height);
                }
                WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                    app.is_dragging = *state == ElementState::Pressed;
                    if !app.is_dragging { app.last_pos = None; }
                }
                WindowEvent::CursorMoved { position, .. } => {
                    if app.is_dragging {
                        if let Some((lx, ly)) = app.last_pos {
                            app.yaw += (position.x - lx) as f32 * 0.005;
                            app.pitch = (app.pitch + (position.y - ly) as f32 * 0.005).clamp(-1.5, 1.5);
                        }
                    }
                    app.last_pos = Some((position.x, position.y));
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    let dy = match delta { MouseScrollDelta::LineDelta(_, y) => *y, MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.01 };
                    app.radius = (app.radius - dy).clamp(2.0, 50.0);
                }
                WindowEvent::RedrawRequested => { app.update(); let _ = app.render(); }
                _ => {}
            }
            Event::AboutToWait => window.request_redraw(),
            _ => {}
        }
    }).unwrap();
}