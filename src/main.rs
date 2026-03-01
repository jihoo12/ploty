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
// 1. WebGPU 깊이 보정 행렬
// ---------------------------------------------------------
#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: Mat4 = Mat4::from_cols_array(&[
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
]);

// ---------------------------------------------------------
// 2. 데이터 구조
// ---------------------------------------------------------
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { offset: 0, shader_location: 0, format: wgpu::VertexFormat::Float32x3 },
                wgpu::VertexAttribute { offset: 12, shader_location: 1, format: wgpu::VertexFormat::Float32x3 },
            ],
        }
    }
}

// ---------------------------------------------------------
// 3. 3면 그리드 생성 함수 (핵심 수정 부분)
// ---------------------------------------------------------
fn create_full_grid_data(size: f32, divisions: usize) -> (Vec<Vertex>, Vec<u16>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let step = size / divisions as f32;
    let start = -size / 2.0;
    let end = size / 2.0;
    let mut curr_idx = 0;
    
    let mut add_line = |p1: [f32; 3], p2: [f32; 3], color: [f32; 3]| {
        vertices.push(Vertex { position: p1, color });
        vertices.push(Vertex { position: p2, color });
        indices.push(curr_idx);
        indices.push(curr_idx + 1);
        curr_idx += 2;
    };

    // 각 평면별로 다른 색상을 주어 구분감을 높임 (선택 사항)
    let color_xz = [0.2, 0.2, 0.2]; // 바닥 (어두운 회색)
    let color_xy = [0.15, 0.2, 0.15]; // 뒷벽 (약간 녹색빛)
    let color_yz = [0.2, 0.15, 0.15]; // 옆벽 (약간 붉은빛)

    for i in 0..=divisions {
        let d = start + (i as f32) * step;

        // 1. 바닥 그리드 (XZ 평면, y = start)
        add_line([d, start, start], [d, start, end], color_xz);
        add_line([start, start, d], [end, start, d], color_xz);

        // 2. 뒷벽 그리드 (XY 평면, z = start)
        add_line([d, start, start], [d, end, start], color_xy);
        add_line([start, d, start], [end, d, start], color_xy);

        // 3. 옆벽 그리드 (YZ 평면, x = start)
        add_line([start, d, start], [start, d, end], color_yz);
        add_line([start, start, d], [start, end, d], color_yz);
    }

    (vertices, indices)
}

// ---------------------------------------------------------
// 4. 그래프 데이터 생성 (기존 유지)
// ---------------------------------------------------------
fn plot_wireframe(
    x_range: &[f32],
    z_range: &[f32],
    y_func: impl Fn(f32, f32) -> f32,
    base_color: [f32; 3],
) -> (Vec<Vertex>, Vec<u32>) {
    let rows = z_range.len();
    let cols = x_range.len();
    let mut vertices = Vec::new();
    let mut y_vals = Vec::new();
    let (mut y_min, mut y_max) = (f32::MAX, f32::MIN);

    for &z in z_range {
        for &x in x_range {
            let y = y_func(x, z);
            y_vals.push(y);
            if y < y_min { y_min = y; }
            if y > y_max { y_max = y; }
        }
    }

    let denom = if y_max != y_min { y_max - y_min } else { 1.0 };

    for (i, &z) in z_range.iter().enumerate() {
        for (j, &x) in x_range.iter().enumerate() {
            let y = y_vals[i * cols + j];
            let y_norm = (y - y_min) / denom;
            let color = [
                base_color[0] * (0.6 + 0.4 * y_norm),
                base_color[1] * (0.6 + 0.4 * y_norm),
                base_color[2] * (0.6 + 0.4 * y_norm),
            ];
            vertices.push(Vertex { position: [x, y, z], color });
        }
    }

    let mut indices = Vec::new();
    for r in 0..rows {
        for c in 0..(cols - 1) {
            indices.push((r * cols + c) as u32);
            indices.push((r * cols + c + 1) as u32);
        }
    }
    for r in 0..(rows - 1) {
        for c in 0..cols {
            indices.push((r * cols + c) as u32);
            indices.push(((r + 1) * cols + c) as u32);
        }
    }
    (vertices, indices)
}

// ---------------------------------------------------------
// 5. App 구조체 및 렌더링 로직 (기존 유지)
// ---------------------------------------------------------
struct GraphResource {
    v_buf: wgpu::Buffer,
    i_buf: wgpu::Buffer,
    index_count: u32,
}

struct App<'a> {
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    pipeline: wgpu::RenderPipeline,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    depth_texture_view: wgpu::TextureView,
    grid_resource: (wgpu::Buffer, wgpu::Buffer, u32),
    graph_resources: Vec<GraphResource>,
    yaw: f32, pitch: f32, camera_radius: f32,
    is_dragging: bool, last_mouse_pos: Option<(f64, f64)>,
}

impl<'a> App<'a> {
    async fn new(window: Arc<Window>, graphs_data: Vec<(Vec<Vertex>, Vec<u32>)>) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }).await.unwrap();

        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default()).await.unwrap();
        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().find(|f| f.is_srgb()).copied().unwrap_or(caps.formats[0]);
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
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(r#"
                struct Camera { view_proj: mat4x4<f32> };
                @group(0) @binding(0) var<uniform> camera: Camera;
                struct VertexInput { @location(0) pos: vec3<f32>, @location(1) col: vec3<f32> };
                struct VertexOutput { @builtin(position) clip_pos: vec4<f32>, @location(0) col: vec3<f32> };
                @vertex fn vs_main(in: VertexInput) -> VertexOutput {
                    var out: VertexOutput;
                    out.clip_pos = camera.view_proj * vec4<f32>(in.pos, 1.0);
                    out.col = in.col;
                    return out;
                }
                @fragment fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
                    return vec4<f32>(in.col, 1.0);
                }
            "#)),
        });

        let mut graph_resources = Vec::new();
        for (v, i) in graphs_data {
            let v_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&v), usage: wgpu::BufferUsages::VERTEX });
            let i_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&i), usage: wgpu::BufferUsages::INDEX });
            graph_resources.push(GraphResource { v_buf, i_buf, index_count: i.len() as u32 });
        }

        let (gv, gi) = create_full_grid_data(10.0, 10);
        let gv_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&gv), usage: wgpu::BufferUsages::VERTEX });
        let gi_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&gi), usage: wgpu::BufferUsages::INDEX });

        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: 64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None,
            }], label: None,
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout, entries: &[wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() }], label: None,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&bind_group_layout], immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None, layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"), buffers: &[Vertex::desc()], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_main"), targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::REPLACE), write_mask: wgpu::ColorWrites::ALL })], compilation_options: Default::default() }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::LineList, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState { format: wgpu::TextureFormat::Depth32Float, depth_write_enabled: true, depth_compare: wgpu::CompareFunction::Less, stencil: Default::default(), bias: Default::default() }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None, cache: None,
        });

        let depth_texture_view = Self::create_depth_view(&device, size.width, size.height);

        Self {
            surface, device, queue, config, size, pipeline, camera_buffer, camera_bind_group, depth_texture_view,
            grid_resource: (gv_buf, gi_buf, gi.len() as u32), graph_resources,
            yaw: -45.0f32.to_radians(), pitch: 20.0f32.to_radians(), camera_radius: 18.0,
            is_dragging: false, last_mouse_pos: None,
        }
    }

    fn create_depth_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
        device.create_texture(&wgpu::TextureDescriptor {
            label: None, size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float, usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        }).create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture_view = Self::create_depth_view(&self.device, new_size.width, new_size.height);
        }
    }

    fn update(&mut self) {
        let aspect = self.size.width as f32 / self.size.height as f32;
        let proj = Mat4::perspective_rh(PI / 4.0, aspect, 0.1, 100.0);
        let eye = Vec3::new(
            self.camera_radius * self.pitch.cos() * self.yaw.cos(),
            self.camera_radius * self.pitch.sin(),
            self.camera_radius * self.pitch.cos() * self.yaw.sin(),
        );
        let view = Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y);
        let view_proj = OPENGL_TO_WGPU_MATRIX * proj * view;
        self.queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&view_proj.to_cols_array()));
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view, resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.01, g: 0.01, b: 0.03, a: 1.0 }), store: wgpu::StoreOp::Store },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            rp.set_pipeline(&self.pipeline);
            rp.set_bind_group(0, &self.camera_bind_group, &[]);
            rp.set_vertex_buffer(0, self.grid_resource.0.slice(..));
            rp.set_index_buffer(self.grid_resource.1.slice(..), wgpu::IndexFormat::Uint16);
            rp.draw_indexed(0..self.grid_resource.2, 0, 0..1);
            for graph in &self.graph_resources {
                rp.set_vertex_buffer(0, graph.v_buf.slice(..));
                rp.set_index_buffer(graph.i_buf.slice(..), wgpu::IndexFormat::Uint32);
                rp.draw_indexed(0..graph.index_count, 0, 0..1);
            }
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }
}

fn main() {
    let n = 60;
    let mut x_range = Vec::new();
    let mut z_range = Vec::new();
    for i in 0..n {
        let t = i as f32 / (n - 1) as f32;
        x_range.push(-5.0 + t * 10.0);
        z_range.push(-5.0 + t * 10.0);
    }
    let mut graphs = Vec::new();
    graphs.push(plot_wireframe(&x_range, &z_range, |x, z| (x*x + z*z).sqrt().sin(), [0.2, 0.6, 1.0]));
    graphs.push(plot_wireframe(&x_range, &z_range, |x, z| ((x*x + z*z).sqrt() + 2.0).cos() * 0.5, [1.0, 0.4, 0.4]));

    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(WindowBuilder::new().with_title("WGPU 3D Plot with Full Grid").with_inner_size(winit::dpi::PhysicalSize::new(1200, 900)).build(&event_loop).unwrap());
    let mut app = pollster::block_on(App::new(window.clone(), graphs));

    event_loop.run(move |event, elwt| {
        match event {
            Event::WindowEvent { ref event, window_id } if window_id == window.id() => match event {
                WindowEvent::CloseRequested => elwt.exit(),
                WindowEvent::Resized(s) => app.resize(*s),
                WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                    app.is_dragging = *state == ElementState::Pressed;
                    if !app.is_dragging { app.last_mouse_pos = None; }
                }
                WindowEvent::CursorMoved { position, .. } => {
                    if app.is_dragging {
                        if let Some((lx, ly)) = app.last_mouse_pos {
                            app.yaw += (position.x - lx) as f32 * 0.005;
                            app.pitch = (app.pitch + (position.y - ly) as f32 * 0.005).clamp(-1.5, 1.5);
                        }
                    }
                    app.last_mouse_pos = Some((position.x, position.y));
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    let dy = match delta { MouseScrollDelta::LineDelta(_, y) => *y, MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.01 };
                    app.camera_radius = (app.camera_radius - dy).clamp(2.0, 50.0);
                }
                WindowEvent::RedrawRequested => {
                    app.update();
                    let _ = app.render();
                }
                _ => {}
            },
            Event::AboutToWait => window.request_redraw(),
            _ => {}
        }
    }).unwrap();
}