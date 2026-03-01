mod plot;

use std::sync::Arc;
use winit::{event::*, event_loop::EventLoop, window::WindowBuilder};

fn main() {
    // 1. 빌더 객체 생성
    let mut plot_data = plot::PlotData::new();

    // 2. 원하는 데이터만 선택적으로 추가 (예: 와이어프레임만)
    let n = 60;
    let mut range = Vec::new();
    for i in 0..n { range.push(-5.0 + (i as f32 / (n - 1) as f32) * 10.0); }

    plot_data = plot_data.add_graph(plot::plot_wireframe(&range, &range, |x, z| (x*x + z*z).sqrt().sin(), [0.2, 0.5, 1.0]));

    // 만약 산점도가 필요하다면 아래처럼 추가 (필요 없으면 주석 처리 가능)
    // plot_data = plot_data.add_scatter(plot::plot_scatter(&[(0.0, 0.0, 0.0), (1.0, 2.0, 1.0)], [1.0, 1.0, 0.0]));

    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(WindowBuilder::new()
        .with_title("ploty - Builder Pattern")
        .with_inner_size(winit::dpi::PhysicalSize::new(1000, 800))
        .build(&event_loop).unwrap());

    // 3. 통합된 PlotData를 전달하여 App 생성
    let mut app = pollster::block_on(plot::App::new(window.clone(), plot_data));

    event_loop.run(move |event, elwt| {
        match event {
            Event::WindowEvent { ref event, .. } => match event {
                WindowEvent::CloseRequested => elwt.exit(),
                WindowEvent::Resized(s) => {
                    app.size = *s;
                    app.config.width = s.width;
                    app.config.height = s.height;
                    app.surface.configure(&app.device, &app.config);
                    app.depth_view = plot::App::create_depth_view(&app.device, s.width, s.height);
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
                    let dy = match delta {
                        MouseScrollDelta::LineDelta(_, y) => *y,
                        MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.01,
                    };
                    app.radius = (app.radius - dy).clamp(2.0, 50.0);
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