mod plot;

use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

struct State {
    window: Arc<Window>,
    app: plot::App<'static>,
}

struct Handler {
    plot_data: Option<plot::PlotData>,
    state: Option<State>,
}

impl ApplicationHandler for Handler {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("ploty")
                        .with_inner_size(winit::dpi::PhysicalSize::new(1000u32, 800u32)),
                )
                .unwrap(),
        );
        let app = pollster::block_on(plot::App::new(
            window.clone(),
            self.plot_data.take().unwrap(),
        ));
        self.state = Some(State { window, app });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        let app = &mut state.app;

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => app.resize(size),

            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                app.camera.on_mouse_button(state == ElementState::Pressed);
            }
            WindowEvent::CursorMoved { position, .. } => {
                app.camera.on_cursor_moved(position.x, position.y);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let dy = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.01,
                };
                app.camera.on_scroll(dy);
            }

            WindowEvent::RedrawRequested => {
                app.update();
                let _ = app.render();
                state.window.request_redraw();
            }
            _ => {}
        }
    }
}

fn main() {
    let n = 60;
    let range: Vec<f32> = (0..n)
        .map(|i| -5.0 + (i as f32 / (n - 1) as f32) * 10.0)
        .collect();

    let config = plot::PlotConfig {
        grid_size: 12.0,
        grid_divisions: 12,
        // 각 그래프에 대응하는 범례 항목
        ..Default::default()
    };

    let plot_data = plot::PlotData::new()
        .with_config(config)
        // 정적 그래프: sin(r)
        .add_animated_graph(
            range.clone(),
            range.clone(),
            |x, z, t| {
                let r = (x * x + z * z).sqrt();
                
                // 1. 가우시안 패킷의 중심 폭을 결정 (값이 커질수록 넓게 퍼짐)
                let width = 2.0; 
                let gaussian = (- (r * r) / (2.0 * width * width)).exp();
                
                // 2. 내부에서 진동하는 파동 (t를 빼주어 중심에서 바깥으로 진행)
                // 4.0은 파동의 주파수(촘촘함), 5.0은 파동이 퍼지는 속도입니다.
                let wave = (4.0 * r - t * 5.0).cos();
                
                // 가우시안 엔벨로프와 파동을 곱해줍니다.
                gaussian * wave
            },
            [0.1, 0.8, 0.4], // 청록색/녹색 계열 레이블 컬러 예시
        );
    let event_loop = EventLoop::new().unwrap();
    let mut handler = Handler {
        plot_data: Some(plot_data),
        state: None,
    };
    event_loop.run_app(&mut handler).unwrap();
}
