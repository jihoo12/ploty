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
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Middle,
                ..
            } => {
                app.camera.on_middle_mouse_button(state == ElementState::Pressed);
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

    // 나선 곡선: 400 샘플, u ∈ [0, 6π]
    let helix_samples: Vec<f32> = (0..=400)
        .map(|i| i as f32 / 400.0 * 6.0 * std::f32::consts::PI)
        .collect();

    // 매듭 곡선 (trefoil knot): 600 샘플, u ∈ [0, 2π]
    let knot_samples: Vec<f32> = (0..=600)
        .map(|i| i as f32 / 600.0 * 2.0 * std::f32::consts::PI)
        .collect();

    let config = plot::PlotConfig {
        grid_size: 12.0,
        grid_divisions: 12,
        show_axis_labels: false,
        ..Default::default()
    };

    let plot_data = plot::PlotData::new()
        .with_config(config)
        // 기존 애니메이션 물결 서피스
        .add_animated_graph(
            range.clone(),
            range.clone(),
            |x, z, t| {
                let r = (x * x + z * z).sqrt();
                let width = 2.0;
                let gaussian = (-(r * r) / (2.0 * width * width)).exp();
                let wave = (4.0 * r - t * 5.0).cos();
                gaussian * wave
            },
            [0.1, 0.8, 0.4],
        )
        // 정적 나선 곡선
        .add_parametric_curve(
            helix_samples,
            |u| {
                let r = 3.0;
                [r * u.cos(), u * 0.3 - 3.0, r * u.sin()]
            },
            [1.0, 0.4, 0.1],
        )
        // 애니메이션 트레포일 매듭: 크기와 위상이 시간에 따라 변함
        .add_animated_parametric_curve(
            knot_samples,
            |u, t| {
                let scale = 2.5 + 0.5 * (t * 0.8).sin();
                let x = scale * (u.sin() + 2.0 * (2.0 * u).sin());
                let y = scale * 0.4 * (t * 0.5 + u).cos();
                let z = scale * (u.cos() - 2.0 * (2.0 * u).cos());
                [x, y, z]
            },
            [0.3, 0.6, 1.0],
        );

    let event_loop = EventLoop::new().unwrap();
    let mut handler = Handler {
        plot_data: Some(plot_data),
        state: None,
    };
    event_loop.run_app(&mut handler).unwrap();
}