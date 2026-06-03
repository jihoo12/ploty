use glam::{Mat4, Vec3};
use std::f32::consts::PI;

/// OpenGL → WGPU 좌표계 변환 (NDC Z: [−1,1] → [0,1])
#[rustfmt::skip]
const OPENGL_TO_WGPU: Mat4 = Mat4::from_cols_array(&[
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
]);

const PITCH_MIN: f32 = -1.5;
const PITCH_MAX: f32 =  1.5;
const RADIUS_MIN: f32 =  2.0;
const RADIUS_MAX: f32 = 50.0;

/// 구면 좌표계 기반 궤도 카메라.
pub struct Camera {
    pub yaw: f32,
    pub pitch: f32,
    pub radius: f32,

    is_dragging: bool,
    last_pos: Option<(f64, f64)>,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            yaw:   -45.0f32.to_radians(),
            pitch:  25.0f32.to_radians(),
            radius: 15.0,
            is_dragging: false,
            last_pos: None,
        }
    }
}

impl Camera {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn on_mouse_button(&mut self, pressed: bool) {
        self.is_dragging = pressed;
        if !pressed {
            self.last_pos = None;
        }
    }

    pub fn on_cursor_moved(&mut self, x: f64, y: f64) {
        if self.is_dragging {
            if let Some((lx, ly)) = self.last_pos {
                self.yaw   += (x - lx) as f32 * 0.005;
                self.pitch  = (self.pitch + (y - ly) as f32 * 0.005)
                    .clamp(PITCH_MIN, PITCH_MAX);
            }
        }
        self.last_pos = Some((x, y));
    }

    pub fn on_scroll(&mut self, dy: f32) {
        self.radius = (self.radius - dy).clamp(RADIUS_MIN, RADIUS_MAX);
    }

    pub fn view_proj_matrix(&self, aspect: f32) -> Mat4 {
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        let (sin_yaw,   cos_yaw)   = self.yaw.sin_cos();
        let eye = Vec3::new(
            self.radius * cos_pitch * cos_yaw,
            self.radius * sin_pitch,
            self.radius * cos_pitch * sin_yaw,
        );
        let proj = Mat4::perspective_rh(PI / 4.0, aspect, 0.1, 100.0);
        let view = Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y);
        OPENGL_TO_WGPU * proj * view
    }
}