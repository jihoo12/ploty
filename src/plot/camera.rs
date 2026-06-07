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
///
/// `yaw`, `pitch`, `radius`는 불변식(clamp 범위)을 보호하기 위해 비공개입니다.
/// 직접 읽으려면 접근자 메서드를 사용하고, 직접 설정하려면 `set_*` 메서드를 사용하세요.
pub struct Camera {
    yaw: f32,
    pitch: f32,
    radius: f32,

    /// 카메라가 바라보는 월드 공간 기준점. 중간 버튼 드래그로 이동합니다.
    target: Vec3,

    is_dragging: bool,
    last_pos: Option<(f64, f64)>,

    is_panning: bool,
    last_pan_pos: Option<(f64, f64)>,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            yaw:   -45.0f32.to_radians(),
            pitch:  25.0f32.to_radians(),
            radius: 15.0,
            target: Vec3::ZERO,
            is_dragging: false,
            last_pos: None,
            is_panning: false,
            last_pan_pos: None,
        }
    }
}

impl Camera {
    pub fn new() -> Self {
        Self::default()
    }

    // ── 읽기 접근자 ──────────────────────────────────────────────────────────

    #[inline] pub fn yaw(&self)    -> f32 { self.yaw    }
    #[inline] pub fn pitch(&self)  -> f32 { self.pitch  }
    #[inline] pub fn radius(&self) -> f32 { self.radius }

    // ── 쓰기 접근자 (불변식 보호) ────────────────────────────────────────────

    pub fn set_yaw(&mut self, yaw: f32) {
        self.yaw = yaw;
    }

    pub fn set_pitch(&mut self, pitch: f32) {
        self.pitch = pitch.clamp(PITCH_MIN, PITCH_MAX);
    }

    pub fn set_radius(&mut self, radius: f32) {
        self.radius = radius.clamp(RADIUS_MIN, RADIUS_MAX);
    }

    // ── 이벤트 핸들러 ────────────────────────────────────────────────────────

    pub fn on_mouse_button(&mut self, pressed: bool) {
        self.is_dragging = pressed;
        if !pressed {
            self.last_pos = None;
        }
    }

    /// 중간 마우스 버튼 상태를 갱신합니다.
    pub fn on_middle_mouse_button(&mut self, pressed: bool) {
        self.is_panning = pressed;
        if !pressed {
            self.last_pan_pos = None;
        }
    }

    pub fn on_cursor_moved(&mut self, x: f64, y: f64) {
        // 좌클릭 궤도 회전
        if self.is_dragging {
            if let Some((lx, ly)) = self.last_pos {
                self.yaw   += (x - lx) as f32 * 0.005;
                self.set_pitch(self.pitch + (y - ly) as f32 * 0.005);
            }
            self.last_pos = Some((x, y));
        }

        // 중간 버튼 팬
        if self.is_panning {
            if let Some((lx, ly)) = self.last_pan_pos {
                let dx = (x - lx) as f32;
                let dy = (y - ly) as f32;

                // 카메라 오른쪽/위 벡터를 월드 공간에서 계산합니다.
                // 팬 속도를 반지름에 비례시켜 줌 레벨에 무관하게 일정하게 만듭니다.
                let pan_speed = self.radius * 0.001;

                let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
                let (sin_pitch, cos_pitch) = self.pitch.sin_cos();

                // 카메라 앞 방향 (eye → target)
                let forward = Vec3::new(
                    -cos_pitch * cos_yaw,
                    -sin_pitch,
                    -cos_pitch * sin_yaw,
                );
                // 오른쪽 = forward × 월드 위 (Y축)
                let right = forward.cross(Vec3::Y).normalize();
                // 실제 위 = right × forward
                let up = right.cross(forward).normalize();

                // 화면 X 드래그 → 반대 방향으로 target 이동 (뷰포트 좌표 반전)
                self.target -= right * dx * pan_speed;
                self.target += up    * dy * pan_speed;
            }
            self.last_pan_pos = Some((x, y));
        }

        // 두 동작 모두 비활성화 상태일 때도 커서 위치를 갱신합니다.
        if !self.is_dragging {
            self.last_pos = None;
        }
        if !self.is_panning {
            self.last_pan_pos = None;
        }
    }

    pub fn on_scroll(&mut self, dy: f32) {
        self.set_radius(self.radius - dy);
    }

    pub fn view_proj_matrix(&self, aspect: f32) -> Mat4 {
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        let (sin_yaw,   cos_yaw)   = self.yaw.sin_cos();
        let eye = self.target + Vec3::new(
            self.radius * cos_pitch * cos_yaw,
            self.radius * sin_pitch,
            self.radius * cos_pitch * sin_yaw,
        );
        let proj = Mat4::perspective_rh(std::f32::consts::PI / 4.0, aspect, 0.1, 100.0);
        let view = Mat4::look_at_rh(eye, self.target, Vec3::Y);
        OPENGL_TO_WGPU * proj * view
    }
}