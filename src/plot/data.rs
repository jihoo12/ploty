use super::{config::PlotConfig, mesh::Mesh};

/// `t`(초)를 받아 y값을 반환하는 애니메이션 함수 타입.
pub type AnimFn = Box<dyn Fn(f32, f32, f32) -> f32 + Send + 'static>;

/// 매개변수 `u`를 받아 3D 점 `(x, y, z)`를 반환하는 파라메트릭 곡선 함수 타입.
pub type CurveFn = Box<dyn Fn(f32) -> [f32; 3] + Send + 'static>;

/// 시간 `t`도 받는 애니메이션 파라메트릭 곡선 함수 타입.
pub type AnimCurveFn = Box<dyn Fn(f32, f32) -> [f32; 3] + Send + 'static>;

/// CPU 측 정적 파라메트릭 곡선 정의.
///
/// `u_range`의 각 값에 대해 `func(u)` → `[x, y, z]`를 호출해 선분 메시를 생성합니다.
pub struct ParametricCurve {
    /// 매개변수 샘플 값 목록 (단조증가 권장)
    pub u_range: Box<[f32]>,
    pub func: CurveFn,
    pub color: [f32; 3],
}

/// CPU 측 애니메이션 파라메트릭 곡선 정의.
/// 매 프레임 `func(u, t)` → `[x, y, z]`를 호출합니다.
pub struct AnimatedParametricCurve {
    pub u_range: Box<[f32]>,
    pub func: AnimCurveFn,
    pub color: [f32; 3],
}

/// CPU 측 애니메이션 그래프 정의.
/// 매 프레임 `func(x, z, t)` 를 호출해 메시를 재생성합니다.
///
/// `x_range` / `z_range`는 읽기 전용이므로 `Box<[f32]>`로 저장합니다.
pub struct AnimatedGraph {
    pub x_range: Box<[f32]>,
    pub z_range: Box<[f32]>,
    pub func: AnimFn,
    pub base_color: [f32; 3],
}

/// 렌더링할 데이터와 설정을 빌더 패턴으로 모읍니다.
#[derive(Default)]
pub struct PlotData {
    pub graphs: Vec<Mesh>,
    pub scatters: Vec<Mesh>,
    pub animated_graphs: Vec<AnimatedGraph>,
    pub parametric_curves: Vec<ParametricCurve>,
    pub animated_parametric_curves: Vec<AnimatedParametricCurve>,
    pub config: PlotConfig,
}

impl PlotData {
    pub fn new() -> Self {
        Self::default()
    }

    /// 렌더러 설정을 교체합니다.
    pub fn with_config(mut self, config: PlotConfig) -> Self {
        self.config = config;
        self
    }

    /// 정적 와이어프레임 그래프를 추가합니다.
    pub fn add_graph(mut self, mesh: Mesh) -> Self {
        self.graphs.push(mesh);
        self
    }

    /// 산점도를 추가합니다.
    pub fn add_scatter(mut self, mesh: Mesh) -> Self {
        self.scatters.push(mesh);
        self
    }

    /// 시간에 따라 변하는 애니메이션 그래프를 추가합니다.
    ///
    /// `func(x, z, t) → y` 형태의 클로저를 전달합니다.
    /// `t` 는 앱 시작 후 경과 시간(초)입니다.
    pub fn add_animated_graph(
        mut self,
        x_range: impl Into<Box<[f32]>>,
        z_range: impl Into<Box<[f32]>>,
        func: impl Fn(f32, f32, f32) -> f32 + Send + 'static,
        base_color: [f32; 3],
    ) -> Self {
        self.animated_graphs.push(AnimatedGraph {
            x_range: x_range.into(),
            z_range: z_range.into(),
            func: Box::new(func),
            base_color,
        });
        self
    }

    /// 정적 3D 파라메트릭 곡선을 추가합니다.
    ///
    /// `func(u) → [x, y, z]` 형태의 클로저를 전달합니다.
    ///
    /// ```rust
    /// // 나선
    /// data.add_parametric_curve(
    ///     (0..=200).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
    ///     |u| [u.cos(), u * 0.1, u.sin()],
    ///     [1.0, 0.5, 0.0],
    /// )
    /// ```
    pub fn add_parametric_curve(
        mut self,
        u_range: impl Into<Box<[f32]>>,
        func: impl Fn(f32) -> [f32; 3] + Send + 'static,
        color: [f32; 3],
    ) -> Self {
        self.parametric_curves.push(ParametricCurve {
            u_range: u_range.into(),
            func: Box::new(func),
            color,
        });
        self
    }

    /// 시간에 따라 변하는 애니메이션 파라메트릭 곡선을 추가합니다.
    ///
    /// `func(u, t) → [x, y, z]` 형태의 클로저를 전달합니다.
    /// `t` 는 앱 시작 후 경과 시간(초)입니다.
    pub fn add_animated_parametric_curve(
        mut self,
        u_range: impl Into<Box<[f32]>>,
        func: impl Fn(f32, f32) -> [f32; 3] + Send + 'static,
        color: [f32; 3],
    ) -> Self {
        self.animated_parametric_curves.push(AnimatedParametricCurve {
            u_range: u_range.into(),
            func: Box::new(func),
            color,
        });
        self
    }
}