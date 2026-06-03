use super::{config::PlotConfig, mesh::Mesh};

/// `t`(초)를 받아 y값을 반환하는 애니메이션 함수 타입.
pub type AnimFn = Box<dyn Fn(f32, f32, f32) -> f32 + Send + 'static>;

/// CPU 측 애니메이션 그래프 정의.
/// 매 프레임 `func(x, z, t)` 를 호출해 메시를 재생성합니다.
pub struct AnimatedGraph {
    pub x_range: Vec<f32>,
    pub z_range: Vec<f32>,
    pub func: AnimFn,
    pub base_color: [f32; 3],
}

/// 렌더링할 데이터와 설정을 빌더 패턴으로 모읍니다.
#[derive(Default)]
pub struct PlotData {
    pub graphs: Vec<Mesh>,
    pub scatters: Vec<Mesh>,
    pub animated_graphs: Vec<AnimatedGraph>,
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
        x_range: Vec<f32>,
        z_range: Vec<f32>,
        func: impl Fn(f32, f32, f32) -> f32 + Send + 'static,
        base_color: [f32; 3],
    ) -> Self {
        self.animated_graphs.push(AnimatedGraph {
            x_range,
            z_range,
            func: Box::new(func),
            base_color,
        });
        self
    }
}