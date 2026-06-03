use super::{config::PlotConfig, mesh::Mesh};

/// 렌더링할 데이터와 설정을 빌더 패턴으로 모읍니다.
#[derive(Default)]
pub struct PlotData {
    pub graphs: Vec<Mesh>,
    pub scatters: Vec<Mesh>,
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

    pub fn add_graph(mut self, mesh: Mesh) -> Self {
        self.graphs.push(mesh);
        self
    }

    pub fn add_scatter(mut self, mesh: Mesh) -> Self {
        self.scatters.push(mesh);
        self
    }
}