/// 범례 항목 하나: 레이블 + 색상.
#[derive(Debug, Clone)]
pub struct LegendEntry {
    pub label: String,
    pub color: [f32; 3],
}

/// 렌더러 외형 설정.
///
/// `PlotData::with_config()`로 주입하거나, 기본값(`Default`)을 그대로 사용합니다.
///
/// ```rust
/// let config = PlotConfig {
///     grid_size: 20.0,
///     grid_divisions: 20,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone)]
pub struct PlotConfig {
    /// 격자 박스의 한 변 길이 (기본 10.0)
    pub grid_size: f32,
    /// 격자 분할 수 (기본 10)
    pub grid_divisions: usize,
    /// 배경색 RGBA — f32 [0, 1] 범위 (기본 거의 검정)
    ///
    /// wgpu의 `Color` 구조체는 f64를 사용하므로 렌더러 내부에서 변환합니다.
    pub background_color: [f32; 4],
    /// 범례 항목 목록 (빈 벡터면 범례 숨김)
    pub legend: Vec<LegendEntry>,
    /// 축 눈금 레이블 표시 여부 (기본 true)
    pub show_axis_labels: bool,
}

impl Default for PlotConfig {
    fn default() -> Self {
        Self {
            grid_size: 10.0,
            grid_divisions: 10,
            background_color: [0.01, 0.01, 0.02, 1.0],
            legend: vec![],
            show_axis_labels: true,
        }
    }
}