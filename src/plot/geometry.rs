use super::{mesh::Mesh, vertex::Vertex};

/// XZ 평면 기준 격자 박스를 생성합니다.
pub fn create_full_grid_data(size: f32, divisions: usize) -> Mesh {
    let step = size / divisions as f32;
    let half = size / 2.0;
    let start = -half;
    let end = half;

    // 각 분할선(divisions + 1개)마다 면당 2선씩, 3면 → 6선. 선 하나 = 정점 2개.
    let line_count = (divisions + 1) * 6;
    let mut vertices: Vec<Vertex> = Vec::with_capacity(line_count * 2);
    let mut indices: Vec<u32>    = Vec::with_capacity(line_count * 2);

    let color = [0.2, 0.2, 0.2];
    let mut add_line = |p1: [f32; 3], p2: [f32; 3]| {
        let base = vertices.len() as u32;
        vertices.push(Vertex::new(p1, color));
        vertices.push(Vertex::new(p2, color));
        indices.push(base);
        indices.push(base + 1);
    };

    for i in 0..=divisions {
        let d = start + i as f32 * step;
        // XZ 바닥면
        add_line([d, start, start], [d, start, end]);
        add_line([start, start, d], [end, start, d]);
        // XY 뒷면 (Z = start)
        add_line([d, start, start], [d, end, start]);
        add_line([start, d, start], [end, d, start]);
        // YZ 왼쪽면 (X = start)
        add_line([start, d, start], [start, d, end]);
        add_line([start, start, d], [start, end, d]);
    }

    Mesh::new(vertices, indices)
}

/// 2D 함수 y = f(x, z) 의 와이어프레임 서피스를 생성합니다.
/// Y 값에 따라 `base_color` 밝기가 그라데이션됩니다.
pub fn plot_wireframe(
    x_range: &[f32],
    z_range: &[f32],
    y_func: impl Fn(f32, f32) -> f32,
    base_color: [f32; 3],
) -> Mesh {
    let rows = z_range.len();
    let cols = x_range.len();

    let mut vertices = Vec::with_capacity(rows * cols);
    let (mut y_min, mut y_max) = (f32::MAX, f32::MIN);

    // 정점 생성 + y 범위 계산을 한 번의 루프로
    for &z in z_range {
        for &x in x_range {
            let y = y_func(x, z);
            if y < y_min { y_min = y; }
            if y > y_max { y_max = y; }
            vertices.push(Vertex {
                position: [x, y, z, 1.0],
                color: [0.0; 4],
            });
        }
    }

    // 그라데이션 색상 적용
    // `base_color`를 `cr/cg/cb`로 분해해 루프 변수 `r`(행 인덱스)과의 이름 충돌을 피합니다.
    let denom = (y_max - y_min).max(f32::EPSILON);
    let [cr, cg, cb] = base_color;
    for v in &mut vertices {
        let t = 0.4 + 0.6 * (v.position[1] - y_min) / denom;
        v.color = [cr * t, cg * t, cb * t, 1.0];
    }

    // 인덱스: 행 방향 + 열 방향 선분
    let h_lines = rows * (cols - 1);
    let v_lines = (rows - 1) * cols;
    let mut indices = Vec::with_capacity((h_lines + v_lines) * 2);

    for r in 0..rows {
        for c in 0..(cols - 1) {
            let i = (r * cols + c) as u32;
            indices.push(i);
            indices.push(i + 1);
        }
    }
    for r in 0..(rows - 1) {
        for c in 0..cols {
            let i = (r * cols + c) as u32;
            indices.push(i);
            indices.push(i + cols as u32);
        }
    }

    Mesh::new(vertices, indices)
}

/// 3D 산점도 메시를 생성합니다.
pub fn plot_scatter(points: &[(f32, f32, f32)], color: [f32; 3]) -> Mesh {
    let vertices: Vec<Vertex> = points
        .iter()
        .map(|&(x, y, z)| Vertex::new([x, y, z], color))
        .collect();
    let indices: Vec<u32> = (0..points.len() as u32).collect();
    Mesh::new(vertices, indices)
}

/// 파라메트릭 곡선 `func(u) → [x, y, z]` 의 와이어프레임 메시를 생성합니다.
///
/// `u_range`의 연속 두 점이 하나의 선분을 이룹니다.
/// 샘플이 1개 이하이면 빈 메시를 반환합니다.
pub fn plot_parametric_curve(
    u_range: &[f32],
    func: impl Fn(f32) -> [f32; 3],
    color: [f32; 3],
) -> Mesh {
    if u_range.len() < 2 {
        return Mesh::new(vec![], vec![]);
    }

    let vertices: Vec<Vertex> = u_range
        .iter()
        .map(|&u| {
            let [x, y, z] = func(u);
            Vertex::new([x, y, z], color)
        })
        .collect();

    // LineList: 각 연속 쌍(i, i+1)이 하나의 선분
    let n = vertices.len() as u32;
    let indices: Vec<u32> = (0..n - 1)
        .flat_map(|i| [i, i + 1])
        .collect();

    Mesh::new(vertices, indices)
}