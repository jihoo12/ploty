use super::{mesh::Mesh, vertex::Vertex};

/// XZ 평면 기준 격자 박스를 생성합니다.
pub fn create_full_grid_data(size: f32, divisions: usize) -> Mesh {
    let step = size / divisions as f32;
    let start = -size / 2.0;
    let end = -start;

    let mut vertices: Vec<Vertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    let color = [0.2, 0.2, 0.2];
    let mut add_line = |p1: [f32; 3], p2: [f32; 3]| {
        let base = vertices.len() as u32;
        vertices.push(Vertex::new(p1, color));
        vertices.push(Vertex::new(p2, color));
        indices.extend_from_slice(&[base, base + 1]);
    };

    for i in 0..=divisions {
        let d = start + i as f32 * step;
        add_line([d, start, start], [d, start, end]);
        add_line([start, start, d], [end, start, d]);
        add_line([d, start, start], [d, end, start]);
        add_line([start, d, start], [end, d, start]);
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

    for &z in z_range {
        for &x in x_range {
            let y = y_func(x, z);
            y_min = y_min.min(y);
            y_max = y_max.max(y);
            vertices.push(Vertex {
                position: [x, y, z, 1.0],
                color: [0.0; 4],
            });
        }
    }

    let denom = (y_max - y_min).max(f32::EPSILON);
    for v in &mut vertices {
        let intensity = 0.5 + 0.5 * (v.position[1] - y_min) / denom;
        v.color = [
            base_color[0] * intensity,
            base_color[1] * intensity,
            base_color[2] * intensity,
            1.0,
        ];
    }

    let mut indices = Vec::with_capacity(rows * cols * 4);
    for r in 0..rows {
        for c in 0..(cols - 1) {
            indices.extend_from_slice(&[(r * cols + c) as u32, (r * cols + c + 1) as u32]);
        }
    }
    for r in 0..(rows - 1) {
        for c in 0..cols {
            indices.extend_from_slice(&[(r * cols + c) as u32, ((r + 1) * cols + c) as u32]);
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