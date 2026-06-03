use super::vertex::Vertex;

/// CPU-side 메시: 정점 목록 + 인덱스 목록
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl Mesh {
    pub fn new(vertices: Vec<Vertex>, indices: Vec<u32>) -> Self {
        Self { vertices, indices }
    }

    pub fn empty() -> Self {
        Self::new(vec![], vec![])
    }
}

/// 여러 Mesh를 인덱스 오프셋을 반영해 하나로 합칩니다.
pub fn merge_meshes(meshes: Vec<Mesh>) -> Mesh {
    let total_v: usize = meshes.iter().map(|m| m.vertices.len()).sum();
    let total_i: usize = meshes.iter().map(|m| m.indices.len()).sum();

    let mut vertices = Vec::with_capacity(total_v);
    let mut indices = Vec::with_capacity(total_i);

    for mesh in meshes {
        let offset = vertices.len() as u32;
        vertices.extend(mesh.vertices);
        indices.extend(mesh.indices.into_iter().map(|i| i + offset));
    }

    Mesh::new(vertices, indices)
}