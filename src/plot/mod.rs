mod camera;
mod config;
mod data;
mod geometry;
mod mesh;
mod renderer;
mod vertex;

pub use camera::Camera;
pub use config::{LegendEntry, PlotConfig};
pub use data::PlotData;
pub use geometry::{create_full_grid_data, plot_parametric_curve, plot_scatter, plot_wireframe};
pub use mesh::Mesh;
pub use renderer::App;
pub use vertex::Vertex;