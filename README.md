## Ploty: A Lightweight 3D Plotting Library
Ploty is a hardware-accelerated 3D data visualization engine built with Rust and wgpu. It allows developers to render complex mathematical surfaces and scatter plots with high performance using a simple builder-style API.

## 🚀 Features
- Hardware Accelerated: Leverages wgpu for cross-platform, high-performance GPU rendering (Vulkan, Metal, DX12).

- Surface Plotting: Easily generate 3D wireframe meshes from mathematical functions.

- Scatter Plots: Render point cloud data efficiently using specialized point-list pipelines.

- Interactive Camera: Full 3D navigation (Orbit, Zoom, and Drag) using winit event handling.

- Builder Pattern API: Clean and modular data construction for adding multiple datasets to a single scene.

- Depth Testing: Proper 3D depth composition ensuring correct visual layering of grids, graphs, and points.
## 📖 How It Works

- Data Structure
<br>
The engine uses a Vertex struct with a Float32x4 layout for both position and color, ensuring alignment compatibility with GPU buffers.

## Controls
- Left Click + Drag: Orbit/Rotate the camera (Yaw and Pitch).

- Mouse Wheel: Zoom in and out (Adjusts camera radius).

- Window Resize: Responsive viewport and depth-buffer scaling.

## 🏗 Installation & Running
- Ensure you have the Rust toolchain installed.

- Clone this repository.

- Run the project:

```bash
cargo run --release
```