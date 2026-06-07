# wplot — Lightweight 3D Plotting in Rust

A hardware-accelerated 3D data visualization engine built with Rust and [wgpu](https://wgpu.rs). Render mathematical surfaces, animated wave functions, and scatter plots with a clean builder-style API — no boilerplate required.

---

## Features

- **Cross-platform GPU rendering** via wgpu (Vulkan, Metal, DirectX 12)
- **Static surface plots** — wireframe meshes from any `f(x, z) → y` function
- **Animated graphs** — time-varying surfaces driven by `f(x, z, t) → y` closures
- **Scatter plots** — efficient point-cloud rendering via a dedicated point-list pipeline
- **Orbit camera** — left-drag to rotate, middle-drag to pan, scroll to zoom, fully interactive
- **Axis tick labels** — numeric labels on all three axes, projected from 3D world space each frame using glyphon
- **Legend overlay** — per-graph labels rendered with GPU-accelerated text (glyphon)
- **Depth-correct compositing** — grid, surfaces, and points layer correctly in 3D
- **Builder API** — chain `.add_graph()`, `.add_animated_graph()`, and `.with_config()` to build a scene in a few lines

---

## Quick Start

Requires the [Rust toolchain](https://rustup.rs).

```bash
git clone https://github.com/wplot/wplot
cd wplot
cargo run --release
```

---

## Usage

```rust
let range: Vec<f32> = (0..60)
    .map(|i| -5.0 + (i as f32 / 59.0) * 10.0)
    .collect();

let config = PlotConfig {
    grid_size: 12.0,
    grid_divisions: 12,
    legend: vec![
        LegendEntry { label: "sin(r)".into(),  color: [0.2, 0.5, 1.0] },
        LegendEntry { label: "wave".into(),     color: [1.0, 0.45, 0.15] },
    ],
    .show_axis_labels: false,
    ..Default::default()
};

let plot_data = PlotData::new()
    .with_config(config)
    // Static surface: sin(r)
    .add_graph(plot_wireframe(
        &range, &range,
        |x, z| (x * x + z * z).sqrt().sin(),
        [0.2, 0.5, 1.0],
    ))
    // Animated surface: outward ripple
    .add_animated_graph(
        range.clone(), range.clone(),
        |x, z, t| {
            let r = (x * x + z * z).sqrt();
            (r - t * 2.5).sin() * (-r * 0.15).exp()
        },
        [1.0, 0.45, 0.15],
    )
    // Static helix
    .add_parametric_curve(
        (0..=400).map(|i| i as f32 / 400.0 * 6.0 * PI).collect::<Vec<_>>(),
        |u| [3.0 * u.cos(), u * 0.3, 3.0 * u.sin()],
        [1.0, 0.4, 0.1],
    )

    // Animated Lissajous figure
    .add_animated_parametric_curve(
        (0..=300).map(|i| i as f32 / 300.0 * 2.0 * PI).collect::<Vec<_>>(),
        |u, t| [3.0 * (2.0 * u + t).sin(), 3.0 * (3.0 * u).sin(), u.cos() * 2.0],
        [0.9, 0.2, 0.8],
    );
```

---

## Camera Controls

| Input | Action |
|---|---|
| Left click + drag | Orbit (yaw & pitch) |
| Middle click + drag | Pan (shift look-at target) |
| Scroll wheel | Zoom in / out |
| Window resize | Viewport and depth buffer update automatically |

---

## Architecture

| File | Responsibility |
|---|---|
| `main.rs` | winit event loop, input routing (left + middle mouse, scroll) |
| `renderer.rs` | wgpu device, pipelines, render loop; axis tick label projection |
| `camera.rs` | Spherical-coordinate orbit camera with pan support (`target: Vec3`) |
| `geometry.rs` | Mesh generation: grid, wireframe, scatter |
| `mesh.rs` | CPU-side vertex/index storage and merging |
| `data.rs` | `PlotData` builder, animated graph definitions |
| `config.rs` | `PlotConfig` and `LegendEntry` |
| `vertex.rs` | GPU vertex layout (`Float32x4` position + color) |

Vertices use a `[f32; 4]` layout for both position and color, matching wgpu buffer alignment requirements directly.

---

## License

Apache-2.0 license