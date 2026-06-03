{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    cargo
    rustc
    rustfmt
    clippy
    rust-analyzer

    # wayland / GPU 관련
    pkg-config
    wayland
    libxkbcommon

    # Vulkan / GPU 백엔드
    vulkan-loader
    vulkan-tools
    mesa
  ];

  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
    wayland
    libxkbcommon
    vulkan-loader
    mesa
  ]);

  shellHook = ''
    export RUST_SRC_PATH=${pkgs.rustPlatform.rustLibSrc}
    echo "🦀 Welcome to the Rust development environment! 🦀"
    rustc --version
  '';
}
