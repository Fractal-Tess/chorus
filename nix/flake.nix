{
  description = "Local CPU and CUDA TTS development environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          espeak-ng
          ffmpeg
          git-lfs
          sox
          libsndfile
          python312
          uv
          zlib
        ];

        env = {
          # Interactive children (the editor, Hyprland clients, ...) inherit this
          # path, so the libstdc++ offered here must be at least as new as the
          # host system's; wheels in .venv still need one on NixOS.
          LD_LIBRARY_PATH = "/run/opengl-driver/lib:" + pkgs.lib.makeLibraryPath [
            pkgs.gcc16.cc.lib
            pkgs.zlib
            pkgs.libsndfile
          ];
          UV_PYTHON = "${pkgs.python312}/bin/python3.12";
          UV_PYTHON_DOWNLOADS = "never";
        };

        shellHook = ''
          uv sync --locked --python "$UV_PYTHON"
          export VIRTUAL_ENV="$PWD/.venv"
          export PATH="$VIRTUAL_ENV/bin:$PATH"
        '';
      };
    };
}
