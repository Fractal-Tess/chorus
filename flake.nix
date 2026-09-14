{
  description = "CPU-only local TTS development environment";

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
          libsndfile
          python312
          tailwindcss
          uv
          zlib
        ];

        env = {
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc
            pkgs.zlib
            pkgs.libsndfile
          ];
          UV_PYTHON = "${pkgs.python312}/bin/python3.12";
          UV_PYTHON_DOWNLOADS = "never";
        };

        shellHook = ''
          uv sync --locked --python "$UV_PYTHON"
          export VIRTUAL_ENV="$PWD/.venv"
          export PATH="$PWD:$VIRTUAL_ENV/bin:$PATH"
        '';
      };
    };
}
