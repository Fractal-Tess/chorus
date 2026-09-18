{
  lib,
  stdenv,
  bash,
  cacert,
  coreutils,
  gnused,
  git,
  libsndfile,
  python312,
  rsync,
  sox,
  ffmpeg,
  espeak-ng,
  uv,
  util-linux,
  zlib,
}:
let
  root = ../.;
  # Only files needed by the service are admitted to the store. In
  # particular, model directories are represented by manifests, while all
  # weights and development/runtime caches stay outside the Nix closure.
  source = lib.cleanSourceWith {
    src = root;
    filter =
      path: type:
      let
        relative =
          if toString path == toString root then "" else lib.removePrefix "${toString root}/" (toString path);
        parts = lib.splitString "/" relative;
        top = if parts == [ ] then "" else builtins.head parts;
        rest = if builtins.length parts > 1 then builtins.tail parts else [ ];
        basename = builtins.baseNameOf (toString path);
        sourceFile = name: type == "regular" && basename == name;
        runtimeFile =
          builtins.length rest == 2
          && builtins.elem (builtins.head rest) [
            "breeze"
            "fish"
          ]
          && builtins.elem basename [
            "worker.py"
            "pyproject.toml"
            "uv.lock"
          ];
      in
      if relative == "" then
        true
      else if
        builtins.elem basename [
          ".git"
          ".hg"
          ".svn"
          ".direnv"
          ".venv"
          "__pycache__"
          ".cache"
          "cache"
          "outputs"
          "node_modules"
          "upstream"
        ]
      then
        false
      else if
        lib.hasSuffix ".pyc" relative
        || lib.hasSuffix ".pyo" relative
        || lib.hasSuffix ".safetensors" relative
        || lib.hasSuffix ".onnx" relative
        || lib.hasSuffix ".pth" relative
        || lib.hasSuffix ".bin" relative
      then
        false
      else if builtins.length parts == 1 then
        (
          type == "directory"
          && builtins.elem top [
            "src"
            "static"
            "models"
            "runtimes"
          ]
        )
        || sourceFile "channels.toml"
        || sourceFile "pyproject.toml"
        || sourceFile "uv.lock"
      else if top == "src" || top == "static" then
        type == "directory" || type == "regular"
      else if top == "models" then
        (type == "directory" && builtins.length parts <= 3)
        || (builtins.length parts == 4 && sourceFile "manifest.json")
      else if top == "runtimes" then
        if builtins.length parts <= 2 then
          type == "directory"
          && builtins.elem basename [
            "breeze"
            "fish"
          ]
        else
          runtimeFile
      else
        false;
  };

  runtimeTools = [
    bash
    coreutils
    gnused
    git
    rsync
    uv
    util-linux
    espeak-ng
    ffmpeg
    sox
  ];
  libraryPath = lib.makeLibraryPath [
    stdenv.cc.cc
    libsndfile
    zlib
  ];
  toolPath = lib.makeBinPath runtimeTools;
  caBundle = "${cacert}/etc/ssl/certs/ca-bundle.crt";
in
stdenv.mkDerivation {
  pname = "chorus";
  version = (builtins.fromTOML (builtins.readFile ../pyproject.toml)).project.version;
  src = source;

  dontConfigure = true;
  dontBuild = true;

  installPhase = ''
    mkdir -p "$out/share/chorus/source" "$out/bin"
    cp -R . "$out/share/chorus/source/"
    substitute ${./launcher.sh} "$out/bin/chorus-service" \
      --subst-var-by bash "${bash}" \
      --subst-var-by chorusSource "$out/share/chorus/source" \
      --subst-var-by python "${python312}" \
      --subst-var-by uv "${uv}" \
      --subst-var-by toolPath "${toolPath}" \
      --subst-var-by rsync "${rsync}" \
      --subst-var-by git "${git}" \
      --subst-var-by libraryPath "${libraryPath}" \
      --subst-var-by caBundle "${caBundle}"
    chmod 0755 "$out/bin/chorus-service"
  '';

  meta = {
    description = "Local multi-engine TTS service";
    mainProgram = "chorus-service";
    platforms = [ "x86_64-linux" ];
  };
}
