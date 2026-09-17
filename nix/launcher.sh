#!@bash@/bin/bash
set -Eeuo pipefail

# The Nix package is immutable.  Keep the application checkout, virtualenvs,
# model catalog and every downloader cache below the configured service paths
# (or their standalone defaults).
readonly STORE_SOURCE="@chorusSource@"
readonly NIX_PYTHON="@python@/bin/python3.12"
readonly UV="@uv@/bin/uv"
readonly TOOL_PATH="@toolPath@"
export PATH="$TOOL_PATH:/run/current-system/sw/bin:${PATH:-/usr/bin:/bin}"

state_dir=${CHORUS_STATE_DIR:-/var/lib/chorus}
cache_dir=${CHORUS_CACHE_DIR:-/var/cache/chorus}
engines=${CHORUS_ENGINES-kokoro}
models_dirs_value=${CHORUS_MODELS_DIRS-"$state_dir/models"}
application="$state_dir/application"

fail() {
    printf 'chorus-service: %s\n' "$*" >&2
    exit 1
}

[[ -d "$STORE_SOURCE" ]] || fail "packaged source is missing: $STORE_SOURCE"
[[ -n "$state_dir" && -n "$cache_dir" ]] || fail "state and cache paths must be non-empty"

case "$models_dirs_value" in
    ''|:*|*:|*::* ) fail "CHORUS_MODELS_DIRS must be a colon-separated list of non-empty paths" ;;
esac
IFS=: read -r -a model_dirs <<< "$models_dirs_value"
for models_dir in "${model_dirs[@]}"; do
    [[ -n "$models_dir" ]] || fail "CHORUS_MODELS_DIRS contains an empty model path"
done
primary_models_dir=${model_dirs[0]}

# Validate this before creating directories or running uv.  The CLI validates
# again, but doing it here prevents a typo from partially provisioning state.
case "$engines" in
    ''|,*|*,|*,,*) fail "CHORUS_ENGINES must be a comma-separated list of supported engines" ;;
esac
declare -A seen=()
IFS=',' read -r -a selected_engines <<< "$engines"
for item in "${selected_engines[@]}"; do
    engine="${item#"${item%%[![:space:]]*}"}"
    engine="${engine%"${engine##*[![:space:]]}"}"
    [[ -n "$engine" ]] || fail "CHORUS_ENGINES contains an empty engine name"
    case "$engine" in
        kokoro|piper|kitten|pocket|supertonic|breeze|fish) ;;
        *) fail "unknown engine '$engine' in CHORUS_ENGINES (supported: kokoro,piper,kitten,pocket,supertonic,breeze,fish)" ;;
    esac
    [[ -z "${seen[$engine]+present}" ]] || fail "CHORUS_ENGINES contains duplicate engine '$engine'"
    seen[$engine]=1
done

mkdir -p "$state_dir" "$cache_dir" "$application" "$primary_models_dir"

# Hold the lock for the server lifetime so a second launch cannot rewrite a
# running worker's source. The kernel releases it even after an unclean exit.
exec 9>"$state_dir/.service.lock"
flock --nonblock 9 || fail "another Chorus instance is using $state_dir"

# Never let uv, Hugging Face, torch, or git put mutable data in $HOME or the
# host user's cache.  Network access remains enabled: the first start installs
# the locked environments and downloads selected model files as requested.
export HOME="$state_dir/home"
export XDG_CACHE_HOME="$cache_dir/xdg"
export UV_CACHE_DIR="$cache_dir/uv"
export UV_PYTHON="$NIX_PYTHON"
export UV_PYTHON_DOWNLOADS=never
export HF_HOME="$cache_dir/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TMPDIR="${TMPDIR:-$cache_dir/tmp}"
export TORCH_HOME="$cache_dir/torch"
export TRITON_CACHE_DIR="$cache_dir/triton"
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$UV_CACHE_DIR" "$HF_HOME" "$HF_HUB_CACHE" \
    "$TMPDIR" "$TORCH_HOME" "$TRITON_CACHE_DIR"

# Nix's dynamic libraries include audio codecs and the OpenGL/CUDA driver.
# Keep the system nvidia-smi visible for GPU diagnostics and the Nix tools
# available to worker subprocesses.
export SSL_CERT_FILE="@caBundle@"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
export CURL_CA_BUNDLE="$SSL_CERT_FILE"
export GIT_SSL_CAINFO="$SSL_CERT_FILE"
export LD_LIBRARY_PATH="/run/opengl-driver/lib:@libraryPath@${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Keep source updates clean while retaining mutable environments and the large
# downloaded upstream checkouts.  The source closure contains manifests only,
# never model weights; the model tree is synced independently below.
"@rsync@/bin/rsync" -a --delete --chmod=u+w --no-owner --no-group \
    --exclude='.venv/***' --exclude='upstream/***' --exclude='models/***' \
    "$STORE_SOURCE/" "$application/"

# The catalog belongs in state, not the read-only Nix store.  Copy only shipped
# manifests so a package upgrade cannot overwrite or remove user-downloaded
# weights (which may have extensions not known to this launcher).
if [[ -d "$STORE_SOURCE/models" ]]; then
    while IFS= read -r -d '' manifest; do
        relative=${manifest#"$STORE_SOURCE/models/"}
        target="$primary_models_dir/$relative"
        mkdir -p "${target%/*}"
        install -m 0644 "$manifest" "$target"
    done < <("@find@/bin/find" "$STORE_SOURCE/models" -type f -name manifest.json -print0)
fi

runtime_metadata() {
    local runtime=$1
    local engine=$2
    "$NIX_PYTHON" - "$runtime/pyproject.toml" "$engine" <<'PY'
import re
import sys
import tomllib

path, engine = sys.argv[1:]
try:
    with open(path, "rb") as stream:
        project = tomllib.load(stream)
    config = project["tool"]["chorus"][engine]
    repository = config["upstream_repository"]
    revision = config["upstream_revision"]
except (KeyError, OSError, tomllib.TOMLDecodeError) as error:
    raise SystemExit(f"invalid {path} runtime metadata: {error}")
if not isinstance(repository, str) or not repository.startswith("https://"):
    raise SystemExit(f"invalid upstream repository for {engine}: {repository!r}")
if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
    raise SystemExit(f"upstream revision for {engine} must be a full commit SHA")
print(repository)
print(revision)
PY
}

provision_runtime() {
    local engine=$1
    local runtime="$application/runtimes/$engine"
    local upstream="$runtime/upstream"
    [[ -f "$runtime/pyproject.toml" && -f "$runtime/uv.lock" && -f "$runtime/worker.py" ]] || \
        fail "$engine runtime is incomplete in the package"

    local metadata repository revision current
    if ! metadata=$(runtime_metadata "$runtime" "$engine"); then
        fail "$engine runtime metadata is invalid; expected HTTPS upstream_repository and full upstream_revision SHA in $runtime/pyproject.toml"
    fi
    repository=$(printf '%s\n' "$metadata" | sed -n '1p')
    revision=$(printf '%s\n' "$metadata" | sed -n '2p')
    if [[ -e "$upstream" && ! -d "$upstream/.git" ]]; then
        fail "$engine upstream exists but is not a git checkout: $upstream"
    fi
    if [[ ! -e "$upstream" ]]; then
        mkdir -p "$upstream"
        "@git@/bin/git" -C "$upstream" init -q
    fi
    if ! "@git@/bin/git" -C "$upstream" remote get-url origin >/dev/null 2>&1; then
        "@git@/bin/git" -C "$upstream" remote add origin "$repository"
    else
        "@git@/bin/git" -C "$upstream" remote set-url origin "$repository"
    fi

    current=$("@git@/bin/git" -C "$upstream" rev-parse HEAD 2>/dev/null || true)
    if [[ "$current" != "$revision" ]]; then
        "@git@/bin/git" -C "$upstream" fetch --no-tags --depth=1 origin "$revision" || \
            fail "could not fetch pinned $engine upstream revision $revision from $repository"
        "@git@/bin/git" -C "$upstream" checkout --detach --force FETCH_HEAD || \
            fail "could not check out pinned $engine upstream revision $revision"
    fi
    current=$("@git@/bin/git" -C "$upstream" rev-parse HEAD 2>/dev/null || true)
    [[ "$current" == "$revision" ]] || fail "$engine upstream is not at requested revision $revision (got ${current:-none})"

    # uv selects the project-local .venv while using the shared cache above.
    "$UV" sync --project "$runtime" --locked --python "$NIX_PYTHON" || \
        fail "$engine runtime dependency provisioning failed; check network access and uv cache"
}

# Avoid inheriting the main environment while uv creates project-local runtime
# environments.  The final VIRTUAL_ENV points at the main application only.
unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT || true
"$UV" sync --project "$application" --locked --python "$NIX_PYTHON" || \
    fail "main dependency provisioning failed; check network access and uv cache"

for item in "${selected_engines[@]}"; do
    engine="${item#"${item%%[![:space:]]*}"}"
    engine="${engine%"${engine##*[![:space:]]}"}"
    case "$engine" in
        breeze|fish) provision_runtime "$engine" ;;
    esac
done
export VIRTUAL_ENV="$application/.venv"
export PATH="$VIRTUAL_ENV/bin:/run/current-system/sw/bin:$TOOL_PATH:${PATH:-/usr/bin:/bin}"
export TTS_MODELS_DIRS="$models_dirs_value"
exec "$VIRTUAL_ENV/bin/python" -m chorus.cli \
    --engines "$engines" "$@"
