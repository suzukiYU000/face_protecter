#!/usr/bin/env bash

uv_is_wsl_mounted_project() {
  [[ -n "${WSL_DISTRO_NAME:-}" && "${1:-}" == /mnt/* ]]
}

uv_project_id() {
  local input="$1"

  if command -v sha256sum >/dev/null 2>&1; then
    printf '%s' "$input" | sha256sum | awk '{print substr($1, 1, 12)}'
    return
  fi

  if command -v shasum >/dev/null 2>&1; then
    printf '%s' "$input" | shasum -a 256 | awk '{print substr($1, 1, 12)}'
    return
  fi

  printf '%s' "$input" | cksum | awk '{print $1}'
}

configure_uv_runtime() {
  local script_dir="$1"
  local cache_namespace="${2:-uv-project}"
  local default_tmp="$script_dir/.tmp"
  local default_pip_cache="$script_dir/.pip-cache"

  TMPDIR="$default_tmp"
  TMP="$default_tmp"
  TEMP="$default_tmp"
  PIP_CACHE_DIR="$default_pip_cache"
  unset UV_PROJECT_ENVIRONMENT
  unset UV_CACHE_DIR

  if uv_is_wsl_mounted_project "$script_dir"; then
    local project_id
    local runtime_root

    project_id="$(uv_project_id "$script_dir")"
    runtime_root="${XDG_CACHE_HOME:-$HOME/.cache}/${cache_namespace}/${project_id}"

    UV_PROJECT_ENVIRONMENT="$runtime_root/.venv"
    UV_CACHE_DIR="$runtime_root/uv-cache"
    TMPDIR="$runtime_root/tmp"
    TMP="$TMPDIR"
    TEMP="$TMPDIR"
    PIP_CACHE_DIR="$runtime_root/pip-cache"
    : "${UV_LINK_MODE:=copy}"
    export UV_PROJECT_ENVIRONMENT UV_CACHE_DIR UV_LINK_MODE
  fi

  mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" models Results runs/logs
  export TMPDIR TMP TEMP PIP_CACHE_DIR
}

persist_uv_runtime_profile() {
  local profile_file="$1"
  local profile="$2"
  local with_onnx="$3"
  local name

  {
    echo "UV_PROFILE=$profile"
    echo "UV_WITH_ONNX=$with_onnx"
    for name in UV_PROJECT_ENVIRONMENT UV_CACHE_DIR UV_LINK_MODE TMPDIR TMP TEMP PIP_CACHE_DIR; do
      if [[ -n "${!name:-}" ]]; then
        echo "$name=${!name}"
      fi
    done
  } > "$profile_file"
}

load_uv_runtime_profile() {
  local profile_file="$1"
  local name

  # shellcheck disable=SC1090
  source "$profile_file"

  for name in UV_PROJECT_ENVIRONMENT UV_CACHE_DIR UV_LINK_MODE TMPDIR TMP TEMP PIP_CACHE_DIR; do
    if [[ -n "${!name:-}" ]]; then
      export "$name"
    fi
  done
}
