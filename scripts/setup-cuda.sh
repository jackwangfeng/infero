#!/usr/bin/env bash
# Point the build at a CUDA userspace (libcublas / libnvrtc / headers) without
# needing a system-wide toolkit install.
#
# This box has no /usr/local/cuda and no nvcc, but the pip `nvidia-*` wheels that
# PyTorch pulls in ship the exact same shared objects and headers. We symlink one
# of those into vendor/cuda/ so the build script has a stable path to bake into
# the binary's rpath.
#
# Override the source with:  CUDA_HOME=/usr/local/cuda ./scripts/setup-cuda.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$ROOT/vendor/cuda"

find_pip_cuda() {
    # Newest CUDA-13 wheel layout first: <site-packages>/nvidia/cu13/{lib,include}
    for d in "$HOME"/miniconda3/lib/python*/site-packages/nvidia/cu13 \
             "$HOME"/miniconda3/envs/*/lib/python*/site-packages/nvidia/cu13 \
             "$HOME"/**/site-packages/nvidia/cu13; do
        if [[ -f "$d/lib/libnvrtc.so.13" && -f "$d/include/cuda_fp16.h" ]]; then
            echo "$d"
            return 0
        fi
    done
    return 1
}

if [[ -n "${CUDA_HOME:-}" ]]; then
    SRC="$CUDA_HOME"
elif SRC="$(find_pip_cuda)"; then
    :
else
    echo "error: no CUDA userspace found." >&2
    echo "       set CUDA_HOME, or 'pip install nvidia-cuda-nvrtc-cu13 nvidia-cublas-cu13'" >&2
    exit 1
fi

# A pip wheel puts the libraries in lib/; a toolkit install uses lib64/, and
# points include/ at the same place either way.
if [[ -d "$SRC/lib" ]]; then
    SRCLIB="$SRC/lib"
elif [[ -d "$SRC/lib64" ]]; then
    SRCLIB="$SRC/lib64"
elif [[ -d "$SRC/targets/x86_64-linux/lib" ]]; then
    SRCLIB="$SRC/targets/x86_64-linux/lib"
else
    echo "error: $SRC has no lib/, lib64/ or targets/x86_64-linux/lib/" >&2
    exit 1
fi
[[ -d "$SRC/include" ]] || { echo "error: $SRC lacks include/" >&2; exit 1; }

mkdir -p "$VENDOR"
ln -sfn "$SRCLIB" "$VENDOR/lib"
ln -sfn "$SRC/include" "$VENDOR/include"

echo "cuda source : $SRC"
echo "vendor link : $VENDOR"
echo
echo "libs:"
ls "$VENDOR/lib" | grep -E '^lib(cublas|cublasLt|nvrtc|cudart)\.so' | sed 's/^/  /'
