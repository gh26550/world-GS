#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
venv="${WORLDGS_VENV:-$HOME/.venvs/world-gs}"
if [[ "$venv" =~ [[:space:]] ]]; then
  echo 'WORLDGS_VENV must not contain whitespace (gsplat linker limitation).' >&2
  exit 2
fi
python="$venv/bin/python"
if [[ ! -x "$python" ]]; then
  echo 'Environment not found. Run bash setup.sh first.' >&2
  exit 2
fi
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-2}"
export NUM_CHANNELS="${NUM_CHANNELS:-3,4}"
if [[ "${1:-}" == train || "${1:-}" == audit-holes ]]; then
  export TORCH_CUDA_ARCH_LIST="${WORLDGS_CUDA_ARCH:-$("$python" -c 'import torch; a,b=torch.cuda.get_device_capability(); print(f"{a}.{b}")')}"
fi
exec "$python" -m worldgs.cli "$@"
