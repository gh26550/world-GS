#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
venv="${WORLDGS_VENV:-$HOME/.venvs/world-gs}"
if [[ "$venv" =~ [[:space:]] ]]; then
  echo 'WORLDGS_VENV must not contain whitespace.' >&2
  exit 2
fi
if [[ ! -e "$venv" ]]; then
  "${PYTHON_BIN:-python3}" -m venv "$venv"
fi
python="$venv/bin/python"
"$python" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "Use Python 3.10, 3.11 or 3.12"'
"$python" -m pip install 'torch==2.7.1' 'torchvision==0.22.1' --index-url https://download.pytorch.org/whl/cu128
"$python" -m pip install -c constraints.txt -e '.[generation,training]'
echo "Ready: $venv. Use ./run.sh generate ... and ./run.sh train ..."
