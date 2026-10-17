#!/usr/bin/env bash
# Recreate the environment from local wheels on Linux x86_64 / Python 3.12.
set -euo pipefail
# Keep ROS/system Python paths outside this environment.
unset PYTHONPATH PYTHONHOME
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"
python_cmd=${ARM100_PYTHON:-python3}
"$python_cmd" -c 'import sys,platform; assert sys.version_info[:2] == (3,12), "Python 3.12 required"; assert platform.system()=="Linux" and platform.machine()=="x86_64", "Wheel bundle is for Linux x86_64"'
if [[ ! -x .tools/uv ]]; then
    echo "Missing .tools/uv: copy the whole project, including hidden directories" >&2
    exit 1
fi
if [[ ! -f .venv/pyvenv.cfg ]]; then
    .tools/uv venv --python "$python_cmd" .venv
fi
(cd wheelhouse && sha256sum --check --quiet SHA256SUMS)
.tools/uv pip sync --python .venv/bin/python --no-index --find-links wheelhouse requirements.lock
.tools/uv pip install --python .venv/bin/python --no-index --find-links wheelhouse --no-deps --no-build-isolation --reinstall-package lerobot --reinstall-package arm100-dataset-project -e vendor/lerobot -e .
"$project_dir/scripts/link_ffmpeg.sh"
.venv/bin/python -m pip check
"$project_dir/scripts/run.sh" doctor
