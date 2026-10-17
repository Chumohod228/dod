#!/usr/bin/env bash
set -euo pipefail
# Keep ROS/system Python paths outside this environment.
unset PYTHONPATH PYTHONHOME
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"
exec "$project_dir/.venv/bin/arm100-collect" "$@"
