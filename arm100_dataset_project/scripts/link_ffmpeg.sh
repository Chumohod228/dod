#!/usr/bin/env bash
set -euo pipefail
# Keep ROS/system Python paths outside this environment.
unset PYTHONPATH PYTHONHOME
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
"$project_dir/.venv/bin/python" - "$project_dir" <<'PY'
from pathlib import Path
import sys
import imageio_ffmpeg
link = Path(sys.argv[1]) / '.venv/bin/ffmpeg'
if link.is_symlink() or link.exists():
    link.unlink()
link.symlink_to(imageio_ffmpeg.get_ffmpeg_exe())
print('ffmpeg:', link.resolve())
PY
