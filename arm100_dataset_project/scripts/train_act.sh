#!/usr/bin/env bash
set -euo pipefail
# Keep ROS/system Python paths outside this environment.
unset PYTHONPATH PYTHONHOME
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"
if [[ $# -lt 1 ]]; then
    echo "Usage: scripts/train_act.sh DATASET_ROOT [additional lerobot-train arguments]" >&2
    exit 1
fi
dataset_root=$(realpath -- "$1")
shift
dataset_id=$("$project_dir/.venv/bin/python" - "$dataset_root" <<'PY'
import json,sys
from pathlib import Path
from arm100_collect.dataset import validate_dataset
p=Path(sys.argv[1])/'capture_config.json'
cfg=json.loads(p.read_text())
if cfg['simulated']:
    raise SystemExit('Synthetic test dataset: use scripts/smoke.sh to test training')
validate_dataset(Path(sys.argv[1]), quick=True)
print(cfg['repo_id'])
PY
)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
exec .venv/bin/lerobot-train \
    --dataset.repo_id="$dataset_id" --dataset.root="$dataset_root" \
    --dataset.video_backend=pyav --dataset.eval_split=0.1 \
    --policy.type=act --policy.device=cpu --policy.push_to_hub=false \
    --policy.pretrained_backbone_weights=null \
    --output_dir="outputs/act_$(date -u +%Y%m%dT%H%M%S)_$$" \
    --batch_size=8 --steps=100000 --num_workers=2 \
    --env_eval_freq=0 --eval_steps=1000 --log_freq=100 --save_freq=10000 \
    --wandb.enable=false "$@"
