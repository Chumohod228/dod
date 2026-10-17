#!/usr/bin/env bash
set -euo pipefail
# Keep ROS/system Python paths outside this environment.
unset PYTHONPATH PYTHONHOME
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"
stamp=$(date -u +%Y%m%dT%H%M%S)_$$
dataset_root="datasets/smoke_$stamp"
.venv/bin/python -m unittest discover -s tests -v
scripts/run.sh simulate --root "$dataset_root"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
.venv/bin/lerobot-train \
    --dataset.repo_id=local/synthetic_bimanual --dataset.root="$dataset_root" \
    --dataset.video_backend=pyav --policy.type=act --policy.device=cpu \
    --policy.push_to_hub=false --policy.pretrained_backbone_weights=null \
    --policy.chunk_size=4 --policy.n_action_steps=4 \
    --policy.dim_model=64 --policy.n_heads=4 --policy.dim_feedforward=128 \
    --policy.n_encoder_layers=1 --policy.n_decoder_layers=1 --policy.n_vae_encoder_layers=1 \
    --output_dir="outputs/smoke_$stamp" --steps=2 --batch_size=2 --num_workers=0 \
    --env_eval_freq=0 --log_freq=1 --save_freq=2 --wandb.enable=false
scripts/run.sh validate "$dataset_root"
