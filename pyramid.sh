#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec ./arm100_dataset_project/.venv/bin/python ./pyramid_cycle.py "$@"
