#!/usr/bin/env bash
set -euo pipefail
cd /home/chumohod/dod
exec /home/chumohod/dod/arm100_dataset_project/.venv/bin/python -u /home/chumohod/dod/leader_follower.py --port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6081918-if00 --leader-ids 7 8 9 10 11 12 --follower-ids 13 14 15 16 17 18 --leader-ranges /home/chumohod/dod/results/pair_setup_7_12_to_13_18_20261015T084734_891910Z/leader_ranges.json --follower-limits /home/chumohod/dod/results/pair_setup_7_12_to_13_18_20261015T084734_891910Z/follower_limits.json --speed 600 --acceleration 50 "$@"
