#!/usr/bin/env bash
set -euo pipefail
cd /home/chumohod/dod
exec /home/chumohod/dod/arm100_dataset_project/.venv/bin/python -u /home/chumohod/dod/leader_follower.py --port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6080141-if00 --leader-ids 19 20 21 22 23 24 --follower-ids 1 2 3 4 5 6 --leader-ranges /home/chumohod/dod/results/second_pair_setup_20261015T090100_036935Z/leader_ranges.json --follower-limits /home/chumohod/dod/results/second_pair_setup_20261015T090100_036935Z/follower_limits.json --speed 600 --acceleration 50 "$@"
