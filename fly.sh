#!/usr/bin/env bash
# Launch the keyboard flight controller inside the project venv.
#
#   ./fly.sh
#   ./fly.sh --master=udpout:127.0.0.1:14550 --max-speed=2
#   ./fly.sh --self-test
#
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
    echo "No .venv found. Create it first (see README.md)." >&2
    exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate
exec python keyboard_control.py "$@"
