#!/usr/bin/env bash
#
# Simple smoke test for MAVProxy.
#
# Runs the same command you would use manually:
#   mavproxy.py --master=udpout:127.0.0.1:14550
# then feeds it "exit" so it does not hang forever.
#
# Usage:
#   ./test_mavproxy.sh

set -u

MASTER="${MASTER:-udpout:127.0.0.1:14550}"
RUN_SECONDS="${RUN_SECONDS:-15}"

cd "$(dirname "$0")" || exit 1

if [ -d .venv ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

if ! command -v mavproxy.py >/dev/null 2>&1; then
    echo "FAIL: mavproxy.py not found. Did the venv install succeed?"
    exit 1
fi

echo "==> MAVProxy version"
mavproxy.py --version 2>/dev/null | grep -i "MAVProxy Version" || {
    echo "FAIL: could not read MAVProxy version"
    exit 1
}

echo
echo "==> Running: mavproxy.py --master=${MASTER}  (auto-exit after ${RUN_SECONDS}s)"
echo

OUTPUT="$(printf 'exit\n' | timeout "${RUN_SECONDS}" mavproxy.py --master="${MASTER}" 2>&1)"
STATUS=$?

echo "$OUTPUT"
echo

# timeout returns 124 when it had to kill the process.
if [ "$STATUS" -eq 124 ]; then
    echo "FAIL: MAVProxy did not exit within ${RUN_SECONDS}s"
    exit 1
fi

# A successful startup connects the link and reaches the MAV> prompt.
if echo "$OUTPUT" | grep -q "Connect ${MASTER}"; then
    echo "PASS: MAVProxy started and opened the master link (${MASTER})"
    exit 0
fi

echo "FAIL: MAVProxy did not report opening the master link"
exit 1
