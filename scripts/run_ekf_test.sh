#!/usr/bin/env bash
# Run one circular comparison flight against an already-running crazybridge node
# and export flight_telemetry.csv + traj.npz + metrics.json to OUTPUT_DIR.
#
# For the 10-flight comparison session (5 stock + 5 oot, interleaved), BOTH
# estimator_mode (stock|oot) and adapt_mode (default: static) are set explicitly
# on every run -- never relying on node defaults.
#
# Usage (inside Ubuntu 24 Distrobox):
#   ./run_ekf_test.sh <config_name> <stock|oot> [extra ros2 args...]
#   ESTIMATOR_MODE=<stock|oot> ADAPT_MODE=static ./run_ekf_test.sh <config_name> [extra ros2 args...]
#
# Interleaved 10-flight session example (all with adapt_mode=static, IMU @ 50 Hz):
#   ./run_ekf_test.sh vuelo_01_stock stock
#   ./run_ekf_test.sh vuelo_02_oot   oot
#   ./run_ekf_test.sh vuelo_03_stock stock
#   ./run_ekf_test.sh vuelo_04_oot   oot
#   ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <config_name> <stock|oot> [extra ros2 args...]" >&2
    exit 2
fi

CONFIG_NAME="$1"
shift

if [[ $# -ge 1 && ( "$1" == "stock" || "$1" == "oot" ) ]]; then
    ESTIMATOR_MODE="$1"
    shift
fi

ESTIMATOR_MODE="${ESTIMATOR_MODE:-}"
if [[ "${ESTIMATOR_MODE}" != "stock" && "${ESTIMATOR_MODE}" != "oot" ]]; then
    echo "ERROR: estimator_mode must be explicitly specified as 'stock' or 'oot' (got '${ESTIMATOR_MODE}')." >&2
    echo "Example: $0 ${CONFIG_NAME} stock" >&2
    exit 2
fi

# Comparison flights MUST use adapt_mode=static unless explicitly overridden later.
ADAPT_MODE="${ADAPT_MODE:-static}"
OUTPUT_DIR="${OUTPUT_DIR:-/run/media/mojarras7/Wiwi/smdp-adaptive-ekf-quadrotor/vuelos_ekf}"

cd "${WORKSPACE_ROOT}"
set +u
# shellcheck source=/dev/null
source "${WORKSPACE_ROOT}/source_all.sh"
set -u

echo "[run_ekf_test] Configuring /crazybridge -> estimator_mode=${ESTIMATOR_MODE}, adapt_mode=${ADAPT_MODE}"
ros2 param set /crazybridge estimator_mode "${ESTIMATOR_MODE}"
ros2 param set /crazybridge adapt_mode "${ADAPT_MODE}"

exec ros2 run crazybridge controller_test --ros-args \
    -p "config_name:=${CONFIG_NAME}" \
    -p "output_dir:=${OUTPUT_DIR}" \
    -p "tracking_mode:=rigid_body" \
    -p "estimator_mode:=${ESTIMATOR_MODE}" \
    -p "adapt_mode:=${ADAPT_MODE}" \
    "$@"
