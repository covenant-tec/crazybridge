#!/usr/bin/env bash
# Run the original circular controller_test sequence against an already-running
# crazybridge node and save flight_telemetry.csv directly to the Wiwi drive.
#
# Usage (inside Ubuntu 24 Distrobox):
#   ./run_ekf_test.sh [config_name] [extra ros2 args...]
#
# Examples:
#   ./run_ekf_test.sh vuelo_stock_01
#   ESTIMATOR_MODE=oot ADAPT_MODE=static ./run_ekf_test.sh vuelo_ekf_static_01
#   ESTIMATOR_MODE=oot ADAPT_MODE=mdp    ./run_ekf_test.sh vuelo_ekf_mdp_01
#   ESTIMATOR_MODE=oot ADAPT_MODE=smdp   ./run_ekf_test.sh vuelo_ekf_smdp_01

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

CONFIG_NAME="${1:-vuelo_ekf_01}"
if [[ $# -ge 1 ]]; then
    shift
fi

OUTPUT_DIR="${OUTPUT_DIR:-/run/media/mojarras7/Wiwi/smdp-adaptive-ekf-quadrotor/vuelos_ekf}"

cd "${WORKSPACE_ROOT}"
set +u
# shellcheck source=/dev/null
source "${WORKSPACE_ROOT}/source_all.sh"
set -u

EXTRA_TEST_ARGS=()
if [[ -n "${ESTIMATOR_MODE:-}" ]]; then
    ros2 param set /crazybridge estimator_mode "${ESTIMATOR_MODE}"
    EXTRA_TEST_ARGS+=(-p "estimator_mode:=${ESTIMATOR_MODE}")
fi
if [[ -n "${ADAPT_MODE:-}" ]]; then
    ros2 param set /crazybridge adapt_mode "${ADAPT_MODE}"
    EXTRA_TEST_ARGS+=(-p "adapt_mode:=${ADAPT_MODE}")
fi

exec ros2 run crazybridge controller_test --ros-args \
    -p "config_name:=${CONFIG_NAME}" \
    -p "output_dir:=${OUTPUT_DIR}" \
    -p "tracking_mode:=rigid_body" \
    "${EXTRA_TEST_ARGS[@]}" \
    "$@"
