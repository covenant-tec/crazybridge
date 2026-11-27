#!/usr/bin/env bash
# Run the original circular controller_test sequence against an already-running
# crazybridge node and save flight_telemetry.csv directly to the Wiwi drive.
#
# Usage (inside Ubuntu 24 Distrobox):
#   ./run_ekf_test.sh [config_name] [extra ros2 args...]
#
# Example:
#   ./run_ekf_test.sh vuelo_ekf_01

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

exec ros2 run crazybridge controller_test --ros-args \
    -p "config_name:=${CONFIG_NAME}" \
    -p "output_dir:=${OUTPUT_DIR}" \
    -p "tracking_mode:=rigid_body" \
    "$@"
