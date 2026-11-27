#!/usr/bin/env python3
"""Stationary bench capture & channel-rate auditor for crazybridge telemetry.

Two modes:
1. Live ROS 2 bench test (no takeoff, motors never armed):
   Records ``odom``, ``setpoint``, ``thrust``, ``torque``, ``pos_err``, ``imu``,
   ``ekf_diag``, ``battery`` and ``imu_bias`` for ``--duration`` seconds from a
   running ``crazybridge`` node, verifies ``EKF_OOT_FLAG_BIAS_READY`` + stillness
   + ``imu - imu_bias ≈ [0, 0, +9.81] m/s^2, [0, 0, 0] rad/s``, writes a full
   run folder (``traj.npz``, ``flight_telemetry.csv``, ``metrics.json``), and
   prints the measured channel-rate table.

2. Offline ``traj.npz`` audit (``--from-npz <path/to/traj.npz>``):
   Inspects an existing ``traj.npz`` (and sibling ``flight_telemetry.csv`` /
   ``metrics.json`` if present) and prints the channel-rate table, CSV header,
   and calibrated IMU bias values.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
from time import sleep

import numpy as np

# Allow running directly without sourcing colcon when auditing --from-npz.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from crazybridge import metrics  # noqa: E402

NOMINAL_RATES_HZ = {
    "odom": 40.0,       # state_pos (20 Hz) + state_quat (20 Hz) -> 40 msg/s
    "setpoint": 20.0,   # log_period_ms = 50 ms
    "thrust": None,     # depends on oot_log_period_ms (10 ms -> 100 Hz, 100 ms -> 10 Hz)
    "torque": None,
    "imu": 50.0,        # imu_log_period_ms = 20 ms
    "ekf_diag": 50.0,   # ekf_log_period_ms = 20 ms
    "battery": 10.0,    # extra_period = 100 ms
    "imu_bias": 2.0,    # bias_log_period_ms = 500 ms
}

REPORT_CHANNELS = (
    "odom",
    "setpoint",
    "thrust",
    "torque",
    "imu",
    "ekf_diag",
    "battery",
    "imu_bias",
)


def print_audit_report(run_folder_or_npz: str) -> int:
    npz_path = (
        run_folder_or_npz
        if run_folder_or_npz.endswith(".npz")
        else os.path.join(run_folder_or_npz, metrics.FILE_TRAJ_NPZ)
    )
    if not os.path.isfile(npz_path):
        print(f"ERROR: {npz_path} not found", file=sys.stderr)
        return 1

    run_folder = os.path.dirname(os.path.abspath(npz_path))
    csv_path = os.path.join(run_folder, metrics.FILE_TELEMETRY_CSV)
    json_path = os.path.join(run_folder, metrics.FILE_METRICS_JSON)

    z = np.load(npz_path, allow_pickle=True)
    traj = {
        "odom": z["odom"] if "odom" in z.files else np.empty((0, 11)),
        "setpoint": z["setpoint"] if "setpoint" in z.files else np.empty((0, 4)),
    }
    samples = {
        k: z[k]
        for k in (
            "raw_mocap",
            "thrust",
            "torque",
            "pos_err",
            "imu",
            "ekf_diag",
            "ekf_adapt",
            "battery",
            "imu_bias",
        )
        if k in z.files
    }
    rates = metrics.compute_channel_rates(traj, samples)

    print("=" * 78)
    print(f"BENCH / TELEMETRY AUDIT: {npz_path}")
    print("=" * 78)
    print(f"{'Canal':<12} | {'Muestras':>8} | {'Ventana (s)':>11} | {'Tasa real (Hz)':>14} | {'dt med (ms)':>11} | {'Nominal':>10}")
    print("-" * 78)
    for ch in REPORT_CHANNELS:
        info = rates[ch]
        nom = NOMINAL_RATES_HZ.get(ch)
        nom_str = f"{nom:.1f} Hz" if nom is not None else "oot_period"
        dt_str = f"{info['median_dt_ms']:.2f}" if info["median_dt_ms"] is not None else "n/a"
        print(
            f"{ch:<12} | {info['samples']:>8d} | {info['span_s']:>11.3f} | "
            f"{info['rate_hz']:>14.2f} | {dt_str:>11} | {nom_str:>10}"
        )
    print("-" * 78)

    bias_summary = metrics._extract_imu_bias_summary(
        samples.get("imu_bias", np.empty((0, 7)))
    )
    if bias_summary is not None:
        print(
            "IMU Bias calibrado (FLU): "
            f"acc = [{bias_summary['ax']:+.5f}, {bias_summary['ay']:+.5f}, {bias_summary['az']:+.5f}] m/s^2 | "
            f"gyro = [{bias_summary['gx']:+.6f}, {bias_summary['gy']:+.6f}, {bias_summary['gz']:+.6f}] rad/s"
        )
        imu = samples.get("imu", np.empty((0, 7)))
        if imu.ndim == 2 and imu.shape[0] > 0:
            bvec = np.array(
                [
                    bias_summary["ax"],
                    bias_summary["ay"],
                    bias_summary["az"],
                    bias_summary["gx"],
                    bias_summary["gy"],
                    bias_summary["gz"],
                ]
            )
            corr_mean = np.mean(imu[:, 1:7] - bvec, axis=0)
            print(
                "Promedio (imu - imu_bias): "
                f"a_corr = [{corr_mean[0]:+.4f}, {corr_mean[1]:+.4f}, {corr_mean[2]:+.4f}] m/s^2 | "
                f"g_corr = [{corr_mean[3]:+.5f}, {corr_mean[4]:+.5f}, {corr_mean[5]:+.5f}] rad/s"
            )
    else:
        print("IMU Bias calibrado: None (NaN en CSV)")

    if os.path.isfile(json_path):
        with open(json_path) as f:
            meta = json.load(f)
        print(
            f"metrics.json: estimator_mode={meta.get('estimator_mode')!r}, "
            f"adapt_mode={meta.get('adapt_mode')!r}, "
            f"ekf_oot_detected={meta.get('ekf_oot_detected')!r}"
        )

    if os.path.isfile(csv_path):
        with open(csv_path, newline="") as f:
            header = next(csv.reader(f))
        print(f"CSV header ({len(header)} cols): {','.join(header)}")
    print("=" * 78)
    return 0


def run_live_bench(args: argparse.Namespace) -> int:
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from crazybridge.controller_test import ControllerTest

    if args.set_params:
        subprocess.run(
            ["ros2", "param", "set", "/crazybridge", "estimator_mode", args.estimator_mode],
            check=True,
        )
        subprocess.run(
            ["ros2", "param", "set", "/crazybridge", "adapt_mode", args.adapt_mode],
            check=True,
        )

    rclpy.init()
    node = ControllerTest()
    node.set_parameters(
        [
            rclpy.parameter.Parameter("config_name", value=args.config_name),
            rclpy.parameter.Parameter("output_dir", value=args.output_dir),
            rclpy.parameter.Parameter("estimator_mode", value=args.estimator_mode),
            rclpy.parameter.Parameter("adapt_mode", value=args.adapt_mode),
            rclpy.parameter.Parameter("require_imu_bias", value=args.require_imu_bias),
        ]
    )
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        node.get_logger().info("Waiting for /crazybridge odometry on bench (no takeoff)...")
        node.wait_until_ready(timeout_s=args.timeout, require_odom=True)
        node.verify_pre_takeoff_bias_and_stillness()

        # Clear buffers so the rate window covers the steady bench interval cleanly,
        # while retaining the latest calibrated imu_bias sample at t_start if needed.
        t0 = node._now()
        with node._lock:
            seed_bias = list(node._imu_bias[-1]) if node._imu_bias else None
            node._odom.clear()
            node._setpoint.clear()
            node._thrust.clear()
            node._torque.clear()
            node._pos_err.clear()
            node._imu.clear()
            node._imu_bias.clear()
            node._ekf_diag.clear()
            node._ekf_adapt.clear()
            node._battery.clear()
            if seed_bias is not None:
                seed_bias[0] = t0
                node._imu_bias.append(seed_bias)

        node.get_logger().info(f"Recording stationary bench telemetry for {args.duration:.1f} s...")
        sleep(args.duration)
        t1 = node._now()
        d = (t1 - t0) / 4.0
        ts = {
            "t_start": t0,
            "t_takeoff_end": t0 + d,
            "t_regulate_end": t0 + 2.0 * d,
            "t_circle_end": t0 + 3.0 * d,
            "t_land_end": t1,
        }
        node.report(ts)
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    return print_audit_report(metrics.run_dir(args.output_dir, args.config_name))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from-npz", default="", help="Audit an existing traj.npz or run folder.")
    p.add_argument("--output-dir", default="/tmp/bench_check", help="Parent output folder.")
    p.add_argument("--config-name", default="bench_oot_static", help="Run folder name.")
    p.add_argument(
        "--estimator-mode",
        choices=("stock", "oot"),
        default="oot",
        help="Explicit estimator_mode for the bench check.",
    )
    p.add_argument(
        "--adapt-mode",
        choices=("static", "mdp", "smdp"),
        default="static",
        help="Explicit adapt_mode for the bench check (default: static).",
    )
    p.add_argument("--duration", type=float, default=6.0, help="Bench capture duration (s).")
    p.add_argument("--timeout", type=float, default=15.0, help="Connection wait timeout (s).")
    p.add_argument(
        "--no-set-params",
        dest="set_params",
        action="store_false",
        help="Skip `ros2 param set /crazybridge ...` before recording.",
    )
    p.add_argument(
        "--no-require-imu-bias",
        dest="require_imu_bias",
        action="store_false",
        help="Allow bench check on firmware without ekfOot/ekfBias.",
    )
    args = p.parse_args()
    if args.from_npz:
        return print_audit_report(args.from_npz)
    return run_live_bench(args)


if __name__ == "__main__":
    raise SystemExit(main())
