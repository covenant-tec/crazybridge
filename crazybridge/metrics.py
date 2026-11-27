"""Node-agnostic metrics + report writing for the controller comparison test.

All functions take plain numpy arrays / dicts (no ROS or node dependency) so
they are easy to test and reuse. Sample arrays are laid out as:

* ``thrust``  -> shape (N, 2): columns ``[t, thrust]``
* ``torque``  -> shape (N, 4): columns ``[t, x, y, z]``
* ``pos_err`` -> shape (N, 4): columns ``[t, ex, ey, ez]``

Timestamps are seconds (any consistent clock). Integrals are trapezoidal.
The headline comparison numbers are the *circle* phase ``int_thrust_abs`` /
``int_torque_norm`` (control effort) and ``int_err_norm`` (tracking error).

:func:`write_reports` lays a run out as::

    <output_dir>/<config_name>/metrics.json   # metrics + the gains flown
                              /metrics.csv
                              /traj.npz
                              /pid.conf       # verbatim copy of the gain file
"""
from __future__ import annotations

import csv
import json
import os
import shutil

import numpy as np

# Each run gets its own ``<output_dir>/<config_name>/`` folder, and the files
# inside are named identically across runs so tooling never has to interpolate
# the config name into a filename.
FILE_METRICS_JSON = 'metrics.json'
FILE_METRICS_CSV = 'metrics.csv'
FILE_TRAJ_NPZ = 'traj.npz'
FILE_PID_CONF = 'pid.conf'
FILE_ADAPT_CONF = 'adapt_policies.conf'
FILE_TELEMETRY_CSV = 'flight_telemetry.csv'

# Calibrated stationary IMU bias columns always present in flight_telemetry.csv
# (m/s^2 and rad/s in FLU body frame, matching imu_* columns; NaN if unavailable).
IMU_BIAS_CSV_HEADERS = (
    'imu_bias_ax',
    'imu_bias_ay',
    'imu_bias_az',
    'imu_bias_gx',
    'imu_bias_gy',
    'imu_bias_gz',
)
IMU_BIAS_NCOLS = len(IMU_BIAS_CSV_HEADERS)

# Optional diagnostic columns appended to flight_telemetry.csv ONLY when an
# Out-Of-Tree EKF (CRTP group ``ekfOot``) was detected during the flight.
EKF_DIAG_CSV_HEADERS = (
    'ekf_d_mahal',
    'ekf_tr_p',
    'ekf_chol_fails',
    'ekf_nan_resets',
    'ekf_flags',
    'ekf_rt_pred',
    'ekf_rt_upd',
)
EKF_DIAG_NCOLS = len(EKF_DIAG_CSV_HEADERS)
_EKF_DIAG_DISCRETE_COLS = (2, 3, 4)

# Optional adaptive decision columns appended ONLY when ``ekfAdapt`` streamed.
EKF_ADAPT_CSV_HEADERS = (
    'ekf_tr_q',
    'ekf_tr_r',
    'ekf_decisions',
    'ekf_qr_mods',
    'ekf_tau',
    'ekf_state2d',
    'ekf_action',
    'ekf_mode',
)
EKF_ADAPT_NCOLS = len(EKF_ADAPT_CSV_HEADERS)
_EKF_ADAPT_DISCRETE_COLS = (2, 3, 4, 5, 6, 7)

# numpy 2.0 renamed trapz -> trapezoid (and later removed the old name).
try:
    from numpy import trapezoid as _np_trapz
except ImportError:  # numpy < 2.0
    from numpy import trapz as _np_trapz

# Phase segments reported. Each is (name, start_key, end_key) into the
# timestamps dict produced by the flight sequence.
SEGMENTS = (
    ('takeoff', 't_start', 't_takeoff_end'),
    ('regulate', 't_takeoff_end', 't_regulate_end'),
    ('circle', 't_regulate_end', 't_circle_end'),
    ('land', 't_circle_end', 't_land_end'),
    ('total', 't_start', 't_land_end'),
)


def _slice(samples: np.ndarray, a: float, b: float) -> np.ndarray:
    """Rows of `samples` (column 0 = time) with timestamp in [a, b]."""
    if samples.size == 0:
        return samples
    mask = (samples[:, 0] >= a) & (samples[:, 0] <= b)
    return samples[mask]


def _trapz_abs(t: np.ndarray, v: np.ndarray) -> float:
    if t.size < 2:
        return 0.0
    return float(_np_trapz(np.abs(v), t))


def _trapz(t: np.ndarray, v: np.ndarray) -> float:
    if t.size < 2:
        return 0.0
    return float(_np_trapz(v, t))


def _segment_metrics(a: float, b: float, thrust: np.ndarray,
                     torque: np.ndarray, pos_err: np.ndarray) -> dict:
    seg: dict[str, float] = {'t_start': a, 't_end': b, 'duration_s': b - a}

    # --- control signals ---
    th = _slice(thrust, a, b)
    if th.size:
        seg['int_thrust_abs'] = _trapz_abs(th[:, 0], th[:, 1])
        seg['int_thrust_sq'] = _trapz(th[:, 0], th[:, 1] ** 2)
    else:
        seg['int_thrust_abs'] = seg['int_thrust_sq'] = 0.0

    tq = _slice(torque, a, b)
    if tq.size:
        seg['int_torque_x_abs'] = _trapz_abs(tq[:, 0], tq[:, 1])
        seg['int_torque_y_abs'] = _trapz_abs(tq[:, 0], tq[:, 2])
        seg['int_torque_z_abs'] = _trapz_abs(tq[:, 0], tq[:, 3])
        tq_norm = np.linalg.norm(tq[:, 1:4], axis=1)
        seg['int_torque_norm'] = _trapz(tq[:, 0], tq_norm)
        seg['int_torque_norm_sq'] = _trapz(tq[:, 0], tq_norm ** 2)
    else:
        for k in ('int_torque_x_abs', 'int_torque_y_abs', 'int_torque_z_abs',
                  'int_torque_norm', 'int_torque_norm_sq'):
            seg[k] = 0.0

    # --- position error (IAE-style) ---
    pe = _slice(pos_err, a, b)
    if pe.size:
        seg['int_err_x_abs'] = _trapz_abs(pe[:, 0], pe[:, 1])
        seg['int_err_y_abs'] = _trapz_abs(pe[:, 0], pe[:, 2])
        seg['int_err_z_abs'] = _trapz_abs(pe[:, 0], pe[:, 3])
        e_norm = np.linalg.norm(pe[:, 1:4], axis=1)
        seg['int_err_norm'] = _trapz(pe[:, 0], e_norm)     # headline metric
        seg['rmse_norm'] = float(np.sqrt(np.mean(e_norm ** 2)))
        seg['max_err_norm'] = float(np.max(e_norm))
    else:
        for k in ('int_err_x_abs', 'int_err_y_abs', 'int_err_z_abs',
                  'int_err_norm', 'rmse_norm', 'max_err_norm'):
            seg[k] = 0.0

    return seg


def compute_metrics(samples: dict, ts: dict) -> dict:
    """Per-phase and total integral metrics.

    `samples` maps 'thrust'/'torque'/'pos_err' to their arrays; `ts` maps the
    segment boundary keys to timestamps (seconds).
    """
    thrust = np.asarray(samples.get('thrust', np.empty((0, 2))), dtype=float)
    torque = np.asarray(samples.get('torque', np.empty((0, 4))), dtype=float)
    pos_err = np.asarray(samples.get('pos_err', np.empty((0, 4))), dtype=float)

    metrics: dict[str, dict] = {}
    for name, a_key, b_key in SEGMENTS:
        metrics[name] = _segment_metrics(
            ts[a_key], ts[b_key], thrust, torque, pos_err)
    return metrics


def flatten_row(config: str, metrics: dict) -> dict:
    """Flatten nested metrics into a single ``seg.metric`` -> value row."""
    row = {'config': config}
    for seg_name, seg in metrics.items():
        for k, v in seg.items():
            row[f'{seg_name}.{k}'] = v
    return row


def run_dir(out_dir: str, config: str) -> str:
    """Directory holding one run's artefacts: ``<out_dir>/<config>/``."""
    return os.path.join(out_dir, config)


def _interp_array(arr: np.ndarray, t_eval: np.ndarray, ncols: int) -> np.ndarray:
    """Linearly interpolate columns 1..ncols of `arr` (col 0 = timestamp) onto `t_eval`."""
    if arr is None or arr.size == 0 or len(arr.shape) != 2 or arr.shape[0] < 2:
        return np.zeros((len(t_eval), ncols))
    t_source = arr[:, 0]
    unique_mask = np.concatenate(([True], np.diff(t_source) > 1e-9))
    arr_clean = arr[unique_mask]
    if arr_clean.shape[0] < 2:
        return np.zeros((len(t_eval), ncols))

    out = np.zeros((len(t_eval), ncols))
    actual_cols = min(ncols, arr_clean.shape[1] - 1)
    for i in range(actual_cols):
        out[:, i] = np.interp(t_eval, arr_clean[:, 0], arr_clean[:, i + 1])
    return out


def _resample_mixed_block(
    arr: np.ndarray, t_eval: np.ndarray, ncols: int, discrete_cols: tuple[int, ...]
) -> np.ndarray:
    """Resample a multi-column block onto ``t_eval`` with ZOH for discrete cols."""
    if arr is None or arr.size == 0 or len(arr.shape) != 2 or arr.shape[0] == 0:
        return np.zeros((len(t_eval), ncols))
    t_source = arr[:, 0]
    unique_mask = np.concatenate(([True], np.diff(t_source) > 1e-9))
    arr_clean = arr[unique_mask]
    actual_cols = min(ncols, arr_clean.shape[1] - 1)
    if actual_cols <= 0:
        return np.zeros((len(t_eval), ncols))

    if arr_clean.shape[0] == 1:
        out = np.zeros((len(t_eval), ncols))
        out[:, :actual_cols] = arr_clean[0, 1 : 1 + actual_cols]
        return out

    out = _interp_array(arr_clean, t_eval, ncols)
    idx = np.clip(
        np.searchsorted(arr_clean[:, 0], t_eval, side="right") - 1,
        0,
        arr_clean.shape[0] - 1,
    )
    for col in discrete_cols:
        if col < actual_cols:
            out[:, col] = arr_clean[idx, col + 1]
    return out


def _resample_ekf_diag(arr: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
    """Resample ``ekf_diag`` onto ``t_eval`` using ZOH for integer flags/counters."""
    return _resample_mixed_block(arr, t_eval, EKF_DIAG_NCOLS, _EKF_DIAG_DISCRETE_COLS)


def _resample_ekf_adapt(arr: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
    """Resample ``ekf_adapt`` onto ``t_eval`` using ZOH for discrete states/actions."""
    return _resample_mixed_block(arr, t_eval, EKF_ADAPT_NCOLS, _EKF_ADAPT_DISCRETE_COLS)


def _resample_imu_bias(arr: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
    """Resample ``imu_bias`` onto ``t_eval`` with ZOH; return ``NaN`` (never ``0``) if absent."""
    n = len(t_eval)
    if (
        arr is None
        or arr.size == 0
        or arr.ndim != 2
        or arr.shape[1] < 1 + IMU_BIAS_NCOLS
    ):
        return np.full((n, IMU_BIAS_NCOLS), np.nan)
    finite_mask = np.all(np.isfinite(arr[:, : 1 + IMU_BIAS_NCOLS]), axis=1)
    arr_valid = arr[finite_mask]
    if arr_valid.shape[0] == 0:
        return np.full((n, IMU_BIAS_NCOLS), np.nan)
    unique_mask = np.concatenate(([True], np.diff(arr_valid[:, 0]) > 1e-9))
    arr_clean = arr_valid[unique_mask]
    if arr_clean.shape[0] == 1:
        return np.tile(arr_clean[0, 1 : 1 + IMU_BIAS_NCOLS], (n, 1))
    idx = np.clip(
        np.searchsorted(arr_clean[:, 0], t_eval, side="right") - 1,
        0,
        arr_clean.shape[0] - 1,
    )
    return arr_clean[idx, 1 : 1 + IMU_BIAS_NCOLS].copy()


def _extract_imu_bias_summary(arr: np.ndarray) -> dict[str, float] | None:
    """Extract calibrated bias dict from ``imu_bias`` sample array, or ``None`` if absent."""
    if (
        arr is None
        or arr.size == 0
        or arr.ndim != 2
        or arr.shape[1] < 1 + IMU_BIAS_NCOLS
    ):
        return None
    finite_mask = np.all(np.isfinite(arr[:, : 1 + IMU_BIAS_NCOLS]), axis=1)
    arr_valid = arr[finite_mask]
    if arr_valid.shape[0] == 0:
        return None
    b = arr_valid[-1, 1 : 1 + IMU_BIAS_NCOLS]
    return {
        "ax": float(b[0]),
        "ay": float(b[1]),
        "az": float(b[2]),
        "gx": float(b[3]),
        "gy": float(b[4]),
        "gz": float(b[5]),
    }


def compute_channel_rates(traj: dict, samples: dict | None = None) -> dict[str, dict]:
    """Compute effective packet count, span, and Hz rate for each logged channel."""
    samples_dict = samples or {}
    channels = (
        ("odom", traj.get("odom")),
        ("setpoint", traj.get("setpoint")),
        ("raw_mocap", samples_dict.get("raw_mocap")),
        ("thrust", samples_dict.get("thrust")),
        ("torque", samples_dict.get("torque")),
        ("pos_err", samples_dict.get("pos_err")),
        ("imu", samples_dict.get("imu")),
        ("ekf_diag", samples_dict.get("ekf_diag")),
        ("ekf_adapt", samples_dict.get("ekf_adapt")),
        ("battery", samples_dict.get("battery")),
        ("imu_bias", samples_dict.get("imu_bias")),
    )
    out: dict[str, dict] = {}
    for name, raw in channels:
        arr = np.asarray(raw) if raw is not None else np.empty((0, 0))
        n = int(arr.shape[0]) if arr.ndim == 2 else 0
        if n >= 2 and arr.shape[1] >= 1:
            t = arr[:, 0].astype(float)
            span_s = float(t[-1] - t[0])
            dt_pos = np.diff(t)
            dt_pos = dt_pos[dt_pos > 1e-9]
            rate_hz = float((n - 1) / span_s) if span_s > 1e-6 else 0.0
            median_dt_ms = float(np.median(dt_pos) * 1000.0) if dt_pos.size > 0 else None
        else:
            span_s = 0.0
            rate_hz = 0.0
            median_dt_ms = None
        out[name] = {
            "samples": n,
            "span_s": round(span_s, 4),
            "rate_hz": round(rate_hz, 3),
            "median_dt_ms": round(median_dt_ms, 3) if median_dt_ms is not None else None,
        }
    return out


def write_telemetry_csv(
    dest: str, ts: dict, traj: dict, samples: dict | None, dt: float = 0.02
) -> str:
    """Write time-synchronized telemetry dataset to ``flight_telemetry.csv``."""
    t_start = float(ts.get("t_start", 0.0))
    t_end = float(ts.get("t_land_end", t_start + 1.0))
    if t_end <= t_start:
        t_end = t_start + 1.0

    t_eval = np.arange(t_start, t_end, dt)
    if len(t_eval) == 0:
        t_eval = np.array([t_start])

    samples_dict = samples or {}
    raw_mocap = np.asarray(samples_dict.get("raw_mocap", np.empty((0, 8))), dtype=float)
    imu = np.asarray(samples_dict.get("imu", np.empty((0, 7))), dtype=float)
    thrust = np.asarray(samples_dict.get("thrust", np.empty((0, 2))), dtype=float)
    torque = np.asarray(samples_dict.get("torque", np.empty((0, 4))), dtype=float)
    pos_err = np.asarray(samples_dict.get("pos_err", np.empty((0, 4))), dtype=float)
    ekf_diag = np.asarray(
        samples_dict.get("ekf_diag", np.empty((0, 1 + EKF_DIAG_NCOLS))), dtype=float
    )
    ekf_adapt = np.asarray(
        samples_dict.get("ekf_adapt", np.empty((0, 1 + EKF_ADAPT_NCOLS))), dtype=float
    )
    battery = np.asarray(samples_dict.get("battery", np.empty((0, 2))), dtype=float)
    imu_bias = np.asarray(
        samples_dict.get("imu_bias", np.empty((0, 1 + IMU_BIAS_NCOLS))), dtype=float
    )
    odom = np.asarray(traj.get("odom", np.empty((0, 11))), dtype=float)
    setpoint = np.asarray(traj.get("setpoint", np.empty((0, 4))), dtype=float)

    interp_raw = _interp_array(raw_mocap, t_eval, 7)
    interp_odom = _interp_array(odom, t_eval, 10)
    interp_imu = _interp_array(imu, t_eval, 6)
    interp_sp = _interp_array(setpoint, t_eval, 3)
    interp_thrust = _interp_array(thrust, t_eval, 1)
    interp_torque = _interp_array(torque, t_eval, 3)
    interp_err = _interp_array(pos_err, t_eval, 3)
    # Battery voltage [V]; NaN (not 0) when the bridge published no samples.
    if battery.ndim == 2 and battery.shape[0] >= 2:
        interp_vbat = _interp_array(battery, t_eval, 1)
    elif battery.ndim == 2 and battery.shape[0] == 1 and np.isfinite(battery[0, 1]):
        interp_vbat = np.full((len(t_eval), 1), float(battery[0, 1]))
    else:
        interp_vbat = np.full((len(t_eval), 1), np.nan)
    interp_imu_bias = _resample_imu_bias(imu_bias, t_eval)

    rel_t = t_eval - t_eval[0]
    columns = [
        rel_t,
        t_eval,
        interp_raw,
        interp_odom,
        interp_imu,
        interp_sp,
        interp_thrust,
        interp_torque,
        interp_err,
        interp_vbat,
        interp_imu_bias,
    ]

    headers = [
        "time_rel",
        "timestamp",
        "raw_x",
        "raw_y",
        "raw_z",
        "raw_qx",
        "raw_qy",
        "raw_qz",
        "raw_qw",
        "fused_x",
        "fused_y",
        "fused_z",
        "fused_qx",
        "fused_qy",
        "fused_qz",
        "fused_qw",
        "fused_vx",
        "fused_vy",
        "fused_vz",
        "imu_ax",
        "imu_ay",
        "imu_az",
        "imu_gx",
        "imu_gy",
        "imu_gz",
        "sp_x",
        "sp_y",
        "sp_z",
        "thrust",
        "torque_x",
        "torque_y",
        "torque_z",
        "pos_err_x",
        "pos_err_y",
        "pos_err_z",
        "vbat",
        *IMU_BIAS_CSV_HEADERS,
    ]

    if ekf_diag.ndim == 2 and ekf_diag.shape[0] > 0:
        columns.append(_resample_ekf_diag(ekf_diag, t_eval))
        headers.extend(EKF_DIAG_CSV_HEADERS)

    if ekf_adapt.ndim == 2 and ekf_adapt.shape[0] > 0:
        columns.append(_resample_ekf_adapt(ekf_adapt, t_eval))
        headers.extend(EKF_ADAPT_CSV_HEADERS)

    data = np.column_stack(columns)

    csv_path = os.path.join(dest, FILE_TELEMETRY_CSV)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)

    return csv_path


def write_reports(
    out_dir: str,
    config: str,
    ts: dict,
    metrics: dict,
    traj: dict,
    samples: dict | None = None,
    gains: dict | None = None,
    pid_conf_path: str | None = None,
    estimator_mode: str | None = None,
    adapt_mode: str | None = None,
    adapt_policies: dict | None = None,
    adapt_conf_path: str | None = None,
) -> dict:
    """Write one run's artefacts into ``<out_dir>/<config>/``.

    Filenames are fixed (``metrics.json``, ``metrics.csv``, ``traj.npz``,
    ``flight_telemetry.csv``, ``pid.conf``) so they are identical across runs;
    the *directory* carries the config name. ``gains`` (the parsed pid.conf, flat
    ``name -> value``) and a verbatim copy of ``pid_conf_path`` are recorded so a
    run's numbers can always be traced back to the gains that produced them.

    Returns the flattened metric row (useful for logging / aggregation).
    """
    dest = run_dir(out_dir, config)
    os.makedirs(dest, exist_ok=True)

    samples_dict = samples or {}
    raw_mocap = np.asarray(samples_dict.get("raw_mocap", np.empty((0, 8))), dtype=float)
    imu = np.asarray(samples_dict.get("imu", np.empty((0, 7))), dtype=float)
    imu_bias = np.asarray(
        samples_dict.get("imu_bias", np.empty((0, 1 + IMU_BIAS_NCOLS))), dtype=float
    )
    thrust = np.asarray(samples_dict.get("thrust", np.empty((0, 2))), dtype=float)
    torque = np.asarray(samples_dict.get("torque", np.empty((0, 4))), dtype=float)
    pos_err = np.asarray(samples_dict.get("pos_err", np.empty((0, 4))), dtype=float)
    ekf_diag = np.asarray(
        samples_dict.get("ekf_diag", np.empty((0, 1 + EKF_DIAG_NCOLS))), dtype=float
    )
    ekf_adapt = np.asarray(
        samples_dict.get("ekf_adapt", np.empty((0, 1 + EKF_ADAPT_NCOLS))), dtype=float
    )
    has_ekf_diag = bool(ekf_diag.ndim == 2 and ekf_diag.shape[0] > 0)
    has_ekf_adapt = bool(ekf_adapt.ndim == 2 and ekf_adapt.shape[0] > 0)
    imu_bias_summary = _extract_imu_bias_summary(imu_bias)
    channel_rates = compute_channel_rates(traj, samples_dict)

    report = {
        "config": config,
        "estimator_mode": estimator_mode,
        "adapt_mode": adapt_mode,
        "ekf_oot_detected": has_ekf_diag,
        "imu_bias": imu_bias_summary,
        "channel_rates": channel_rates,
        "timestamps": ts,
        "metrics": metrics,
    }
    if adapt_policies is not None:
        report["adapt_policies"] = adapt_policies
    if adapt_conf_path:
        report["adapt_conf_path"] = os.path.abspath(adapt_conf_path)
    if gains is not None:
        report["gains"] = gains
    if pid_conf_path:
        report["pid_conf_path"] = os.path.abspath(pid_conf_path)
    with open(os.path.join(dest, FILE_METRICS_JSON), "w") as f:
        json.dump(report, f, indent=2)

    row = flatten_row(config, metrics)
    with open(os.path.join(dest, FILE_METRICS_CSV), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    npz_payload = dict(
        odom=np.asarray(traj.get("odom", np.empty((0, 11))), dtype=float),
        setpoint=np.asarray(traj.get("setpoint", np.empty((0, 4))), dtype=float),
        raw_mocap=raw_mocap,
        imu=imu,
        imu_bias=imu_bias,
        thrust=thrust,
        torque=torque,
        pos_err=pos_err,
        goal=np.asarray(traj.get("goal", [0.0, 0.0, 0.0]), dtype=float),
        radius=float(traj.get("radius", 0.0)),
        clockwise=bool(traj.get("clockwise", False)),
        t_start=float(ts["t_start"]),
        t_land_end=float(ts["t_land_end"]),
        battery=np.asarray(samples_dict.get("battery", np.empty((0, 2))), dtype=float),
    )
    if has_ekf_diag:
        npz_payload["ekf_diag"] = ekf_diag
    if has_ekf_adapt:
        npz_payload["ekf_adapt"] = ekf_adapt

    np.savez(os.path.join(dest, FILE_TRAJ_NPZ), **npz_payload)

    # Export synchronized 50 Hz telemetry table to CSV
    write_telemetry_csv(dest, ts, traj, samples, dt=0.02)

    # Copy the gain file (and policy file when OOT EKF is active) so the run
    # folder is self-contained even if the source configs are later edited.
    if pid_conf_path and os.path.isfile(pid_conf_path):
        shutil.copyfile(pid_conf_path, os.path.join(dest, FILE_PID_CONF))
    if adapt_conf_path and os.path.isfile(adapt_conf_path):
        shutil.copyfile(adapt_conf_path, os.path.join(dest, FILE_ADAPT_CONF))

    return row


def ideal_circle(goal, radius: float, clockwise: bool, n: int = 180) -> np.ndarray:
    """The circle the built-in non-sideways spiral traces from `goal`, yaw 0.

    Mirrors ``plan_spiral_from`` in the firmware planner: for a forward spiral
    the centre sits at ``goal + sense * radius * (sin psi, -cos psi)`` with
    yaw psi = 0, i.e. offset by ``sense * radius`` along -y (sense = +1 CW,
    -1 CCW). The flown path is that circle in the ``z = goal.z`` plane.
    """
    goal = np.asarray(goal, dtype=float)
    sense = 1.0 if clockwise else -1.0
    cx, cy, cz = goal[0], goal[1] - sense * radius, goal[2]
    # Start angle so the curve begins exactly at `goal`.
    start = np.arctan2(goal[0] - cx, -(goal[1] - cy))
    sweep = -sense * np.linspace(0.0, 2 * np.pi, n)
    px = cx + radius * np.sin(start + sweep)
    py = cy - radius * np.cos(start + sweep)
    pz = np.full(n, cz)
    return np.column_stack([px, py, pz])
