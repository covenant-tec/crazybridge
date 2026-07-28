"""Node-agnostic metrics + report writing for the controller comparison test.

All functions take plain numpy arrays / dicts (no ROS or node dependency) so
they are easy to test and reuse. Sample arrays are laid out as:

* ``thrust``  -> shape (N, 2): columns ``[t, thrust]``
* ``torque``  -> shape (N, 4): columns ``[t, x, y, z]``
* ``pos_err`` -> shape (N, 4): columns ``[t, ex, ey, ez]``

Timestamps are seconds (any consistent clock). Integrals are trapezoidal.
The headline comparison numbers are the *circle* phase ``int_thrust_abs`` /
``int_torque_norm`` (control effort) and ``int_err_norm`` (tracking error).
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np

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


def write_reports(out_dir: str, config: str, ts: dict, metrics: dict,
                  traj: dict) -> dict:
    """Write ``<config>_metrics.json``, ``.csv`` and ``_traj.npz``.

    Returns the flattened metric row (useful for logging / aggregation).
    """
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, f'{config}_metrics.json'), 'w') as f:
        json.dump({'config': config, 'timestamps': ts, 'metrics': metrics},
                  f, indent=2)

    row = flatten_row(config, metrics)
    with open(os.path.join(out_dir, f'{config}_metrics.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    np.savez(
        os.path.join(out_dir, f'{config}_traj.npz'),
        odom=np.asarray(traj.get('odom', np.empty((0, 4))), dtype=float),
        setpoint=np.asarray(traj.get('setpoint', np.empty((0, 4))), dtype=float),
        goal=np.asarray(traj.get('goal', [0.0, 0.0, 0.0]), dtype=float),
        radius=float(traj.get('radius', 0.0)),
        clockwise=bool(traj.get('clockwise', False)),
        t_start=float(ts['t_start']), t_land_end=float(ts['t_land_end']),
    )
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
