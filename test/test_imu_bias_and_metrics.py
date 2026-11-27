"""Unit tests for IMU bias telemetry export, NaN fallback, channel rates, and pre-takeoff guards."""
from __future__ import annotations

import csv
import json
import math
import os
import tempfile
import unittest

import numpy as np

from crazybridge import metrics
from crazybridge.pid_conf import PidConf


class TestImuBiasAndMetrics(unittest.TestCase):
    def _dummy_ts(self) -> dict[str, float]:
        return {
            "t_start": 10.0,
            "t_takeoff_end": 11.0,
            "t_regulate_end": 12.0,
            "t_circle_end": 13.0,
            "t_land_end": 14.0,
        }

    def test_missing_imu_bias_writes_nan_never_zero(self) -> None:
        ts = self._dummy_ts()
        traj = {"odom": np.empty((0, 11)), "setpoint": np.empty((0, 4))}
        samples = {"imu": np.empty((0, 7)), "battery": np.empty((0, 2))}
        m = metrics.compute_metrics(samples, ts)

        with tempfile.TemporaryDirectory() as tmp:
            metrics.write_reports(
                tmp,
                "vuelo_stock_nobias",
                ts,
                m,
                traj,
                samples=samples,
                estimator_mode="stock",
                adapt_mode="static",
            )
            dest = metrics.run_dir(tmp, "vuelo_stock_nobias")
            with open(os.path.join(dest, metrics.FILE_TELEMETRY_CSV), newline="") as f:
                rows = list(csv.reader(f))
            header = rows[0]
            self.assertEqual(len(header), 42)
            for col in ("vbat", *metrics.IMU_BIAS_CSV_HEADERS):
                self.assertIn(col, header)
            bias_idx = [header.index(c) for c in metrics.IMU_BIAS_CSV_HEADERS]
            vbat_idx = header.index("vbat")
            for row in rows[1:]:
                self.assertTrue(math.isnan(float(row[vbat_idx])))
                for idx in bias_idx:
                    self.assertTrue(math.isnan(float(row[idx])))

            with open(os.path.join(dest, metrics.FILE_METRICS_JSON)) as f:
                meta = json.load(f)
            self.assertEqual(meta["estimator_mode"], "stock")
            self.assertEqual(meta["adapt_mode"], "static")
            self.assertIsNone(meta["imu_bias"])
            self.assertNotIn("adapt_policies", meta)

            z = np.load(os.path.join(dest, metrics.FILE_TRAJ_NPZ))
            self.assertIn("imu_bias", z.files)
            self.assertEqual(z["imu_bias"].shape, (0, 7))

    def test_bias_sign_unit_convention_and_rates(self) -> None:
        ts = self._dummy_ts()
        t_imu = np.arange(10.0, 14.0001, 0.02)  # 50 Hz
        b_true = np.array([0.12, -0.08, 0.19, 0.0031, -0.0022, 0.0014])
        # Stationary IMU in FLU: raw = [0, 0, +9.81] + bias_a, [0, 0, 0] + bias_g
        imu_vals = np.tile(
            np.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0]) + b_true, (len(t_imu), 1)
        )
        imu = np.column_stack([t_imu, imu_vals])

        t_bias = np.arange(10.0, 14.0001, 0.5)  # 2 Hz
        imu_bias = np.column_stack([t_bias, np.tile(b_true, (len(t_bias), 1))])

        t_batt = np.arange(10.0, 14.0001, 0.1)  # 10 Hz
        battery = np.column_stack([t_batt, np.full(len(t_batt), 3.98)])

        t_ekf = np.arange(10.0, 14.0001, 0.02)  # 50 Hz
        ekf_diag = np.column_stack(
            [t_ekf, np.tile([0.42, 0.21, 0.0, 0.0, 3.0, 50.0, 50.0], (len(t_ekf), 1))]
        )

        traj = {
            "odom": np.column_stack(
                [np.arange(10.0, 14.0001, 0.025), np.zeros((161, 10))]
            ),
            "setpoint": np.column_stack(
                [np.arange(10.0, 14.0001, 0.05), np.zeros((81, 3))]
            ),
        }
        samples = {
            "imu": imu,
            "imu_bias": imu_bias,
            "battery": battery,
            "ekf_diag": ekf_diag,
            "thrust": np.column_stack(
                [np.arange(10.0, 14.0001, 0.1), np.full(41, 0.41)]
            ),
            "torque": np.column_stack(
                [np.arange(10.0, 14.0001, 0.1), np.zeros((41, 3))]
            ),
        }
        m = metrics.compute_metrics(samples, ts)

        pid_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "config",
            "pid_rigid_body.conf",
        )
        conf = PidConf.load(pid_path)
        self.assertAlmostEqual(conf.as_dict()["mass"], 0.042, places=6)

        with tempfile.TemporaryDirectory() as tmp:
            metrics.write_reports(
                tmp,
                "vuelo_oot_static",
                ts,
                m,
                traj,
                samples=samples,
                gains=conf.as_dict(),
                pid_conf_path=pid_path,
                estimator_mode="oot",
                adapt_mode="static",
            )
            dest = metrics.run_dir(tmp, "vuelo_oot_static")
            with open(os.path.join(dest, metrics.FILE_TELEMETRY_CSV), newline="") as f:
                reader = csv.reader(f)
                header = next(reader)
                data = np.array([[float(x) for x in r] for r in reader], dtype=float)

            self.assertEqual(len(header), 49)
            imu_cols = [
                header.index(f"imu_{c}") for c in ("ax", "ay", "az", "gx", "gy", "gz")
            ]
            bias_cols = [header.index(c) for c in metrics.IMU_BIAS_CSV_HEADERS]
            corr = data[:, imu_cols] - data[:, bias_cols]
            np.testing.assert_allclose(corr[:, 0], 0.0, atol=1e-9)
            np.testing.assert_allclose(corr[:, 1], 0.0, atol=1e-9)
            np.testing.assert_allclose(corr[:, 2], 9.81, atol=1e-9)
            np.testing.assert_allclose(corr[:, 3:6], 0.0, atol=1e-9)

            with open(os.path.join(dest, metrics.FILE_METRICS_JSON)) as f:
                meta = json.load(f)
            self.assertEqual(meta["estimator_mode"], "oot")
            self.assertEqual(meta["adapt_mode"], "static")
            self.assertAlmostEqual(meta["imu_bias"]["az"], 0.19, places=6)
            self.assertAlmostEqual(
                meta["channel_rates"]["imu"]["rate_hz"], 50.0, places=2
            )
            self.assertAlmostEqual(
                meta["channel_rates"]["imu_bias"]["rate_hz"], 2.0, places=2
            )
            self.assertAlmostEqual(
                meta["channel_rates"]["odom"]["rate_hz"], 40.0, places=2
            )

    def test_single_bias_sample_uses_zoh_not_zeros(self) -> None:
        ts = self._dummy_ts()
        b_true = np.array([-0.05, 0.04, -0.11, 0.001, -0.002, 0.0005])
        samples = {
            "imu_bias": np.array([[10.2, *b_true]]),
            "battery": np.array([[10.1, 4.05]]),
        }
        traj = {"odom": np.empty((0, 11)), "setpoint": np.empty((0, 4))}
        m = metrics.compute_metrics(samples, ts)

        with tempfile.TemporaryDirectory() as tmp:
            metrics.write_reports(
                tmp,
                "vuelo_single_bias",
                ts,
                m,
                traj,
                samples=samples,
                estimator_mode="stock",
                adapt_mode="static",
            )
            dest = metrics.run_dir(tmp, "vuelo_single_bias")
            with open(os.path.join(dest, metrics.FILE_TELEMETRY_CSV), newline="") as f:
                reader = csv.reader(f)
                header = next(reader)
                data = np.array([[float(x) for x in r] for r in reader], dtype=float)
            bias_cols = [header.index(c) for c in metrics.IMU_BIAS_CSV_HEADERS]
            vbat_col = header.index("vbat")
            np.testing.assert_allclose(data[:, vbat_col], 4.05, atol=1e-9)
            for j, expected in enumerate(b_true):
                np.testing.assert_allclose(data[:, bias_cols[j]], expected, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
