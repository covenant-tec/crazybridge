"""Automated OOT-controller comparison flight test.

Flies a fixed-duration sequence against the ``crazybridge`` node:

    takeoff -> regulate to a fixed point -> track a circle (built-in spiral) -> land

Every phase has a fixed duration, so the total test is identical across runs and
the integral metrics are computed over aligned time windows. While flying, the
node records the control signals (``thrust``, ``torque``), the position error
(``pos_error``), the flown path (``odometry``, via the base node's hook) and the
planned reference path (``setpoint`` = firmware ``ctrltarget``). When the
sequence finishes it hands the samples to :mod:`crazybridge.metrics`, which
writes one folder per run::

    <output_dir>/<config_name>/metrics.json   # metrics + the gains flown
                              /metrics.csv
                              /traj.npz
                              /pid.conf       # verbatim copy of the gain file

The filenames are identical across runs -- the folder name carries the config.
``pid_conf_path`` should be the same gain file the bridge was launched with; the
node parses it (via :class:`crazybridge.pid_conf.PidConf`) and stores the gains
in ``metrics.json`` so results are never orphaned from the gains behind them.

Visualisation is left to the ``rerun`` node (launched alongside), which draws the
planned-vs-actual paths live from the same ``setpoint`` / ``odometry`` topics.

The node exits when the sequence + reporting finish, so an orchestrator can
relaunch it for the next config. Service clients, request builders, the
odometry subscription and the parameter accessors all come from
:class:`crazybridge.interface.BridgeClientNode`.
"""

from __future__ import annotations

import math
import threading
from time import monotonic, sleep

import numpy as np

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PointStamped, PoseStamped, Vector3
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32, Float32MultiArray

from crazybridge import metrics
from ament_index_python.packages import get_package_share_directory
import os

from crazybridge.adapt_conf import AdaptConf, default_adapt_conf_path
from crazybridge.interface import (
    AdaptMode,
    BridgeClientNode,
    EkfAdaptDiag,
    EkfBiasDiag,
    EkfOotDiag,
    EstimatorMode,
    Topics,
    TrackingMode,
)
from crazybridge.pid_conf import PidConf

GRAVITY_MPS2 = 9.81
PREFLIGHT_MAX_GYRO_RAD_S = 0.20
PREFLIGHT_MAX_ACC_DEV_MPS2 = 1.00
PREFLIGHT_MAX_TILT_ACC_MPS2 = 1.50
PREFLIGHT_MIN_IMU_SAMPLES = 5
PREFLIGHT_IMU_WINDOW_S = 1.0
_EKF_FLAGS_COL = 1 + [f[0] for f in EkfOotDiag.FIELDS].index("flags")


class ControllerTest(BridgeClientNode):
    def __init__(self) -> None:
        # Base declares the service-name + odom_topic params, builds the service
        # clients, subscribes to odometry (-> on_odometry hook below) and gives
        # us _sp/_dp/_bp + the request builders.
        super().__init__("controller_test")

        # identity / output
        self.declare_parameter("config_name", "default")
        self.declare_parameter("output_dir", "/tmp/ctrl_test")
        # The same pid.conf the bridge loaded; recorded alongside the metrics so
        # a run's numbers can be traced back to the gains that produced them.
        self.declare_parameter("pid_conf_path", "")
        self.declare_parameter("adapt_conf_path", "")
        self.declare_parameter("tracking_mode", TrackingMode.AUTO)
        self.declare_parameter("estimator_mode", EstimatorMode.STOCK)
        self.declare_parameter("adapt_mode", AdaptMode.STATIC)

        # telemetry topic names (odom_topic comes from the base)
        self.declare_parameter("setpoint_topic", Topics.SETPOINT)
        self.declare_parameter("thrust_topic", Topics.THRUST)
        self.declare_parameter("torque_topic", Topics.TORQUE)
        self.declare_parameter("pos_error_topic", Topics.POS_ERROR)
        self.declare_parameter("imu_topic", Topics.IMU)
        self.declare_parameter("imu_bias_topic", Topics.IMU_BIAS)
        self.declare_parameter("ekf_diag_topic", Topics.EKF_DIAG)
        self.declare_parameter("ekf_adapt_topic", Topics.EKF_ADAPT)
        self.declare_parameter("battery_voltage_topic", Topics.BATTERY_VOLTAGE)
        self.declare_parameter("rigid_body_topic", Topics.RIGID_BODY)
        self.declare_parameter("marker_topic", Topics.MARKER)
        self.declare_parameter("require_imu_bias", True)
        self.declare_parameter("bias_wait_timeout_s", 5.0)

        # trajectory geometry
        self.declare_parameter("takeoff_height_m", 1.0)
        self.declare_parameter("goal_xyz", [0.5, 0.0, 1.0])
        self.declare_parameter("circle_radius_m", 0.5)
        self.declare_parameter("circle_clockwise", False)

        # fixed phase durations (seconds) -> identical total across configs
        self.declare_parameter("takeoff_duration_s", 3.0)
        self.declare_parameter("settle_s", 2.0)
        self.declare_parameter("goto_duration_s", 3.0)
        self.declare_parameter("regulate_s", 3.0)
        self.declare_parameter("circle_duration_s", 8.0)
        self.declare_parameter("circle_settle_s", 2.0)
        self.declare_parameter("land_duration_s", 3.0)

        self.declare_parameter("startup_timeout_s", 30.0)

        # sample buffers (timestamp = node-clock seconds)
        self._lock = threading.Lock()
        self._thrust: list[list[float]] = []
        self._torque: list[list[float]] = []
        self._pos_err: list[list[float]] = []
        self._odom: list[list[float]] = []
        self._setpoint: list[list[float]] = []
        self._raw_mocap: list[list[float]] = []
        self._raw_mocap_source: str = "none"
        self._imu: list[list[float]] = []
        self._imu_bias: list[list[float]] = []
        self._ekf_diag: list[list[float]] = []
        self._ekf_adapt: list[list[float]] = []
        self._battery: list[list[float]] = []
        self._took_off: bool = False

        self.create_subscription(Float32, self._sp("thrust_topic"), self._thrust_cb, 50)
        self.create_subscription(Vector3, self._sp("torque_topic"), self._torque_cb, 50)
        self.create_subscription(
            Vector3, self._sp("pos_error_topic"), self._pos_err_cb, 50
        )
        self.create_subscription(
            PointStamped, self._sp("setpoint_topic"), self._setpoint_cb, 50
        )
        self.create_subscription(Imu, self._sp("imu_topic"), self._imu_cb, 50)
        self.create_subscription(
            Float32MultiArray, self._sp("imu_bias_topic"), self._imu_bias_cb, 50
        )
        self.create_subscription(
            Float32MultiArray, self._sp("ekf_diag_topic"), self._ekf_diag_cb, 50
        )
        self.create_subscription(
            Float32MultiArray, self._sp("ekf_adapt_topic"), self._ekf_adapt_cb, 50
        )
        self.create_subscription(
            Float32, self._sp("battery_voltage_topic"), self._battery_cb, 50
        )
        qos_best_effort = QoSProfile(
            depth=10, reliability=ReliabilityPolicy.BEST_EFFORT
        )
        self.create_subscription(
            PoseStamped,
            self._sp("rigid_body_topic"),
            self._rigid_body_cb,
            qos_best_effort,
        )
        self.create_subscription(
            PointStamped,
            self._sp("marker_topic"),
            self._marker_cb,
            qos_best_effort,
        )

    # -- helpers ---------------------------------------------------------------
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _fp_array(self, name: str) -> list[float]:
        return list(self.get_parameter(name).get_parameter_value().double_array_value)

    # -- sample callbacks ------------------------------------------------------
    def _thrust_cb(self, msg: Float32) -> None:
        with self._lock:
            self._thrust.append([self._now(), float(msg.data)])

    def _torque_cb(self, msg: Vector3) -> None:
        with self._lock:
            self._torque.append([self._now(), msg.x, msg.y, msg.z])

    def _pos_err_cb(self, msg: Vector3) -> None:
        with self._lock:
            self._pos_err.append([self._now(), msg.x, msg.y, msg.z])

    def _setpoint_cb(self, msg: PointStamped) -> None:
        p = msg.point
        with self._lock:
            self._setpoint.append([self._now(), p.x, p.y, p.z])

    def _imu_cb(self, msg: Imu) -> None:
        with self._lock:
            self._imu.append([
                self._now(),
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,
            ])

    def _imu_bias_cb(self, msg: Float32MultiArray) -> None:
        if len(msg.data) < EkfBiasDiag.NUM_FIELDS:
            return
        vals = [float(v) for v in msg.data[: EkfBiasDiag.NUM_FIELDS]]
        if not all(math.isfinite(v) for v in vals):
            return
        with self._lock:
            self._imu_bias.append([self._now(), *vals])

    def _ekf_diag_cb(self, msg: Float32MultiArray) -> None:
        if len(msg.data) < EkfOotDiag.NUM_FIELDS:
            return
        with self._lock:
            self._ekf_diag.append([
                self._now(),
                *(float(v) for v in msg.data[: EkfOotDiag.NUM_FIELDS]),
            ])

    def _ekf_adapt_cb(self, msg: Float32MultiArray) -> None:
        if len(msg.data) < EkfAdaptDiag.NUM_FIELDS:
            return
        with self._lock:
            self._ekf_adapt.append([
                self._now(),
                *(float(v) for v in msg.data[: EkfAdaptDiag.NUM_FIELDS]),
            ])

    def _battery_cb(self, msg: Float32) -> None:
        with self._lock:
            self._battery.append([self._now(), float(msg.data)])

    def _rigid_body_cb(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        q = msg.pose.orientation
        with self._lock:
            self._raw_mocap.append([
                self._now(),
                p.x, p.y, p.z,
                q.x, q.y, q.z, q.w,
            ])
            self._raw_mocap_source = TrackingMode.RIGID_BODY

    def _marker_cb(self, msg: PointStamped) -> None:
        with self._lock:
            if self._raw_mocap_source != TrackingMode.RIGID_BODY:
                p = msg.point
                self._raw_mocap.append([
                    self._now(),
                    p.x, p.y, p.z,
                    0.0, 0.0, 0.0, 1.0,
                ])
                self._raw_mocap_source = TrackingMode.MARKER

    def on_odometry(self, msg: Odometry) -> None:
        # Base already stashed the latest message; we record full odometry
        # (pos, quat, vel) for trajectory overlay and flight telemetry export.
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist.linear
        with self._lock:
            self._odom.append([
                self._now(),
                p.x, p.y, p.z,
                q.x, q.y, q.z, q.w,
                v.x, v.y, v.z,
            ])

    # -- blocking service call -------------------------------------------------
    def _call(self, client, request, label: str, timeout: float = 15.0):
        if not client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f"{label}: service unavailable")
        future = client.call_async(request)
        deadline = monotonic() + timeout
        while not future.done():
            if monotonic() > deadline:
                raise RuntimeError(f"{label}: timed out")
            sleep(0.02)
        resp = future.result()
        if not getattr(resp, "success", True):
            raise RuntimeError(f"{label}: failed ({resp.message})")
        self.get_logger().info(f"{label}: ok")
        return resp

    def verify_pre_takeoff_bias_and_stillness(self) -> None:
        """Verify stationary IMU bias calibration (`bit0` of `ekf_flags`) and stillness before takeoff."""
        require_bias = self._bp("require_imu_bias")
        timeout_s = max(0.5, self._dp("bias_wait_timeout_s"))
        deadline = monotonic() + timeout_s
        last_reason = "waiting for telemetry"

        while monotonic() < deadline:
            now_t = self._now()
            with self._lock:
                latest_diag = list(self._ekf_diag[-1]) if self._ekf_diag else None
                latest_bias = list(self._imu_bias[-1]) if self._imu_bias else None
                recent_imu = [
                    row for row in self._imu if (now_t - row[0]) <= PREFLIGHT_IMU_WINDOW_S
                ]

            if require_bias or latest_diag is not None:
                if latest_diag is None:
                    last_reason = (
                        "no ekf_diag packet received (cannot verify EKF_OOT_FLAG_BIAS_READY)"
                    )
                    sleep(0.1)
                    continue
                flags = int(round(latest_diag[_EKF_FLAGS_COL]))
                if (flags & EkfOotDiag.FLAG_BIAS_READY) == 0:
                    last_reason = (
                        f"IMU bias calibration incomplete (ekf_flags=0x{flags:02x}, "
                        "bit0 EKF_OOT_FLAG_BIAS_READY=0)"
                    )
                    sleep(0.1)
                    continue
                if latest_bias is None:
                    last_reason = (
                        f"ekf_flags=0x{flags:02x} ready, waiting for calibrated /crazybridge/imu_bias packet"
                    )
                    sleep(0.1)
                    continue

            if len(recent_imu) < PREFLIGHT_MIN_IMU_SAMPLES:
                last_reason = (
                    f"waiting for recent IMU samples ({len(recent_imu)}/{PREFLIGHT_MIN_IMU_SAMPLES})"
                )
                sleep(0.1)
                continue

            b = latest_bias[1:7] if latest_bias is not None else [0.0] * 6
            max_gyro = 0.0
            max_acc_dev = 0.0
            tilted = False
            for row in recent_imu:
                acx = row[1] - b[0]
                acy = row[2] - b[1]
                acz = row[3] - b[2]
                gcx = row[4] - b[3]
                gcy = row[5] - b[4]
                gcz = row[6] - b[5]
                gyro_norm = math.sqrt(gcx * gcx + gcy * gcy + gcz * gcz)
                acc_norm = math.sqrt(acx * acx + acy * acy + acz * acz)
                max_gyro = max(max_gyro, gyro_norm)
                max_acc_dev = max(max_acc_dev, abs(acc_norm - GRAVITY_MPS2))
                if (
                    abs(acx) > PREFLIGHT_MAX_TILT_ACC_MPS2
                    or abs(acy) > PREFLIGHT_MAX_TILT_ACC_MPS2
                    or acz < (GRAVITY_MPS2 - PREFLIGHT_MAX_TILT_ACC_MPS2)
                ):
                    tilted = True
                    last_reason = (
                        "drone is tilted or not resting upright "
                        f"(a_corr=[{acx:.2f}, {acy:.2f}, {acz:.2f}] m/s^2)"
                    )
                    break

            if tilted:
                sleep(0.1)
                continue
            if (
                max_gyro > PREFLIGHT_MAX_GYRO_RAD_S
                or max_acc_dev > PREFLIGHT_MAX_ACC_DEV_MPS2
            ):
                last_reason = (
                    "drone is not stationary on the ground "
                    f"(max ||gyro||={max_gyro:.3f} rad/s, max | ||acc||-9.81 |={max_acc_dev:.3f} m/s^2)"
                )
                sleep(0.1)
                continue

            if latest_bias is not None and latest_diag is not None:
                flags = int(round(latest_diag[_EKF_FLAGS_COL]))
                self.get_logger().info(
                    "Pre-takeoff IMU bias & stillness verified: "
                    f"ekf_flags=0x{flags:02x} (BIAS_READY=1), "
                    f"ba=[{b[0]:+.4f}, {b[1]:+.4f}, {b[2]:+.4f}] m/s^2, "
                    f"bg=[{b[3]:+.5f}, {b[4]:+.5f}, {b[5]:+.5f}] rad/s, "
                    f"max ||gyro||={max_gyro:.4f} rad/s"
                )
            else:
                self.get_logger().info(
                    f"Pre-takeoff stillness verified (max ||gyro||={max_gyro:.4f} rad/s)"
                )
            return

        msg = f"PRE-TAKEOFF CHECK FAILED — refusing takeoff: {last_reason}"
        self.get_logger().error(msg)
        raise RuntimeError(msg)

    # -- the flight sequence ---------------------------------------------------
    def run_sequence(self) -> dict:
        """Execute the fixed-duration sequence, returning phase timestamps."""
        self.get_logger().info("waiting for services and odometry...")
        self.wait_until_ready(self._dp("startup_timeout_s"))
        self.get_logger().info("services + odometry ready; verifying IMU bias & stillness...")
        self.verify_pre_takeoff_bias_and_stillness()

        height = self._dp("takeoff_height_m")
        goal = self._fp_array("goal_xyz")
        if len(goal) < 3:
            goal = [0.5, 0.0, height]
        radius = self._dp("circle_radius_m")
        clockwise = self._bp("circle_clockwise")

        t_takeoff = self._dp("takeoff_duration_s")
        t_settle = self._dp("settle_s")
        t_goto = self._dp("goto_duration_s")
        t_regulate = self._dp("regulate_s")
        t_circle = self._dp("circle_duration_s")
        t_circle_settle = self._dp("circle_settle_s")
        t_land = self._dp("land_duration_s")

        ts: dict[str, float] = {"t_start": self._now()}

        # Phase 1: takeoff to `height` above current xy, then hold.
        self._took_off = True
        self._call(self.takeoff_cli, self.takeoff_request(height, t_takeoff), "takeoff")
        sleep(t_takeoff + t_settle)
        ts["t_takeoff_end"] = self._now()

        # Phase 2: regulate to the predetermined point (absolute go_to), hold.
        self._call(
            self.goto_cli,
            self.goto_request(goal[0], goal[1], goal[2], 0.0, t_goto, relative=False),
            "go_to",
        )
        sleep(t_goto + t_regulate)
        ts["t_regulate_end"] = self._now()

        # Phase 3: track a full circle via the built-in spiral (r0 == rF).
        self._call(
            self.spiral_cli,
            self.spiral_request(
                360.0,
                radius,
                radius,
                0.0,
                t_circle,
                sideways=False,
                clockwise=clockwise,
            ),
            "spiral",
        )
        sleep(t_circle + t_circle_settle)
        ts["t_circle_end"] = self._now()

        # Phase 4: land.
        self._call(self.land_cli, self.land_request(0.0, t_land), "land")
        sleep(t_land)
        ts["t_land_end"] = self._now()

        self._goal = goal
        self._radius = radius
        self._clockwise = clockwise
        return ts

    def emergency_stop(self) -> None:
        try:
            self._call(
                self.land_cli,
                self.land_request(0.0, 2.0),
                "emergency-land",
                timeout=6.0,
            )
            sleep(2.5)
        except Exception as exc:
            self.get_logger().error(f"emergency land failed: {exc}")
        try:
            self._call(self.kill_cli, self.bool_request(True), "kill", timeout=4.0)
        except Exception as exc:
            self.get_logger().error(f"kill failed: {exc}")

    # -- reporting -------------------------------------------------------------
    def _samples(self) -> dict:
        with self._lock:
            return {
                "thrust": np.array(self._thrust, dtype=float)
                if self._thrust
                else np.empty((0, 2)),
                "torque": np.array(self._torque, dtype=float)
                if self._torque
                else np.empty((0, 4)),
                "pos_err": np.array(self._pos_err, dtype=float)
                if self._pos_err
                else np.empty((0, 4)),
                "imu": np.array(self._imu, dtype=float)
                if self._imu
                else np.empty((0, 7)),
                "imu_bias": np.array(self._imu_bias, dtype=float)
                if self._imu_bias
                else np.empty((0, 1 + EkfBiasDiag.NUM_FIELDS)),
                "raw_mocap": np.array(self._raw_mocap, dtype=float)
                if self._raw_mocap
                else np.empty((0, 8)),
                "ekf_diag": np.array(self._ekf_diag, dtype=float)
                if self._ekf_diag
                else np.empty((0, 1 + EkfOotDiag.NUM_FIELDS)),
                "ekf_adapt": np.array(self._ekf_adapt, dtype=float)
                if self._ekf_adapt
                else np.empty((0, 1 + EkfAdaptDiag.NUM_FIELDS)),
                "battery": np.array(self._battery, dtype=float)
                if self._battery
                else np.empty((0, 2)),
            }

    def _traj(self) -> dict:
        with self._lock:
            odom = np.array(self._odom, dtype=float) if self._odom else np.empty((0, 11))
            setp = (
                np.array(self._setpoint, dtype=float)
                if self._setpoint
                else np.empty((0, 4))
            )
        return {
            "odom": odom,
            "setpoint": setp,
            "goal": getattr(self, "_goal", [0.0, 0.0, 0.0]),
            "radius": getattr(self, "_radius", 0.0),
            "clockwise": getattr(self, "_clockwise", False),
        }

    def _gains(self) -> tuple[dict | None, str]:
        """The gains this run flew with, as ``(flat dict | None, path)``."""
        path = self._sp("pid_conf_path")
        if not path:
            tracking_mode = self._sp("tracking_mode")
            if tracking_mode == TrackingMode.AUTO:
                with self._lock:
                    active = self._raw_mocap_source
                if active in (TrackingMode.RIGID_BODY, TrackingMode.MARKER):
                    detected = active
                else:
                    detected = TrackingMode.RIGID_BODY
                conf_file = "pid.conf" if detected == TrackingMode.MARKER else "pid_rigid_body.conf"
                path = os.path.join(
                    get_package_share_directory("crazybridge"), "config", conf_file
                )
                self.get_logger().info(
                    f"[AUTO MODE] Auto-resolved PID path based on detected stream ({detected}): {path}"
                )
            else:
                conf_file = "pid.conf" if tracking_mode == TrackingMode.MARKER else "pid_rigid_body.conf"
                path = os.path.join(
                    get_package_share_directory("crazybridge"), "config", conf_file
                )
                self.get_logger().info(f"Auto-resolved default PID path for mode={tracking_mode!r}: {path}")
        try:
            conf = PidConf.load(path)
        except Exception as exc:
            self.get_logger().error(f"could not read gains from {path}: {exc}")
            return None, path
        return conf.as_dict(), path

    def _adapt_policies(
        self, est_mode: str, adapt_mode: str
    ) -> tuple[dict | None, str | None]:
        """The MDP/SMDP Bellman policies loaded when running adaptive OOT EKF."""
        if est_mode == EstimatorMode.STOCK or adapt_mode == AdaptMode.STATIC:
            return None, None
        raw_path = self._sp("adapt_conf_path").strip()
        path = raw_path if raw_path else default_adapt_conf_path()
        try:
            conf = AdaptConf.load(path)
        except Exception as exc:
            self.get_logger().warning(f"could not read EKF policies from {path}: {exc}")
            return None, path
        return conf.as_dict(), path

    def report(self, ts: dict) -> dict:
        samples = self._samples()
        m = metrics.compute_metrics(samples, ts)
        gains, pid_conf_path = self._gains()
        est_mode = self._sp("estimator_mode").strip().lower() or EstimatorMode.STOCK
        adapt_mode = AdaptMode.normalize(self._sp("adapt_mode")) or AdaptMode.STATIC
        adapt_policies, adapt_conf_path = self._adapt_policies(est_mode, adapt_mode)
        config = self._sp("config_name")
        out_dir = self._sp("output_dir")
        metrics.write_reports(
            out_dir,
            config,
            ts,
            m,
            self._traj(),
            samples=samples,
            gains=gains,
            pid_conf_path=pid_conf_path,
            estimator_mode=est_mode,
            adapt_mode=adapt_mode,
            adapt_policies=adapt_policies,
            adapt_conf_path=adapt_conf_path,
        )
        self.get_logger().info(
            f"wrote run artefacts to {metrics.run_dir(out_dir, config)}"
        )
        return m


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ControllerTest()

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    exit_code = 0
    try:
        ts = node.run_sequence()
        m = node.report(ts)
        circ = m["circle"]
        node.get_logger().info(
            f"DONE [{node._sp('config_name')}] circle: "
            f"int|thrust|={circ['int_thrust_abs']:.3f} "
            f"int|torque|={circ['int_torque_norm']:.4f} "
            f"int|pos_err|={circ['int_err_norm']:.4f} "
            f"RMSE={circ['rmse_norm']:.4f}"
        )
    except Exception as exc:
        node.get_logger().error(f"test sequence failed: {exc}")
        if node._took_off:
            node.emergency_stop()
        else:
            node.get_logger().warning(
                "Aborted on ground prior to takeoff; skipping emergency_stop."
            )
        exit_code = 1
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
