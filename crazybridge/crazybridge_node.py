"""ROS2 bridge for a single Crazyflie, ported from forerunner2/crazybridge.

Mirrors crazybridge.cpp: brings up a cflib link (defaulting to the CrazySim
UDP URI), enables the high-level commander and Kalman estimator, streams
position + quaternion as nav_msgs/Odometry, forwards mocap marker fixes via
the external-position channel, and exposes takeoff/land/go_to/spiral services
that drive the high-level commander, plus upload_trajectory/start_trajectory
for polynomial trajectories stored in the Crazyflie's trajectory memory.
"""
from __future__ import annotations

import math
import os
import threading

import rclpy
from time import sleep, time
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.mem import MemoryElement, Poly4D
from logging import getLogger

from ament_index_python.packages import get_package_share_directory

from std_msgs.msg import Float32, Header
from std_srvs.srv import SetBool
from geometry_msgs.msg import PointStamped, Vector3, Quaternion, PointStamped, Point
from nav_msgs.msg import Odometry

from crazybridge_interfaces.srv import (
    GoTo, Land, Spiral, StartTrajectory, Takeoff, UploadTrajectory)

from crazybridge.interface import BridgePublishers, Create
from crazybridge.pid_conf import PidConf

logger = getLogger()

PARAM_HL_COMMANDER = 'commander.enHighLevel'
PARAM_ESTIMATOR = 'stabilizer.estimator'
PARAM_CONTROLLER = 'stabilizer.controller'
ESTIMATOR_KALMAN = 2
ESTIMATOR_COMPLEMENTARY = 1

# Firmware trajectory limits (crtp_commander_high_level.c): 4096 B of trajectory
# memory / 132 B per raw poly4d piece = 31 pieces, and 31 trajectory slots.
MAX_TRAJ_PIECES = 31
NUM_TRAJ_SLOTS = 31
TRAJ_COEFFS = 8   # degree-7 polynomial per axis

# The ootParams gain names live in crazybridge.pid_conf.LAYOUT.


class CrazyBridge(Node):
    def __init__(self) -> None:
        super().__init__('crazybridge')

        self.declare_parameter('uri', 'udp://127.0.0.1:19850')
        # cflib log periods are in real milliseconds; the firmware quantises
        # them to 10 ms ticks, so values must be a multiple of 10 and >= 10.
        # The C++ port called start(5)/start(1) which were raw 10 ms ticks,
        # i.e. 50 ms for pos/q and 10 ms for the OOT blocks.
        self.declare_parameter('log_period_ms', 50)
        self.declare_parameter('oot_log_period_ms', 10)
        self.declare_parameter('connection_timeout_s', 10.0)
        self.declare_parameter('odom_frame_id', 'world')
        self.declare_parameter('child_frame_id', 'crazyflie')
        self.declare_parameter('pid_conf_path', '')
        self.declare_parameter('load_pid_conf', True)
        self.declare_parameter('max_distance_from_origin', 3.0)
        # Trajectory upload: per-attempt timeout for the memory write, number
        # of attempts, and the extpos rate kept up while the upload has the
        # radio link to itself (0 = pause extpos entirely).
        self.declare_parameter('upload_attempt_timeout_s', 15.0)
        self.declare_parameter('upload_attempts', 2)
        self.declare_parameter('upload_extpos_hz', 10.0)

        self._uri = self.get_parameter(
            'uri').get_parameter_value().string_value
        self._log_period_ms = self._sanitise_log_period(
            self.get_parameter(
                'log_period_ms').get_parameter_value().integer_value,
            'log_period_ms',
        )
        self._max_distance_from_origin = self.get_parameter("max_distance_from_origin").get_parameter_value().double_value
        self._oot_log_period_ms = self._sanitise_log_period(
            self.get_parameter(
                'oot_log_period_ms').get_parameter_value().integer_value,
            'oot_log_period_ms',
        )
        self._odom_frame = (
            self.get_parameter(
                'odom_frame_id').get_parameter_value().string_value
        )
        self._child_frame = (
            self.get_parameter(
                'child_frame_id').get_parameter_value().string_value
        )
        self._pid_conf_path = self._resolve_pid_conf_path()
        self._load_pid_conf = bool(
            self.get_parameter(
                'load_pid_conf').get_parameter_value().bool_value
        )

        self._pos = [0.0, 0.0, 0.0]
        self._quat = [0.0, 0.0, 0.0, 1.0]  # (x, y, z, w)
        self._pos_lock = threading.Lock()

        # Latest OOT samples (mirrors last_q_error / last_qd / last_pos_error in C++).
        self._oot_lock = threading.Lock()
        self._oot_pos_err = (0.0, 0.0, 0.0)
        self._oot_q_err = (0.0, 0.0, 0.0, 1.0)  # (x, y, z, w)
        self._oot_qd = (0.0, 0.0, 0.0, 1.0)     # (x, y, z, w)

        self._upload_attempt_timeout_s = max(1.0, float(self.get_parameter(
            'upload_attempt_timeout_s').get_parameter_value().double_value))
        self._upload_attempts = max(1, int(self.get_parameter(
            'upload_attempts').get_parameter_value().integer_value))
        upload_hz = float(self.get_parameter(
            'upload_extpos_hz').get_parameter_value().double_value)
        self._upload_extpos_period = 1.0 / upload_hz if upload_hz > 0 else math.inf
        self._upload_active = threading.Event()
        self._last_extpos_sent = 0.0

        self._marker_id: int = None
        self._safe_to_fly: bool = False
        self._bad_marker_counter: int = 0
        self._last_seen_marker: float = 0
        self._safe_to_fly_lock = threading.Lock()

        # Telemetry publishers (names/types live in interface.BridgePublishers,
        # shared with the client nodes so they cannot drift). ~/odometry and
        # ~/setpoint carry the actual and planned paths; thrust/torque/pos_error
        # carry the OOT control signals and tracking error.
        self._pubs = BridgePublishers(self)

        self._marker_sub = self.create_subscription(
            PointStamped, Create.MARKER, self._marker_cb,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        )
        self._takeoff_srv=None
       # self._safe_to_fly = True
       # self._takeoff_srv = self.create_service(
       #             Takeoff, Create.SRV_TAKEOFF, self._takeoff_cb)
        self._land_srv = self.create_service(
            Land, Create.SRV_LAND, self._land_cb)
        self._goto_srv = self.create_service(
            GoTo, Create.SRV_GOTO, self._goto_cb)
        self._spiral_srv = self.create_service(
            Spiral, Create.SRV_SPIRAL, self._spiral_cb)
        # Uploading a trajectory blocks for seconds (hundreds of acknowledged
        # radio packets). It gets its own callback group so that, under the
        # MultiThreadedExecutor in main(), the mocap marker callback (which feeds
        # extpos to the Kalman filter) and the safety timer keep running instead
        # of being starved -- a 2 s marker gap would otherwise trigger the
        # safety-landing path in _safety_check.
        self._traj_group = MutuallyExclusiveCallbackGroup()
        self._upload_traj_srv = self.create_service(
            UploadTrajectory, Create.SRV_UPLOAD_TRAJ, self._upload_traj_cb,
            callback_group=self._traj_group)
        self._start_traj_srv = self.create_service(
            StartTrajectory, Create.SRV_START_TRAJ, self._start_traj_cb)
        self._kill_srv = self.create_service(
            SetBool, Create.SRV_KILL, self._kill_cb)

        cflib.crtp.init_drivers()
        self._connected = threading.Event()
        self._connect_failed = threading.Event()
        self._connect_error: str | None = None

        self._cf = Crazyflie(rw_cache='./cache')
        self._cf.fully_connected.add_callback(self._on_connected)
        self._cf.disconnected.add_callback(self._on_disconnected)
        self._cf.connection_failed.add_callback(self._on_connection_failed)
        self._cf.connection_lost.add_callback(self._on_connection_lost)
        self._cf.console.receivedChar.add_callback(self._on_console)

        self.get_logger().info(f'Opening link to {self._uri}')
        self._cf.open_link(self._uri)

        timeout = float(
            self.get_parameter(
                'connection_timeout_s').get_parameter_value().double_value
        )
        if not self._connected.wait(timeout=timeout):
            self._cf.close_link()
            raise RuntimeError(
                f'Timed out connecting to {self._uri} after {timeout}s '
                f'({self._connect_error or "no failure callback fired"})'
            )
            return
        self._safety_interval = self.create_timer(1, self._safety_check)

    def _on_console(self, text):
        self.get_logger().info(text)

    def _sanitise_log_period(self, value: int, name: str) -> int:
        # cflib divides period_in_ms by 10 (uint8 firmware ticks). Must be a
        # positive multiple of 10 and < 2550 ms, or add_config rejects it.
        v = int(value)
        if v < 10:
            self.get_logger().warning(
                f'{name}={v} ms is below cflib minimum 10 ms; clamping to 10 ms'
            )
            v = 10
        if v % 10 != 0:
            adjusted = (v // 10) * 10 or 10
            self.get_logger().warning(
                f'{name}={v} ms is not a multiple of 10; rounding to {adjusted} ms'
            )
            v = adjusted
        if v > 2540:
            self.get_logger().warning(
                f'{name}={v} ms exceeds cflib max 2540 ms; clamping'
            )
            v = 2540
        return v

    def _resolve_pid_conf_path(self) -> str:
        path = self.get_parameter(
            'pid_conf_path').get_parameter_value().string_value
        if path:
            return path
        return os.path.join(
            get_package_share_directory('crazybridge'), 'config', 'pid.conf'
        )

    def _on_connected(self, link_uri: str) -> None:
        self.get_logger().info(f'Connected to {link_uri}')
        if self._load_pid_conf:
            try:
                self._apply_pid_conf(self._pid_conf_path)
            except Exception as exc:
                self.get_logger().error(
                    f'Failed to load pid.conf from {self._pid_conf_path}: {exc}')  # noqa
        try:
            self._cf.param.set_value(PARAM_HL_COMMANDER, 1)
#            self._cf.param.set_value(PARAM_CONTROLLER, 5) # Add constats for the controller oot = 5 and auto = 0
            self._cf.param.set_value(PARAM_ESTIMATOR, ESTIMATOR_KALMAN)
        except Exception as exc:
            self.get_logger().error(f'Failed to set startup params: {exc}')

        self._pos_log = LogConfig(
            name='kalman_pos', period_in_ms=self._log_period_ms)
        self._pos_log.add_variable('kalman.stateX', 'float')
        self._pos_log.add_variable('kalman.stateY', 'float')
        self._pos_log.add_variable('kalman.stateZ', 'float')

        self._q_log = LogConfig(
            name='kalman_q', period_in_ms=self._log_period_ms)
        self._q_log.add_variable('kalman.q0', 'float')
        self._q_log.add_variable('kalman.q1', 'float')
        self._q_log.add_variable('kalman.q2', 'float')
        self._q_log.add_variable('kalman.q3', 'float')

        extra_period = self._sanitise_log_period(500, "extra_log_ms")
        self._pm_log = LogConfig(name="battery", period_in_ms=extra_period)
        self._pm_log.add_variable("pm.batteryLevel", "uint8_t")

        # ctrltarget.* is the controller's desired position (the planned
        # trajectory the high-level commander is driving toward). It is a core
        # firmware log group, so it is populated regardless of which controller
        # or gain set is active -- ideal as the reference to compare the flown
        # path against.
        self._setpoint_log = LogConfig(
            name='ctrltarget', period_in_ms=self._log_period_ms)
        self._setpoint_log.add_variable('ctrltarget.x', 'float')
        self._setpoint_log.add_variable('ctrltarget.y', 'float')
        self._setpoint_log.add_variable('ctrltarget.z', 'float')

        try:
            self._cf.log.add_config(self._pos_log)
            self._cf.log.add_config(self._q_log)
            self._cf.log.add_config(self._pm_log)
            self._cf.log.add_config(self._setpoint_log)
            self._pos_log.data_received_cb.add_callback(self._on_pos_log)
            self._q_log.data_received_cb.add_callback(self._on_q_log)
            self._pm_log.data_received_cb.add_callback(self._on_pm_log)
            self._setpoint_log.data_received_cb.add_callback(
                self._on_setpoint_log)
            self._pos_log.start()
            self._q_log.start()
            self._pm_log.start()
            self._setpoint_log.start()
        except Exception as exc:
            self.get_logger().error(f'Failed to register log blocks: {exc}')

        self._setup_oot_logs()

        self._connected.set()

    def _setup_oot_logs(self) -> None:
        period = self._oot_log_period_ms
        self._input_log = LogConfig(name='input', period_in_ms=period)
        self._input_log.add_variable('oot.thrust', 'float')
        self._input_log.add_variable('oot.torque_x', 'float')
        self._input_log.add_variable('oot.torque_y', 'float')
        self._input_log.add_variable('oot.torque_z', 'float')

        self._pos_err_log = LogConfig(name='oot_pos_err', period_in_ms=period)
        self._pos_err_log.add_variable('oot.pos_err_x', 'float')
        self._pos_err_log.add_variable('oot.pos_err_y', 'float')
        self._pos_err_log.add_variable('oot.pos_err_z', 'float')

        self._q_err_log = LogConfig(name='oot_q_err', period_in_ms=period)
        self._q_err_log.add_variable('oot.q_err_x', 'float')
        self._q_err_log.add_variable('oot.q_err_y', 'float')
        self._q_err_log.add_variable('oot.q_err_z', 'float')
        self._q_err_log.add_variable('oot.q_err_w', 'float')

        self._qd_log = LogConfig(name='oot_qd', period_in_ms=period)
        self._qd_log.add_variable('oot.qd_x', 'float')
        self._qd_log.add_variable('oot.qd_y', 'float')
        self._qd_log.add_variable('oot.qd_z', 'float')
        self._qd_log.add_variable('oot.qd_w', 'float')

        self._ang_vel_err_log = LogConfig(
            name='oot_ang_vel_err', period_in_ms=period)
        self._ang_vel_err_log.add_variable('oot.ang_vel_err_x', 'float')
        self._ang_vel_err_log.add_variable('oot.ang_vel_err_y', 'float')
        self._ang_vel_err_log.add_variable('oot.ang_vel_err_z', 'float')

        try:
            self._cf.log.add_config(self._input_log)
            self._cf.log.add_config(self._pos_err_log)
            self._cf.log.add_config(self._q_err_log)
            self._cf.log.add_config(self._qd_log)
            self._cf.log.add_config(self._ang_vel_err_log)

            self._input_log.data_received_cb.add_callback(
                self._on_input_log)
            self._pos_err_log.data_received_cb.add_callback(
                self._on_pos_err_log)
            self._q_err_log.data_received_cb.add_callback(self._on_q_err_log)
            self._qd_log.data_received_cb.add_callback(self._on_qd_log)
            self._ang_vel_err_log.data_received_cb.add_callback(
                self._on_ang_vel_err_log
            )

           # self._input_log.start()
            self._pos_err_log.start()
           # self._q_err_log.start()
            self._qd_log.start()
           # self._ang_vel_err_log.start()
            self.get_logger().info('OOT log blocks started')
        except Exception as exc:
            self.get_logger().error(
                f'Failed to register OOT log blocks: {exc}')

    def _on_pm_log(self, _timestamp, data, _logconf):
        batt_msg = Float32()
        batt_msg.data = float(data["pm.batteryLevel"])
        self._pubs.battery.publish(batt_msg)

    def _on_setpoint_log(self, _timestamp, data, _logconf):
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._odom_frame
        msg.point.x = float(data['ctrltarget.x'])
        msg.point.y = float(data['ctrltarget.y'])
        msg.point.z = float(data['ctrltarget.z'])
        self._pubs.setpoint.publish(msg)

    def _on_input_log(self, _timestamp, data, _logconf):
        thrust_msg = Float32()
        thrust_msg.data = float(data["oot.thrust"])
        self._pubs.thrust.publish(thrust_msg)

        torque_msg = Vector3()
        torque_msg.x = float(data["oot.torque_x"])
        torque_msg.y = float(data["oot.torque_y"])
        torque_msg.z = float(data["oot.torque_z"])
        self._pubs.torque.publish(torque_msg)

    def _apply_pid_conf(self, path: str) -> None:
        # PidConf owns the file layout (24 values -> ootParams names) so the
        # bridge and controller_test cannot disagree about what a run flew with.
        conf = PidConf.load(path)

        self.get_logger().info(f'Loading PID gains from {path}')
        for line in conf.describe():
            self.get_logger().info(line)

        for name, value in conf.params():
            self._cf.param.set_value(name, value)

        self.get_logger().info('Done configuring PID')

    def _on_disconnected(self, link_uri: str) -> None:
        self.get_logger().info(f'Disconnected from {link_uri}')

    def _marker_cb(self, msg: PointStamped) -> None:
        header: Header = msg.header
        if self._marker_id is None:
            self._marker_id = header.frame_id
        if self._marker_id != header.frame_id:
            return
        point: Point = msg.point
        axis = [point.x, point.y, point.z]
        over_limit = [abs(i) > self._max_distance_from_origin for i in axis]
        if any(over_limit):
            print(over_limit)
            self._bad_marker_counter += 1
            return
        self._last_seen_marker = time()
        self._bad_marker_counter = self._bad_marker_counter -1 if self._bad_marker_counter > 0 else 0
        if self._cf.connected:
            if self._upload_active.is_set():
                # A trajectory upload needs the radio link: send extpos at a
                # reduced rate (or not at all) so the memory writes get through.
                now = time()
                if now - self._last_extpos_sent < self._upload_extpos_period:
                    return
                self._last_extpos_sent = now
            self._cf.extpos.send_extpos(point.x, point.y, point.z)

    def _safety_check(self):
        if not self._cf.is_connected():
            self.get_logger().error("Disconnected from crazyflie. Trying to reconnect")
            with self._safe_to_fly_lock:
                self._cf.open_link(self._uri)
                if self._connected.wait(timeout=1):
                    self.get_logger().error("LANDING!!")
                    if self._pos[2] > 0.1:
                        self._cf.high_level_commander.land(0, 2)
                        sleep(5)
                    self._cf.supervisor.send_emergency_stop()

        if self._marker_id is None: 
            return
        time_diff = time() - self._last_seen_marker
        if time_diff > 2 or self._bad_marker_counter > 10:
            with self._safe_to_fly_lock:
                self.get_logger().error(f"Bad marker detected. Not seen for {time_diff} and had {self._bad_marker_counter} infractions")
                if self._safe_to_fly:
                    self._safe_to_fly = False
                    self.get_logger().error("LANDING!!")
                    self._cf.high_level_commander.land(0, 2)
                    sleep(5)
                    self._cf.supervisor.send_emergency_stop()
            return
        with self._safe_to_fly_lock:
            if not self._takeoff_srv:
                self._takeoff_srv = self.create_service(
                    Takeoff, Create.SRV_TAKEOFF, self._takeoff_cb)
            self._safe_to_fly = True

    def _on_connection_failed(self, link_uri: str, msg: str) -> None:
        self._connect_error = msg
        self.get_logger().error(f'Connection to {link_uri} failed: {msg}')
        self._connect_failed.set()
        self._connected.set()

    def _on_connection_lost(self, link_uri: str, msg: str) -> None:
        self.get_logger().warning(f'Connection to {link_uri} lost: {msg}')

    def _on_pos_log(self, _timestamp, data, _logconf) -> None:
        with self._pos_lock:
            self._pos = [
                float(data['kalman.stateX']),
                float(data['kalman.stateY']),
                float(data['kalman.stateZ']),
            ]
        self._publish_odom()

    def _on_q_log(self, _timestamp, data, _logconf) -> None:
        # Bitcraze kalman logs (q0, q1, q2, q3) as (w, x, y, z).
        x, y, z, w = self._normalize_quat(
            float(data['kalman.q1']),
            float(data['kalman.q2']),
            float(data['kalman.q3']),
            float(data['kalman.q0']),
        )
        with self._pos_lock:
            self._quat = [x, y, z, w]
        self._publish_odom()

    def _on_pos_err_log(self, _timestamp, data, _logconf) -> None:
        v = (
            float(data['oot.pos_err_x']),
            float(data['oot.pos_err_y']),
            float(data['oot.pos_err_z']),
        )
        with self._oot_lock:
            self._oot_pos_err = v

        msg = Vector3()
        msg.x = v[0]
        msg.y = v[1]
        msg.z = v[2]
        self._pubs.pos_error.publish(msg)

    def _on_q_err_log(self, _timestamp, data, _logconf) -> None:
        v = self._normalize_quat(
            float(data['oot.q_err_x']),
            float(data['oot.q_err_y']),
            float(data['oot.q_err_z']),
            float(data['oot.q_err_w']),
        )
        with self._oot_lock:
            self._oot_q_err = v
        msg = Quaternion()
        msg.x = v[0]
        msg.y = v[1]
        msg.z = v[2]
        msg.w = v[3]
        self._pubs.orientation_error.publish(msg)

    def _on_qd_log(self, _timestamp, data, _logconf) -> None:
        v = self._normalize_quat(
            float(data['oot.qd_x']),
            float(data['oot.qd_y']),
            float(data['oot.qd_z']),
            float(data['oot.qd_w']),
        )
        with self._oot_lock:
            self._oot_qd = v
        msg = Quaternion()
        msg.x = v[0]
        msg.y = v[1]
        msg.z = v[2]
        msg.w = v[3]
        self._pubs.orientation_desired.publish(msg)

    def _on_ang_vel_err_log(self, _timestamp, data, _logconf) -> None:
        v = (
            float(data['oot.ang_vel_err_x']),
            float(data['oot.ang_vel_err_y']),
            float(data['oot.ang_vel_err_z']),
        )

    def _publish_odom(self) -> None:
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._odom_frame
        msg.child_frame_id = self._child_frame
        with self._pos_lock:
            pos = tuple(self._pos)
            quat = tuple(self._quat)  # (x, y, z, w)
        msg.pose.pose.position.x = pos[0]
        msg.pose.pose.position.y = pos[1]
        msg.pose.pose.position.z = pos[2]
        msg.pose.pose.orientation.x = quat[0]
        msg.pose.pose.orientation.y = quat[1]
        msg.pose.pose.orientation.z = quat[2]
        msg.pose.pose.orientation.w = quat[3]
        self._pubs.odom.publish(msg)

    @staticmethod
    def _normalize_quat(x: float, y: float, z: float, w: float) -> tuple[float, float, float, float]:
        """Return (x, y, z, w) as a unit quaternion.

        The Crazyflie estimator occasionally emits quaternions that are not
        quite unit-length (e.g. right after connect, before the filter has
        settled), so we normalize before publishing to ROS. Falls back to
        identity when the norm is degenerate.
        """
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm < 1e-9:
            return 0.0, 0.0, 0.0, 1.0
        return x / norm, y / norm, z / norm, w / norm

    @staticmethod
    def _duration_to_seconds(duration) -> float:
        return float(duration.sec) + float(duration.nanosec) * 1e-9

    def _takeoff_cb(
        self, request: Takeoff.Request, response: Takeoff.Response
    ) -> Takeoff.Response:
        if not self._safe_to_fly:
            # Must still answer, or rclpy raises TypeError on send_response and
            # tears down the spin loop (killing the link).
            self.get_logger().warning('takeoff refused: not safe to fly')
            response.success = False
            response.message = 'not safe to fly'
            return response
        duration = self._duration_to_seconds(request.duration)
        self.get_logger().info(
            f'takeoff height={request.height:.2f}m duration={duration:.2f}s '
            f'group_mask={request.group_mask}'
        )
        try:
            self._cf.high_level_commander.takeoff(
                float(request.height),
                duration
            )
#            sleep(duration * 0.75)
#            self._cf.high_level_commander.go_to(1, 1, 2, 0, 3)
            response.success = True
            response.message = ''
        except Exception as exc:
            self.get_logger().error(f'takeoff failed: {exc}')
            response.success = False
            response.message = str(exc)
        return response

    def _kill_cb(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> Land.Response:
        if not request.data:
            response.success = False
            response.message = 'kill not requested (data=false)'
            return response
        self.get_logger().info(
            f'Kill!'
        )
        try:
            self._cf.supervisor.send_emergency_stop()
            response.success = True
            response.message = ''
        except Exception as exc:
            self.get_logger().error(f'Kill failed: {exc}')
            response.success = False
            response.message = str(exc)
        return response

    def _land_cb(
        self, request: Land.Request, response: Land.Response
    ) -> Land.Response:
        duration = self._duration_to_seconds(request.duration)
        self.get_logger().info(
            f'land height={request.height:.2f}m duration={duration:.2f}s '
            f'group_mask={request.group_mask}'
        )
        try:
            self._cf.high_level_commander.land(
                float(request.height),
                duration,
                group_mask=int(request.group_mask),
            )
            response.success = True
            response.message = ''
        except Exception as exc:
            self.get_logger().error(f'land failed: {exc}')
            response.success = False
            response.message = str(exc)
        return response

    def _goto_cb(
        self, request: GoTo.Request, response: GoTo.Response
    ) -> GoTo.Response:
        duration = self._duration_to_seconds(request.duration)
        # crazybridge_interfaces specifies yaw in degrees, matching
        # crazyswarm2; cflib's high_level_commander.go_to expects radians.
        yaw_rad = math.radians(float(request.yaw))
        self.get_logger().info(
            f'go_to x={request.goal.x:.2f} y={request.goal.y:.2f} '
            f'z={request.goal.z:.2f} yaw={request.yaw:.2f}deg '
            f'duration={duration:.2f}s relative={request.relative} '
            f'group_mask={request.group_mask}'
        )
        try:
            self._cf.high_level_commander.go_to(
                float(request.goal.x),
                float(request.goal.y),
                float(request.goal.z),
                yaw_rad,
                duration,
                relative=bool(request.relative),
                group_mask=int(request.group_mask),
            )
            response.success = True
            response.message = ''
        except Exception as exc:
            self.get_logger().error(f'go_to failed: {exc}')
            response.success = False
            response.message = str(exc)
        return response

    def _spiral_cb(
        self, request: Spiral.Request, response: Spiral.Response
    ) -> Spiral.Response:
        duration = self._duration_to_seconds(request.duration)
        # As with go_to, the interface specifies the spiral angle in degrees;
        # cflib's high_level_commander.spiral expects radians.
        angle_rad = math.radians(float(request.angle))
        self.get_logger().info(
            f'spiral angle={request.angle:.2f}deg r0={request.r0:.2f} '
            f'rf={request.rf:.2f} ascent={request.ascent:.2f} '
            f'duration={duration:.2f}s sideways={request.sideways} '
            f'clockwise={request.clockwise} group_mask={request.group_mask}'
        )
        try:
            self._cf.high_level_commander.spiral(
                angle_rad,
                float(request.r0),
                float(request.rf),
                float(request.ascent),
                duration,
                sideways=bool(request.sideways),
                clockwise=bool(request.clockwise),
                group_mask=int(request.group_mask),
            )
            response.success = True
            response.message = ''
        except Exception as exc:
            self.get_logger().error(f'spiral failed: {exc}')
            response.success = False
            response.message = str(exc)
        return response

    # Log blocks that stream continuously; all are stopped during an upload and
    # restarted afterwards (the OOT blocks run at 10 ms and saturate the link).
    _STREAM_LOGS = (
        '_pos_log', '_q_log', '_pm_log', '_setpoint_log',
        '_input_log', '_pos_err_log', '_q_err_log', '_qd_log', '_ang_vel_err_log',
    )

    def _set_logs_running(self, running: bool) -> None:
        for name in self._STREAM_LOGS:
            log = getattr(self, name, None)
            if log is None:
                continue
            try:
                log.start() if running else log.stop()
            except Exception as exc:
                self.get_logger().warning(
                    f'could not {"restart" if running else "stop"} {name}: {exc}')

    def _write_traj_memory(self, mem) -> None:
        """Write mem.trajectory to the Crazyflie with a bounded wait.

        cflib's write_data_sync() waits on an Event with no timeout, so a lost
        acknowledgement hangs the caller forever. Here every attempt is bounded;
        streaming logs are stopped and extpos is throttled for the duration so
        the memory chunks are not starved on the radio link.
        """
        self._upload_active.set()
        self._set_logs_running(False)
        try:
            last_error = 'trajectory memory write failed'
            for attempt in range(1, self._upload_attempts + 1):
                done = threading.Event()
                result = {'ok': False}

                def finished(*_args, _r=result, _d=done):
                    _r['ok'] = True
                    _d.set()

                def failed(*_args, _d=done):
                    _d.set()

                started = time()
                mem.write_data(finished, write_failed_cb=failed)
                if done.wait(timeout=self._upload_attempt_timeout_s):
                    if result['ok']:
                        self.get_logger().info(
                            f'trajectory memory written in {time() - started:.2f} s '
                            f'(attempt {attempt})')
                        return
                    last_error = 'trajectory memory write failed'
                else:
                    last_error = (
                        f'trajectory memory write timed out after '
                        f'{self._upload_attempt_timeout_s:.0f} s')
                    # Drop the stalled request, otherwise the next write_data()
                    # would queue behind it and never start.
                    mem._write_finished_cb = None
                    mem._write_failed_cb = None
                    handler = self._cf.mem
                    try:
                        with handler._write_requests_lock:
                            handler._write_requests[mem.id] = []
                    except Exception as exc:
                        self.get_logger().warning(
                            f'could not clear stalled write request: {exc}')
                self.get_logger().warning(
                    f'{last_error} (attempt {attempt}/{self._upload_attempts})')
            raise RuntimeError(last_error)
        finally:
            self._set_logs_running(True)
            self._upload_active.clear()

    def _upload_traj_cb(
        self, request: UploadTrajectory.Request,
        response: UploadTrajectory.Response
    ) -> UploadTrajectory.Response:
        n = len(request.pieces)
        self.get_logger().info(
            f'upload_trajectory id={request.trajectory_id} pieces={n} '
            f'offset={request.piece_offset}'
        )
        try:
            if request.piece_offset != 0:
                raise ValueError(
                    'piece_offset != 0 is not supported (the whole trajectory '
                    'is rewritten on every upload)')
            if not 1 <= n <= MAX_TRAJ_PIECES:
                raise ValueError(
                    f'{n} pieces; the firmware memory holds 1..{MAX_TRAJ_PIECES}')
            if request.trajectory_id >= NUM_TRAJ_SLOTS:
                raise ValueError(
                    f'trajectory_id {request.trajectory_id} >= {NUM_TRAJ_SLOTS}')

            polys = []
            for i, piece in enumerate(request.pieces):
                axes = [list(piece.poly_x), list(piece.poly_y),
                        list(piece.poly_z), list(piece.poly_yaw)]
                if any(len(c) != TRAJ_COEFFS for c in axes):
                    raise ValueError(
                        f'piece {i}: every poly_* needs {TRAJ_COEFFS} coefficients')
                duration = self._duration_to_seconds(piece.duration)
                if duration <= 0.0:
                    raise ValueError(f'piece {i}: duration must be > 0')
                polys.append(Poly4D(duration, *(Poly4D.Poly(c) for c in axes)))

            mems = self._cf.mem.get_mems(MemoryElement.TYPE_TRAJ)
            if not mems:
                raise RuntimeError('Crazyflie exposes no trajectory memory')
            mem = mems[0]
            mem.trajectory = polys
            self._write_traj_memory(mem)
            self._cf.high_level_commander.define_trajectory(
                int(request.trajectory_id), 0, n)
            self.get_logger().info(f'trajectory {request.trajectory_id} uploaded')
            response.success = True
            response.message = ''
        except Exception as exc:
            self.get_logger().error(f'upload_trajectory failed: {exc}')
            response.success = False
            response.message = str(exc)
        return response

    def _start_traj_cb(
        self, request: StartTrajectory.Request,
        response: StartTrajectory.Response
    ) -> StartTrajectory.Response:
        self.get_logger().info(
            f'start_trajectory id={request.trajectory_id} '
            f'timescale={request.timescale:.2f} relative={request.relative} '
            f'reversed={request.reversed} group_mask={request.group_mask}'
        )
        try:
            self._cf.high_level_commander.start_trajectory(
                int(request.trajectory_id),
                float(request.timescale),
                relative_position=bool(request.relative),
                reversed=bool(request.reversed),
                group_mask=int(request.group_mask),
            )
            response.success = True
            response.message = ''
        except Exception as exc:
            self.get_logger().error(f'start_trajectory failed: {exc}')
            response.success = False
            response.message = str(exc)
        return response

    def shutdown(self) -> None:
        for name in (
            '_pos_log', '_q_log', '_setpoint_log',
            '_pos_err_log', '_q_err_log', '_qd_log', '_ang_vel_err_log',
        ):
            log = getattr(self, name, None)
            if log is None:
                continue
            try:
                log.stop()
            except Exception:
                pass
        self._cf.close_link()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CrazyBridge()
    # Multi-threaded so the (slow, blocking) trajectory upload can run in its own
    # callback group without stalling marker/safety callbacks; everything else
    # stays in the default mutually-exclusive group, i.e. serialised as before.
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
