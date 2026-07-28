"""ROS2 bridge for a single Crazyflie, ported from forerunner2/crazybridge.

Mirrors crazybridge.cpp: brings up a cflib link (defaulting to the CrazySim
UDP URI), enables the high-level commander and Kalman estimator, streams
position + quaternion as nav_msgs/Odometry, forwards mocap marker fixes via
the external-position channel, and exposes takeoff/land/go_to services that
drive the high-level commander.
"""
from __future__ import annotations

import math
import os
import threading

import rclpy
from time import sleep, time
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from logging import getLogger

from ament_index_python.packages import get_package_share_directory

from std_msgs.msg import Float32, Header
from std_srvs.srv import SetBool
from geometry_msgs.msg import PointStamped, Vector3, Quaternion, PointStamped, Point
from nav_msgs.msg import Odometry

from crazybridge_interfaces.srv import GoTo, Land, Spiral, Takeoff

from crazybridge.interface import BridgePublishers, Create

logger = getLogger()

PARAM_HL_COMMANDER = 'commander.enHighLevel'
PARAM_ESTIMATOR = 'stabilizer.estimator'
PARAM_CONTROLLER = 'stabilizer.controller'
ESTIMATOR_KALMAN = 2
ESTIMATOR_COMPLEMENTARY = 1

OOT_PARAM_GROUP = 'ootParams'
OOT_TRANS_KP = ('trans_kp_x', 'trans_kp_y', 'trans_kp_z')
OOT_TRANS_KD = ('trans_kd_x', 'trans_kd_y', 'trans_kd_z')
OOT_TRANS_KI = ('trans_ki_x', 'trans_ki_y', 'trans_ki_z')
OOT_ROT_KP = ('rot_kp_x', 'rot_kp_y', 'rot_kp_z')
OOT_ROT_KD = ('rot_kd_x', 'rot_kd_y', 'rot_kd_z')
OOT_ROT_KI = ('rot_ki_x', 'rot_ki_y', 'rot_ki_z')


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

        self._takeoff_srv = self.create_service(
            Takeoff, Create.SRV_TAKEOFF, self._takeoff_cb)
        self._land_srv = self.create_service(
            Land, Create.SRV_LAND, self._land_cb)
        self._goto_srv = self.create_service(
            GoTo, Create.SRV_GOTO, self._goto_cb)
        self._spiral_srv = self.create_service(
            Spiral, Create.SRV_SPIRAL, self._spiral_cb)
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
#        self._pm_log = LogConfig(name="battery", period_in_ms=extra_period)
#        self._pm_log.add_variable("pm.batteryLevel", "uint8_t")

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
#            self._cf.log.add_config(self._pm_log)
            self._cf.log.add_config(self._setpoint_log)
            self._pos_log.data_received_cb.add_callback(self._on_pos_log)
            self._q_log.data_received_cb.add_callback(self._on_q_log)
#            self._pm_log.data_received_cb.add_callback(self._on_pm_log)
            self._setpoint_log.data_received_cb.add_callback(
                self._on_setpoint_log)
            self._pos_log.start()
            self._q_log.start()
#            self._pm_log.start()
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

            self._input_log.start()
            self._pos_err_log.start()
            self._q_err_log.start()
            self._qd_log.start()
            self._ang_vel_err_log.start()
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
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

        values: list[float] = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                values.append(float(line))

        if len(values) < 12:
            raise ValueError(
                f'pid.conf {path} needs 12 numeric values, got {len(values)}'
            )

        trans_kp = values[0:3]
        trans_kd = values[3:6]
        trans_ki = values[6:9]
        trans_h = values[9:12]
        rot_kp = values[12:15]
        rot_kd = values[15:18]
        rot_ki = values[18:21]
        rot_h = values[21:24]

        self.get_logger().info(f'Loading PID gains from {path}')
        self.get_logger().info(f'  trans kp = {trans_kp}, kd = {trans_kd}, ki = {trans_ki}')  # noqa
        self.get_logger().info(f'  trans homogenous   emax = {trans_h[0]} mu = {trans_h[1]} gamma = {trans_h[2]}')  # noqa
        self.get_logger().info(f'  rot   kp = {rot_kp}, kd = {rot_kd}, ki = {rot_ki}')  # noqa
        self.get_logger().info(f'  rot homogenous   emax = {rot_h[0]} mu = {rot_h[1]} gamma = {rot_h[2]}')  # noqa

        self._cf.param.set_value("ootParams.trans_kp_x", trans_kp[0])
        self._cf.param.set_value("ootParams.trans_kp_y", trans_kp[1])
        self._cf.param.set_value("ootParams.trans_kp_z", trans_kp[2])

        self._cf.param.set_value("ootParams.trans_kd_x", trans_kd[0])
        self._cf.param.set_value("ootParams.trans_kd_y", trans_kd[1])
        self._cf.param.set_value("ootParams.trans_kd_z", trans_kd[2])

        self._cf.param.set_value("ootParams.trans_ki_x", trans_ki[0])
        self._cf.param.set_value("ootParams.trans_ki_y", trans_ki[1])
        self._cf.param.set_value("ootParams.trans_ki_z", trans_ki[2])

        self._cf.param.set_value("ootParams.trans_emax", trans_h[0])
        self._cf.param.set_value("ootParams.trans_mu", trans_h[1])
        self._cf.param.set_value("ootParams.trans_gamma", trans_h[2])

        self._cf.param.set_value("ootParams.rot_kp_x", rot_kp[0])
        self._cf.param.set_value("ootParams.rot_kp_y", rot_kp[1])
        self._cf.param.set_value("ootParams.rot_kp_z", rot_kp[2])

        self._cf.param.set_value("ootParams.rot_kd_x", rot_kd[0])
        self._cf.param.set_value("ootParams.rot_kd_y", rot_kd[1])
        self._cf.param.set_value("ootParams.rot_kd_z", rot_kd[2])

        self._cf.param.set_value("ootParams.rot_ki_x", rot_ki[0])
        self._cf.param.set_value("ootParams.rot_ki_y", rot_ki[1])
        self._cf.param.set_value("ootParams.rot_ki_z", rot_ki[2])

        self._cf.param.set_value("ootParams.rot_emax", rot_h[0])
        self._cf.param.set_value("ootParams.rot_mu", rot_h[1])
        self._cf.param.set_value("ootParams.rot_gamma", rot_h[2])

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
            self._cf.extpos.send_extpos(point.x, point.y, point.z)

    def _safety_check(self):
        if not self._cf.is_connected():
            self.get_logger().error("Disconnected from crazyflie. Trying to reconnect")
            with self._safe_to_fly:
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
            return 
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
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
