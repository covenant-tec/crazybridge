"""Shared topic/service names and endpoint helpers for the crazybridge stack.

Keeping the pub/sub/service names and the request-building boilerplate in one
place ensures the bridge node, the TUI and the automated test node stay in sync.

Two views of each name exist:

* the **create-side** name the bridge passes to ``create_publisher`` /
  ``create_service`` -- some are node-private (``~/...``);
* the **fully-qualified** name other nodes subscribe or connect to.

``~/odometry`` from node ``crazybridge`` (no namespace) resolves to
``/crazybridge/odometry``, whereas a plain relative name like ``thrust``
resolves to ``/thrust``. Both forms are listed side by side below so they can
never drift apart.
"""

from __future__ import annotations

import threading
from time import monotonic, sleep

from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, PointStamped, Quaternion, Vector3
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32, Float32MultiArray
from std_srvs.srv import SetBool

from crazybridge_interfaces.srv import GoTo, Land, Spiral, Takeoff

BRIDGE_NODE = "crazybridge"


class TrackingMode:
    """Supported external motion capture tracking modes."""
    AUTO = "auto"
    RIGID_BODY = "rigid_body"
    MARKER = "marker"


class EstimatorMode:
    """Supported onboard state estimator selection modes."""
    AUTO = "auto"
    OOT = "oot"
    STOCK = "stock"


class AdaptMode:
    """Supported covariance adaptation policies for the Out-Of-Tree 13-state EKF."""

    STATIC = "static"
    MDP = "mdp"
    SMDP = "smdp"

    ALIASES = {
        "static": STATIC,
        "none": STATIC,
        "off": STATIC,
        "base": STATIC,
        "0": STATIC,
        "mdp": MDP,
        "markov": MDP,
        "1": MDP,
        "smdp": SMDP,
        "semimarkov": SMDP,
        "semi_markov": SMDP,
        "semi-markov": SMDP,
        "2": SMDP,
    }

    FIRMWARE_ID = {
        STATIC: 0,
        MDP: 1,
        SMDP: 2,
    }

    @classmethod
    def normalize(cls, raw: str | None) -> str | None:
        if raw is None:
            return None
        return cls.ALIASES.get(str(raw).strip().lower())


class EkfOotDiag:
    """CRTP and ROS 2 schema for Out-Of-Tree EKF control and diagnostics.

    Published on ``Topics.EKF_DIAG`` (``std_msgs/Float32MultiArray``) only when
    the connected Crazyflie exposes the ``ekfOot`` log group in its TOC. When
    flying stock Bitcraze / lab firmware, the topic stays silent and CSV exports
    preserve their original 35-column layout.
    """

    LOG_GROUP = "ekfOot"
    PARAM_USE_STOCK = "ekfOot.useStock"
    PARAM_MODE = "ekfOot.mode"
    PARAM_RESET = "ekfOot.reset"
    PARAM_CALIB_BIAS = "ekfOot.calibBias"
    PARAM_ENABLE_CLAMP = "ekfOot.enableClamp"
    PARAM_RECONN_SKIP = "ekfOot.reconnSkip"

    # Bitmask flags exported in ``ekfOot.flags`` (uint8).
    FLAG_BIAS_READY = 1 << 0
    FLAG_POSE_RECENT = 1 << 1
    FLAG_CHOL_FAIL_SEEN = 1 << 2
    FLAG_NAN_RESET_SEEN = 1 << 3
    FLAG_STOCK_ACTIVE = 1 << 4

    # Ordered tuple of (crtp_var_name, cflib_type, rerun_entity_path, rgb_color).
    # Total CRTP payload: 4 + 4 + 2 + 2 + 1 + 4 + 4 = 21 bytes (<= 26 B limit).
    FIELDS = (
        ("dMahal", "float", "/ekf_oot/d_mahal", [255, 165, 0]),
        ("trP", "float", "/ekf_oot/tr_p", [0, 200, 255]),
        ("cholFails", "uint16_t", "/ekf_oot/chol_fails", [255, 60, 60]),
        ("nanResets", "uint16_t", "/ekf_oot/nan_resets", [255, 0, 180]),
        ("flags", "uint8_t", "/ekf_oot/flags", [120, 255, 120]),
        ("rtPred", "float", "/ekf_oot/rt_pred", [180, 180, 255]),
        ("rtUpd", "float", "/ekf_oot/rt_upd", [255, 220, 100]),
    )
    NUM_FIELDS = len(FIELDS)


class EkfBiasDiag:
    """CRTP and ROS 2 schema for calibrated stationary IMU bias (``ekfBias.*``).

    Total CRTP payload: 6 * 4 = 24 bytes (<= 26 B limit), subscribed in its own
    dedicated low-rate block (default 2 Hz / 500 ms) in both ``stock`` and ``oot``
    estimator modes so existing log blocks retain their full rates.

    Units, axes (FLU body frame: +X front, +Y left, +Z up), and sign match
    ``Topics.IMU``: ``biasAx, biasAy, biasAz`` are in m/s^2 (with +9.81 m/s^2
    gravity subtracted on +Z) and ``biasGx, biasGy, biasGz`` are in rad/s, so
    ``imu_a - imu_bias_a`` is ~[0, 0, +9.81] m/s^2 at level rest and
    ``imu_g - imu_bias_g`` is ~[0, 0, 0] rad/s at rest.
    """

    LOG_GROUP = "ekfBias"

    FIELDS = (
        ("biasAx", "float", "/ekf_bias/ax", [255, 110, 110]),
        ("biasAy", "float", "/ekf_bias/ay", [110, 255, 110]),
        ("biasAz", "float", "/ekf_bias/az", [110, 160, 255]),
        ("biasGx", "float", "/ekf_bias/gx", [255, 180, 90]),
        ("biasGy", "float", "/ekf_bias/gy", [180, 255, 90]),
        ("biasGz", "float", "/ekf_bias/gz", [180, 130, 255]),
    )
    NUM_FIELDS = len(FIELDS)


class EkfAdaptDiag:
    """CRTP and ROS 2 schema for Adaptive EKF (MDP 2D / SMDP) decision telemetry.

    Published on ``Topics.EKF_ADAPT`` (``std_msgs/Float32MultiArray``) only when
    the connected Crazyflie exposes ``ekfAdapt`` in its TOC and OOT EKF is active.
    """

    LOG_GROUP = "ekfAdapt"

    # Total CRTP payload: 4 + 4 + 2 + 2 + 2 + 1 + 1 + 1 = 17 bytes (<= 26 B limit).
    FIELDS = (
        ("trQ", "float", "/ekf_adapt/tr_q", [80, 180, 255]),
        ("trR", "float", "/ekf_adapt/tr_r", [255, 100, 100]),
        ("decisions", "uint16_t", "/ekf_adapt/decisions", [255, 200, 60]),
        ("qrMods", "uint16_t", "/ekf_adapt/qr_mods", [255, 130, 60]),
        ("tau", "uint16_t", "/ekf_adapt/tau", [160, 255, 160]),
        ("state2d", "uint8_t", "/ekf_adapt/state2d", [200, 140, 255]),
        ("action", "uint8_t", "/ekf_adapt/action", [255, 90, 200]),
        ("mode", "uint8_t", "/ekf_adapt/mode", [180, 220, 220]),
    )
    NUM_FIELDS = len(FIELDS)


class Create:
    """Names the bridge node passes to create_publisher / create_service."""

    ODOM = "~/odometry"
    IMU = "~/imu"
    IMU_BIAS = "~/imu_bias"
    BATTERY = "~/battery"
    BATTERY_VOLTAGE = "~/battery_voltage"
    EKF_DIAG = "~/ekf_diag"
    EKF_ADAPT = "~/ekf_adapt"
    THRUST = "thrust"
    TORQUE = "torque"
    POS_ERROR = "pos_error"
    SETPOINT = "setpoint"
    ORI_DESIRED = "orientation/desired"
    ORI_ERROR = "orientation/error"
    MARKER = "optitrack/marker"
    RIGID_BODY = "optitrack/rigid_body"
    SRV_TAKEOFF = "~/takeoff"
    SRV_LAND = "~/land"
    SRV_GOTO = "~/go_to"
    SRV_SPIRAL = "~/spiral"
    SRV_KILL = "~/kill"


class Topics:
    """Fully-qualified topic names for clients (TUI, test node, rerun)."""

    ODOM = "/crazybridge/odometry"
    IMU = "/crazybridge/imu"
    IMU_BIAS = "/crazybridge/imu_bias"
    BATTERY = "/crazybridge/battery"
    BATTERY_VOLTAGE = "/crazybridge/battery_voltage"
    EKF_DIAG = "/crazybridge/ekf_diag"
    EKF_ADAPT = "/crazybridge/ekf_adapt"
    THRUST = "/thrust"
    TORQUE = "/torque"
    POS_ERROR = "/pos_error"
    SETPOINT = "/setpoint"
    ORI_DESIRED = "/orientation/desired"
    ORI_ERROR = "/orientation/error"
    MARKER = "/optitrack/marker"
    RIGID_BODY = "/optitrack/rigid_body"


class Services:
    """Fully-qualified service names for clients."""

    TAKEOFF = "/crazybridge/takeoff"
    LAND = "/crazybridge/land"
    GOTO = "/crazybridge/go_to"
    SPIRAL = "/crazybridge/spiral"
    KILL = "/crazybridge/kill"


def seconds_to_duration(seconds: float) -> Duration:
    d = Duration()
    d.sec = int(seconds)
    d.nanosec = int((seconds - int(seconds)) * 1e9)
    return d


class BridgePublishers:
    """All telemetry publishers the bridge node exposes.

    Constructed with the bridge node; each publisher is a plain attribute so the
    node can publish via ``self._pubs.thrust.publish(msg)`` etc.
    """

    def __init__(self, node) -> None:
        self.odom = node.create_publisher(Odometry, Create.ODOM, 1)
        self.battery = node.create_publisher(Float32, Create.BATTERY, 1)
        self.battery_voltage = node.create_publisher(
            Float32, Create.BATTERY_VOLTAGE, 10
        )
        self.thrust = node.create_publisher(Float32, Create.THRUST, 1)
        self.torque = node.create_publisher(Vector3, Create.TORQUE, 1)
        self.pos_error = node.create_publisher(Vector3, Create.POS_ERROR, 1)
        self.setpoint = node.create_publisher(PointStamped, Create.SETPOINT, 1)
        self.orientation_desired = node.create_publisher(
            Quaternion, Create.ORI_DESIRED, 1
        )
        self.orientation_error = node.create_publisher(Quaternion, Create.ORI_ERROR, 1)
        self.imu = node.create_publisher(Imu, Create.IMU, 10)
        self.imu_bias = node.create_publisher(Float32MultiArray, Create.IMU_BIAS, 10)
        self.ekf_diag = node.create_publisher(Float32MultiArray, Create.EKF_DIAG, 10)
        self.ekf_adapt = node.create_publisher(Float32MultiArray, Create.EKF_ADAPT, 10)


class BridgeClientNode(Node):
    """Base node for clients that drive the bridge (TUI, automated test).

    Owns the parameter accessors (``_sp``/``_dp``/``_bp``/``_ip``), declares the
    service-name parameters (defaulting to the canonical fully-qualified names),
    creates the service clients as attributes, and exposes the request builders.
    Subclasses call ``super().__init__(node_name)`` and then add their own
    subscriptions, tuning parameters and calling style (the TUI calls
    asynchronously; the test node blocks).
    """

    def __init__(self, node_name: str, *, subscribe_odometry: bool = True) -> None:
        super().__init__(node_name)

        self.declare_parameter("takeoff_srv", Services.TAKEOFF)
        self.declare_parameter("land_srv", Services.LAND)
        self.declare_parameter("goto_srv", Services.GOTO)
        self.declare_parameter("spiral_srv", Services.SPIRAL)
        self.declare_parameter("kill_srv", Services.KILL)

        self.takeoff_cli = self.create_client(Takeoff, self._sp("takeoff_srv"))
        self.land_cli = self.create_client(Land, self._sp("land_srv"))
        self.goto_cli = self.create_client(GoTo, self._sp("goto_srv"))
        self.spiral_cli = self.create_client(Spiral, self._sp("spiral_srv"))
        self.kill_cli = self.create_client(SetBool, self._sp("kill_srv"))

        # Odometry is the one subscription every client needs. The base keeps
        # the latest message; subclasses override on_odometry() to also record
        # or react to it.
        self._odom_lock = threading.Lock()
        self._latest_odom: Odometry | None = None
        if subscribe_odometry:
            self.declare_parameter("odom_topic", Topics.ODOM)
            self.create_subscription(
                Odometry, self._sp("odom_topic"), self._ingest_odom, 10
            )

    # -- parameter accessors -------------------------------------------------
    def _sp(self, name: str) -> str:
        return self.get_parameter(name).get_parameter_value().string_value

    def _dp(self, name: str) -> float:
        return self.get_parameter(name).get_parameter_value().double_value

    def _bp(self, name: str) -> bool:
        return self.get_parameter(name).get_parameter_value().bool_value

    def _ip(self, name: str) -> int:
        return self.get_parameter(name).get_parameter_value().integer_value

    # -- odometry ------------------------------------------------------------
    def _ingest_odom(self, msg: Odometry) -> None:
        with self._odom_lock:
            self._latest_odom = msg
        self.on_odometry(msg)

    def on_odometry(self, msg: Odometry) -> None:
        """Hook for subclasses to record/react to odometry; default no-op."""

    def latest_odom(self) -> Odometry | None:
        with self._odom_lock:
            return self._latest_odom

    def has_odometry(self) -> bool:
        with self._odom_lock:
            return self._latest_odom is not None

    # -- service introspection ----------------------------------------------
    def service_clients(self) -> dict:
        return {
            "takeoff": self.takeoff_cli,
            "land": self.land_cli,
            "go_to": self.goto_cli,
            "spiral": self.spiral_cli,
            "kill": self.kill_cli,
        }

    def service_ready(self) -> dict[str, bool]:
        return {k: c.service_is_ready() for k, c in self.service_clients().items()}

    def wait_until_ready(self, timeout_s: float, require_odom: bool = True) -> None:
        """Block until all services are up (and odometry seen), or raise."""
        deadline = monotonic() + timeout_s
        while monotonic() < deadline:
            if all(self.service_ready().values()) and (
                not require_odom or self.has_odometry()
            ):
                return
            sleep(0.1)
        missing = [n for n, r in self.service_ready().items() if not r]
        raise RuntimeError(
            f"startup timeout after {timeout_s}s (missing services: {missing}, "
            f"odometry: {self.has_odometry()})"
        )

    # -- request builders (single source of truth for field layout) ----------
    @staticmethod
    def takeoff_request(height: float, duration_s: float, group_mask: int = 0):
        req = Takeoff.Request()
        req.height = float(height)
        req.duration = seconds_to_duration(duration_s)
        req.group_mask = int(group_mask)
        return req

    @staticmethod
    def land_request(height: float, duration_s: float, group_mask: int = 0):
        req = Land.Request()
        req.height = float(height)
        req.duration = seconds_to_duration(duration_s)
        req.group_mask = int(group_mask)
        return req

    @staticmethod
    def goto_request(
        x: float,
        y: float,
        z: float,
        yaw_deg: float,
        duration_s: float,
        relative: bool = False,
        group_mask: int = 0,
    ):
        req = GoTo.Request()
        req.relative = bool(relative)
        req.goal = Point(x=float(x), y=float(y), z=float(z))
        req.yaw = float(yaw_deg)
        req.duration = seconds_to_duration(duration_s)
        req.group_mask = int(group_mask)
        return req

    @staticmethod
    def spiral_request(
        angle_deg: float,
        r0: float,
        rf: float,
        ascent: float,
        duration_s: float,
        sideways: bool = False,
        clockwise: bool = False,
        group_mask: int = 0,
    ):
        req = Spiral.Request()
        req.angle = float(angle_deg)
        req.r0 = float(r0)
        req.rf = float(rf)
        req.ascent = float(ascent)
        req.duration = seconds_to_duration(duration_s)
        req.sideways = bool(sideways)
        req.clockwise = bool(clockwise)
        req.group_mask = int(group_mask)
        return req

    @staticmethod
    def bool_request(value: bool = True):
        req = SetBool.Request()
        req.data = bool(value)
        return req
