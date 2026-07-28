"""Textual TUI for driving the crazybridge node.

Spins a small rclpy node in a background thread that subscribes to the
bridge's odometry and holds service clients for takeoff, land and go_to.
Hotkeys:

    t           takeoff (uses takeoff_height_m / takeoff_duration_s params)
    l           land    (uses land_duration_s)
    k           kill    
    w / s       nudge +x / -x by nudge_step_m
    a / d       nudge +y / -y
    r / f       nudge +z / -z
    z / x       yaw     +nudge_yaw_deg / -nudge_yaw_deg
    g           go to absolute position (opens an input dialog)
    c           spiral (opens an input dialog; high-level commander)
    q          quit

Nudges fan out as relative go_to calls with goto_duration_s seconds each.
The 'g' dialog fires a single absolute go_to (relative=False).
"""
from __future__ import annotations

import threading
from collections import deque
from queue import Empty, Queue

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from std_srvs.srv import SetBool

from crazybridge_interfaces.srv import GoTo, Land, Spiral, Takeoff

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Input, Label, Static


def _seconds_to_duration(seconds: float) -> Duration:
    d = Duration()
    d.sec = int(seconds)
    d.nanosec = int((seconds - int(seconds)) * 1e9)
    return d


class BridgeClient(Node):
    def __init__(self) -> None:
        super().__init__('crazybridge_tui')

        self.declare_parameter('odom_topic', '/crazybridge/odometry')
        self.declare_parameter('battery_topic', '/crazybridge/battery')
        self.declare_parameter('takeoff_srv', '/crazybridge/takeoff')
        self.declare_parameter('land_srv', '/crazybridge/land')
        self.declare_parameter('goto_srv', '/crazybridge/go_to')
        self.declare_parameter('spiral_srv', '/crazybridge/spiral')
        self.declare_parameter('kill_srv', '/crazybridge/kill')
        self.declare_parameter('test_srv', '/crazybridge/test')

        self.declare_parameter('takeoff_height_m', 1.0)
        self.declare_parameter('takeoff_duration_s', 2.0)
        self.declare_parameter('land_duration_s', 3.0)
        self.declare_parameter('goto_duration_s', 1.5)
        self.declare_parameter('spiral_duration_s', 4.0)
        self.declare_parameter('nudge_step_m', 0.2)
        self.declare_parameter('nudge_yaw_deg', 15.0)

        odom_topic = self._sp('odom_topic')
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)
        batt_topic = self._sp('battery_topic')
        self.create_subscription(Float32, batt_topic, self._batt_cb, 10)

        self._takeoff_cli = self.create_client(Takeoff, self._sp('takeoff_srv'))
        self._kill_cli = self.create_client(SetBool, self._sp('kill_srv'))
        self._test_cli = self.create_client(SetBool, self._sp('test_srv'))
        self._land_cli = self.create_client(Land, self._sp('land_srv'))
        self._goto_cli = self.create_client(GoTo, self._sp('goto_srv'))
        self._spiral_cli = self.create_client(Spiral, self._sp('spiral_srv'))

        self._state_lock = threading.Lock()
        self._latest_odom: Odometry | None = None
        self._latest_batt: float | None = None
        self.events: Queue[str] = Queue()

    def _sp(self, name: str) -> str:
        return self.get_parameter(name).get_parameter_value().string_value

    def _dp(self, name: str) -> float:
        return self.get_parameter(name).get_parameter_value().double_value

    def _odom_cb(self, msg: Odometry) -> None:
        with self._state_lock:
            self._latest_odom = msg

    def snapshot(self) -> Odometry | None:
        with self._state_lock:
            return (self._latest_odom, self._latest_batt)

    def service_status(self) -> dict[str, bool]:
        return {
            'takeoff': self._takeoff_cli.service_is_ready(),
            'land': self._land_cli.service_is_ready(),
            'go_to': self._goto_cli.service_is_ready(),
            'spiral': self._spiral_cli.service_is_ready(),
            'kill': self._kill_cli.service_is_ready(),
            'test': self._test_cli.service_is_ready(),
        }

    def _on_response(self, label: str, future) -> None:
        try:
            r = future.result()
            self.events.put(
                f'{label} -> success={r.success}'
                + (f' msg="{r.message}"' if r.message else '')
            )
        except Exception as exc:
            self.events.put(f'{label} -> exception: {exc}')

    def _batt_cb(self, msg: Float32):
        with self._state_lock:
            self._latest_batt = msg.data

    def call_takeoff(self) -> None:
        if not self._takeoff_cli.service_is_ready():
            self.events.put('takeoff: service not available')
            return
        req = Takeoff.Request()
        req.height = float(self._dp('takeoff_height_m'))
        req.duration = _seconds_to_duration(self._dp('takeoff_duration_s'))
        req.group_mask = 0
        fut = self._takeoff_cli.call_async(req)
        fut.add_done_callback(
            lambda f: self._on_response(f'takeoff h={req.height:.2f}', f)
        )

    def call_kill(self) -> None:
        if not self._kill_cli.service_is_ready():
            self.events.put('kill: service not available')
            return
        req = SetBool.Request()
        req.data = True
        fut = self._kill_cli.call_async(req)
        fut.add_done_callback(
            lambda f: self._on_response(f'Killed! {f.message}')
        )

    def call_test(self) -> None:
        if not self._test_cli.service_is_ready():
            self.events.put('Test: service not available')
            return
        req = SetBool.Request()
        req.data = True
        fut = self._test_cli.call_async(req)
        fut.add_done_callback(
            lambda f: self._on_response(f'Testing! {f.message}')
        )

    def call_land(self) -> None:
        if not self._land_cli.service_is_ready():
            self.events.put('land: service not available')
            return
        req = Land.Request()
        req.height = 0.0
        req.duration = _seconds_to_duration(self._dp('land_duration_s'))
        req.group_mask = 0
        fut = self._land_cli.call_async(req)
        fut.add_done_callback(lambda f: self._on_response('land', f))

    def call_nudge(self, dx: float, dy: float, dz: float, dyaw_deg: float = 0.0) -> None:
        if not self._goto_cli.service_is_ready():
            self.events.put('go_to: service not available')
            return
        req = GoTo.Request()
        req.relative = True
        req.goal = Point(x=float(dx), y=float(dy), z=float(dz))
        req.yaw = float(dyaw_deg)
        req.duration = _seconds_to_duration(self._dp('goto_duration_s'))
        req.group_mask = 0
        fut = self._goto_cli.call_async(req)
        fut.add_done_callback(
            lambda f: self._on_response(
                f'nudge dx={dx:+.2f} dy={dy:+.2f} dz={dz:+.2f} dyaw={dyaw_deg:+.1f}',
                f,
            )
        )

    def call_goto_abs(
        self, x: float, y: float, z: float, yaw_deg: float = 0.0
    ) -> None:
        if not self._goto_cli.service_is_ready():
            self.events.put('go_to: service not available')
            return
        req = GoTo.Request()
        req.relative = False
        req.goal = Point(x=float(x), y=float(y), z=float(z))
        req.yaw = float(yaw_deg)
        req.duration = _seconds_to_duration(self._dp('goto_duration_s'))
        req.group_mask = 0
        fut = self._goto_cli.call_async(req)
        fut.add_done_callback(
            lambda f: self._on_response(
                f'goto x={x:+.2f} y={y:+.2f} z={z:+.2f} yaw={yaw_deg:+.1f}',
                f,
            )
        )

    def call_spiral(
        self,
        angle_deg: float,
        r0: float,
        rf: float,
        ascent: float,
        duration_s: float,
        sideways: bool = False,
        clockwise: bool = False,
    ) -> None:
        if not self._spiral_cli.service_is_ready():
            self.events.put('spiral: service not available')
            return
        req = Spiral.Request()
        req.angle = float(angle_deg)
        req.r0 = float(r0)
        req.rf = float(rf)
        req.ascent = float(ascent)
        req.duration = _seconds_to_duration(duration_s)
        req.sideways = bool(sideways)
        req.clockwise = bool(clockwise)
        req.group_mask = 0
        fut = self._spiral_cli.call_async(req)
        fut.add_done_callback(
            lambda f: self._on_response(
                f'spiral angle={angle_deg:+.1f} r0={r0:.2f} rf={rf:.2f} '
                f'ascent={ascent:+.2f}'
                + (' sideways' if sideways else '')
                + (' cw' if clockwise else ' ccw'),
                f,
            )
        )

    @property
    def spiral_duration(self) -> float:
        return float(self._dp('spiral_duration_s'))

    @property
    def step(self) -> float:
        return float(self._dp('nudge_step_m'))

    @property
    def yaw_step(self) -> float:
        return float(self._dp('nudge_yaw_deg'))


class GoToScreen(ModalScreen[tuple[float, float, float, float] | None]):
    """Modal that collects an absolute x/y/z/yaw target.

    Dismisses with the parsed (x, y, z, yaw_deg) tuple, or None on cancel.
    """

    CSS = """
    GoToScreen { align: center middle; }
    #goto-dialog {
        width: 44;
        height: auto;
        border: round green;
        background: $surface;
        padding: 1 2;
    }
    #goto-dialog Input { margin-bottom: 1; }
    #goto-buttons { height: auto; align: center middle; }
    #goto-buttons Button { margin: 0 1; }
    #goto-error { color: red; height: auto; }
    """

    BINDINGS = [Binding('escape', 'cancel', 'Cancel')]

    def __init__(self, seed: tuple[float, float, float] | None = None) -> None:
        super().__init__()
        self._seed = seed

    def compose(self) -> ComposeResult:
        sx, sy, sz = self._seed if self._seed else (0.0, 0.0, 1.0)
        with Vertical(id='goto-dialog'):
            yield Label('Go to absolute position (world frame)')
            yield Input(value=f'{sx:.2f}', placeholder='x (m)', id='goto-x')
            yield Input(value=f'{sy:.2f}', placeholder='y (m)', id='goto-y')
            yield Input(value=f'{sz:.2f}', placeholder='z (m)', id='goto-z')
            yield Input(value='0.0', placeholder='yaw (deg)', id='goto-yaw')
            yield Label('', id='goto-error')
            with Vertical(id='goto-buttons'):
                yield Button('Go', variant='success', id='goto-go')
                yield Button('Cancel', variant='error', id='goto-cancel')

    def on_mount(self) -> None:
        self.query_one('#goto-x', Input).focus()

    def _submit(self) -> None:
        try:
            x = float(self.query_one('#goto-x', Input).value)
            y = float(self.query_one('#goto-y', Input).value)
            z = float(self.query_one('#goto-z', Input).value)
            yaw = float(self.query_one('#goto-yaw', Input).value or '0')
        except ValueError:
            self.query_one('#goto-error', Label).update('invalid number')
            return
        self.dismiss((x, y, z, yaw))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._submit()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == 'goto-go':
            self._submit()
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class SpiralScreen(ModalScreen[dict | None]):
    """Modal that collects high-level-commander spiral parameters.

    Dismisses with a dict of spiral kwargs, or None on cancel.
    """

    CSS = """
    SpiralScreen { align: center middle; }
    #spiral-dialog {
        width: 48;
        height: auto;
        border: round green;
        background: $surface;
        padding: 1 2;
    }
    #spiral-dialog Input { margin-bottom: 1; }
    #spiral-flags { height: auto; }
    #spiral-buttons { height: auto; align: center middle; }
    #spiral-buttons Button { margin: 0 1; }
    #spiral-error { color: red; height: auto; }
    """

    BINDINGS = [Binding('escape', 'cancel', 'Cancel')]

    def __init__(self, duration_s: float) -> None:
        super().__init__()
        self._duration = duration_s

    def compose(self) -> ComposeResult:
        with Vertical(id='spiral-dialog'):
            yield Label('Spiral (high-level commander)')
            yield Input(value='360.0', placeholder='angle (deg)', id='sp-angle')
            yield Input(value='0.50', placeholder='r0 start radius (m)', id='sp-r0')
            yield Input(value='0.50', placeholder='rF end radius (m)', id='sp-rf')
            yield Input(value='0.00', placeholder='ascent (m)', id='sp-ascent')
            yield Input(
                value=f'{self._duration:.2f}',
                placeholder='duration (s)',
                id='sp-duration',
            )
            with Vertical(id='spiral-flags'):
                yield Button('sideways: OFF', id='sp-sideways')
                yield Button('direction: CCW', id='sp-clockwise')
            yield Label('', id='spiral-error')
            with Vertical(id='spiral-buttons'):
                yield Button('Go', variant='success', id='sp-go')
                yield Button('Cancel', variant='error', id='sp-cancel')

    def on_mount(self) -> None:
        self._sideways = False
        self._clockwise = False
        self.query_one('#sp-angle', Input).focus()

    def _submit(self) -> None:
        try:
            params = dict(
                angle_deg=float(self.query_one('#sp-angle', Input).value),
                r0=float(self.query_one('#sp-r0', Input).value),
                rf=float(self.query_one('#sp-rf', Input).value),
                ascent=float(self.query_one('#sp-ascent', Input).value or '0'),
                duration_s=float(self.query_one('#sp-duration', Input).value),
                sideways=self._sideways,
                clockwise=self._clockwise,
            )
        except ValueError:
            self.query_one('#spiral-error', Label).update('invalid number')
            return
        self.dismiss(params)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._submit()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == 'sp-sideways':
            self._sideways = not self._sideways
            event.button.label = f'sideways: {"ON" if self._sideways else "OFF"}'
        elif bid == 'sp-clockwise':
            self._clockwise = not self._clockwise
            event.button.label = (
                f'direction: {"CW" if self._clockwise else "CCW"}'
            )
        elif bid == 'sp-go':
            self._submit()
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class CrazyBridgeTUI(App):
    CSS = """
    Screen { layout: vertical; }
    #status {
        height: 11;
        border: round green;
        padding: 0 1;
    }
    #log {
        border: round blue;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding('t', 'takeoff', 'Takeoff'),
        Binding('l', 'land', 'Land'),
        Binding('k', 'kill', 'Kill'),
        Binding('p', 'test', 'Test'),
        Binding('w', 'nudge_fwd', '+X'),
        Binding('s', 'nudge_back', '-X'),
        Binding('a', 'nudge_left', '+Y'),
        Binding('d', 'nudge_right', '-Y'),
        Binding('r', 'nudge_up', '+Z'),
        Binding('f', 'nudge_down', '-Z'),
        Binding('z', 'yaw_left', '+Yaw'),
        Binding('x', 'yaw_right', '-Yaw'),
        Binding('g', 'goto', 'GoTo'),
        Binding('c', 'spiral', 'Spiral'),
        Binding('q', 'quit', 'Quit'),
    ]

    def __init__(self, client: BridgeClient) -> None:
        super().__init__()
        self._client = client
        self._log: deque[str] = deque(maxlen=14)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Vertical(
            Static('', id='status'),
            Static('', id='log'),
        )
        yield Footer()

    def on_mount(self) -> None:
        self.set_interval(0.1, self._refresh)

    def _drain_events(self) -> None:
        while True:
            try:
                self._log.append(self._client.events.get_nowait())
            except Empty:
                return

    def _refresh(self) -> None:
        self._drain_events()
        (odom, batt) = self._client.snapshot()
        svc = self._client.service_status()
        svc_line = '  '.join(
            f'{k}={"OK" if v else "--"}' for k, v in svc.items()
        )
        if odom is None:
            body = (
                'waiting for odometry...\n'
                f'step={self._client.step:.2f}m  yaw_step={self._client.yaw_step:.1f}deg\n'
                f'services: {svc_line}\n'
            )
        else:
            p = odom.pose.pose.position
            q = odom.pose.pose.orientation
            stamp = odom.header.stamp
            body = (
                f'pos   x={p.x:+8.3f}  y={p.y:+8.3f}  z={p.z:+8.3f}\n'
                f'quat  x={q.x:+8.3f}  y={q.y:+8.3f}  z={q.z:+8.3f}  w={q.w:+8.3f}\n'
                f'frame={odom.header.frame_id} child={odom.child_frame_id} '
                f'stamp={stamp.sec}.{stamp.nanosec:09d}\n'
                f'step={self._client.step:.2f}m  yaw_step={self._client.yaw_step:.1f}deg\n'
                f'services: {svc_line}\n'
            )

        if batt is not None:
            body += f'battery={batt:.1f}%\n'

        self.query_one('#status', Static).update(body)
        self.query_one('#log', Static).update('\n'.join(self._log))

    def action_takeoff(self) -> None:
        self._client.call_takeoff()

    def action_kill(self) -> None:
        self._client.call_kill()

    def action_test(self) -> None:
        self._client.call_test()

    def action_land(self) -> None:
        self._client.call_land()

    def action_nudge_fwd(self) -> None:
        self._client.call_nudge(self._client.step, 0.0, 0.0)

    def action_nudge_back(self) -> None:
        self._client.call_nudge(-self._client.step, 0.0, 0.0)

    def action_nudge_left(self) -> None:
        self._client.call_nudge(0.0, self._client.step, 0.0)

    def action_nudge_right(self) -> None:
        self._client.call_nudge(0.0, -self._client.step, 0.0)

    def action_nudge_up(self) -> None:
        self._client.call_nudge(0.0, 0.0, self._client.step)

    def action_nudge_down(self) -> None:
        self._client.call_nudge(0.0, 0.0, -self._client.step)

    def action_yaw_left(self) -> None:
        self._client.call_nudge(0.0, 0.0, 0.0, dyaw_deg=self._client.yaw_step)

    def action_yaw_right(self) -> None:
        self._client.call_nudge(0.0, 0.0, 0.0, dyaw_deg=-self._client.yaw_step)

    def action_goto(self) -> None:
        seed = None
        (odom, _batt) = self._client.snapshot()
        if odom is not None:
            p = odom.pose.pose.position
            seed = (p.x, p.y, p.z)

        def _on_close(result: tuple[float, float, float, float] | None) -> None:
            if result is not None:
                x, y, z, yaw = result
                self._client.call_goto_abs(x, y, z, yaw)

        self.push_screen(GoToScreen(seed), _on_close)

    def action_spiral(self) -> None:
        def _on_close(result: dict | None) -> None:
            if result is not None:
                self._client.call_spiral(**result)

        self.push_screen(SpiralScreen(self._client.spiral_duration), _on_close)


def main(args=None) -> None:
    rclpy.init(args=args)
    client = BridgeClient()
    executor = MultiThreadedExecutor()
    executor.add_node(client)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    app = CrazyBridgeTUI(client)
    try:
        app.run()
    finally:
        executor.shutdown()
        client.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
