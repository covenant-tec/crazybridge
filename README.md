# Crazybridge

ROS 2 bridge to fly the Bitcraze Crazyflie 2.X with OptiTrack motion capture.

This package connects to the Crazyflie via Crazyradio, fusing external OptiTrack tracking data with the onboard state estimator. It exposes ROS 2 services for autonomous flight and publishes real-time odometry.

This package runs inside the [crazyflie-optitrack](https://github.com/covenant-tec/crazyflie-optitrack) workspace.

## Tracking Modes

The bridge receives motion capture data from OptiTrack and forwards coordinates to the Crazyflie:

* `rigid_body`: Uses 6-DoF position and orientation pose from `optitrack/rigid_body`.
* `marker`: Uses 3-DoF position coordinates from `optitrack/marker`.
* `auto`: Default mode. Automatically selects between rigid body and single marker based on the incoming OptiTrack stream.

## State Estimator & Adaptation Modes

The bridge accepts `estimator_mode` and `adapt_mode` launch parameters to switch between the stock Bitcraze Kalman filter and the Out-Of-Tree 13-state Adaptive EKF (`static`, `mdp`, `smdp`) at startup **or live via `ros2 param set`** without reflashing:

* `estimator_mode`:
  * `stock` *(default)*: Selects the stock Bitcraze Extended Kalman Filter (`ekfOot.useStock=1` on Out-Of-Tree firmware or `stabilizer.estimator=2` on standard firmware).
  * `oot`: Activates the custom Out-Of-Tree 13-state EKF (`ekfOot.useStock=0`), uploads the Bellman decision matrices from `config/adapt_policies.conf`, streams diagnostics on `/crazybridge/ekf_diag` and `/crazybridge/ekf_adapt`, and appends the diagnostic columns to `flight_telemetry.csv`.
  * `auto`: Uses the Out-Of-Tree EKF when `ekfOot` is present in the firmware TOC and falls back to the stock Kalman filter otherwise.
* `adapt_mode` *(used when `estimator_mode:=oot`)*:
  * `static` *(default)*: Fixed nominal covariances $Q_{\text{base}}$ and $R_{\text{base}}$ (no adaptation, `ekfOot.mode=0`).
  * `mdp`: Time-driven 2D Markov policy evaluated every 50 Hz update (`ekfOot.mode=1`).
  * `smdp`: Event-driven Semi-Markov policy evaluated only on discrete state transitions (`ekfOot.mode=2`).

## Configuration Files

Controller gains, vehicle mass, and EKF adaptation policies are configured through the following files:

* `config/pid.conf`: Tuned for single-marker flight with vehicle mass set to 0.029 kg.
* `config/pid_rigid_body.conf`: Tuned for rigid-body flight with vehicle mass set to 0.042 kg.
* `config/adapt_policies.conf`: Contains the two $3\times 3$ optimal Bellman decision matrices (`MDP 2D` and `SMDP`) pushed in RAM to `ekfPolicy.*` on connect. If you re-train policies in the notebook, update this file directly — no firmware recompilation is required.

Each file contains 28 numeric values defining translational PID gains, translational homogeneous terms, rotational PID gains, rotational homogeneous terms, vehicle mass, and thrust/torque saturation limits.

## Installation & Build

Activate the Python virtual environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

To compile the package within the ROS 2 workspace, navigate to the `src` directory and execute the build using Python's colcon module to ensure the active virtual environment is used:

```bash
cd ..
python -m colcon build --symlink-install --packages-select crazybridge crazybridge_interfaces
```

## Running

### Flight with OptiTrack

```bash
ros2 launch crazybridge crazyoptitrack.launch.py
```

If run without arguments, the launch file executes with default parameters:
* Connects to radio address `radio://0/80/2M/E7E7E7E7E7`
* Runs tracking in `auto` mode, estimator in `stock` mode, and adaptation in `static` mode
* Loads `config/pid.conf` or `config/pid_rigid_body.conf` automatically
* Opens a Rerun visualization window

To override the default address or estimator/adaptation mode:

```bash
ros2 launch crazybridge crazyoptitrack.launch.py tracking_mode:=rigid_body estimator_mode:=oot adapt_mode:=smdp
```

### Automated Benchmark Test

Flies an automated test sequence consisting of takeoff, position hold, circular trajectory tracking, and landing, saving trajectory data and performance metrics to the specified output folder:

```bash
# 1. Default: Stock Bitcraze Kalman
ros2 launch crazybridge controller_test.launch.py config_name:=stock output_dir:=/tmp/ctrl_test

# 2. Custom 13-state EKF without adaptation (default adapt_mode:=static)
ros2 launch crazybridge controller_test.launch.py config_name:=ekf_static estimator_mode:=oot output_dir:=/tmp/ctrl_test

# 3. Custom 13-state EKF + Markov 2D (50 Hz)
ros2 launch crazybridge controller_test.launch.py config_name:=ekf_mdp estimator_mode:=oot adapt_mode:=mdp output_dir:=/tmp/ctrl_test

# 4. Custom 13-state EKF + Semi-Markov (event-driven)
ros2 launch crazybridge controller_test.launch.py config_name:=ekf_smdp estimator_mode:=oot adapt_mode:=smdp output_dir:=/tmp/ctrl_test
```

### Terminal Dashboard

Opens an interactive terminal interface to trigger takeoff, landing, or emergency motor shutdown:

```bash
ros2 run crazybridge crazybridge_tui
```

## Credits

The core software architecture, ROS 2 bridge implementation, controller testing framework, and OptiTrack motion capture integration were designed and developed by [Kevin Martinez](https://github.com/Fairbrook) as part of doctoral research.
