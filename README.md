# Crazybridge

ROS 2 bridge to fly the Bitcraze Crazyflie 2.X with OptiTrack motion capture.

This package connects to the Crazyflie via Crazyradio, fusing external OptiTrack tracking data with the onboard state estimator. It exposes ROS 2 services for autonomous flight and publishes real-time odometry.

This package runs inside the [crazyflie-optitrack](https://github.com/covenant-tec/crazyflie-optitrack) workspace.

## Tracking Modes

The bridge receives motion capture data from OptiTrack and forwards coordinates to the Crazyflie:

* `rigid_body`: Uses 6-DoF position and orientation pose from `optitrack/rigid_body`.
* `marker`: Default mode. Uses 3-DoF position coordinates from `optitrack/marker`.
* `auto`: Automatically selects between rigid body and single marker based on the incoming OptiTrack stream.

## Configuration Files

Controller gains and vehicle mass are configured through the following files:

* `config/pid.conf`: Tuned for single-marker flight with vehicle mass set to 0.029 kg.
* `config/pid_rigid_body.conf`: Tuned for rigid-body flight with vehicle mass set to 0.037 kg.

Each file contains 25 numeric values defining translational PID gains, translational homogeneous terms, rotational PID gains, rotational homogeneous terms, and vehicle mass.

## Running

### Flight with OptiTrack

```bash
ros2 launch crazybridge crazyoptitrack.launch.py
```

If run without arguments, the launch file executes with default parameters:
* Connects to radio address `radio://0/80/2M/E7E7E7E7E7`
* Runs in `marker` mode
* Loads `config/pid.conf`
* Opens a Rerun visualization window

To override the default address or tracking mode:

```bash
ros2 launch crazybridge crazyoptitrack.launch.py tracking_mode:=rigid_body uri:=<your-crazyflie-uri>
```

### Automated Benchmark Test

Flies an automated test sequence consisting of takeoff, position hold, circular trajectory tracking, and landing, saving trajectory data and performance metrics to the specified output folder:

```bash
ros2 launch crazybridge controller_test.launch.py output_dir:=/tmp/ctrl_test
```

### Terminal Dashboard

Opens an interactive terminal interface to trigger takeoff, landing, or emergency motor shutdown:

```bash
ros2 run crazybridge crazybridge_tui
```

## Credits

The core software architecture, ROS 2 bridge implementation, controller testing framework, and OptiTrack motion capture integration were designed and developed by [Kevin Martinez](https://github.com/Fairbrook) as part of doctoral research.
