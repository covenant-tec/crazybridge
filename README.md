# crazybridge

ROS 2 (Humble) bridge for Crazyflie drones. Requires the sibling package `crazybridge_interfaces`
in the same workspace.

## Setup

```bash
src/crazybridge/scripts/setup_rosdep.sh                    # once per machine, from the workspace root
rosdep install --from-paths src --ignore-src -r -y
colcon build
```

`cflib` and `rerun-sdk` are only published on PyPI, and apt's `textual` is far too old, so
`rosdep/local.yaml` maps `python3-cflib`, `python3-textual` and `python3-rerun-sdk` to pip packages; `scripts/setup_rosdep.sh`
registers that file with rosdep on your machine.
