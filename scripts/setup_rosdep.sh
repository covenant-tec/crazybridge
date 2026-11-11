#!/usr/bin/env bash
# Register this repo's local rosdep rules (rosdep/local.yaml) so that
# `rosdep install --from-paths src --ignore-src -r -y` can resolve python3-cflib.
# Idempotent: safe to re-run (e.g. after moving the repo).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # the crazybridge package dir
YAML="$REPO_ROOT/rosdep/local.yaml"
LIST=/etc/ros/rosdep/sources.list.d/10-crazyflies.list

[ -f "$YAML" ] || { echo "missing $YAML" >&2; exit 1; }
command -v rosdep >/dev/null || { echo "rosdep not found; install python3-rosdep first" >&2; exit 1; }

# rosdep needs its base config before sources can be added.
if [ ! -d /etc/ros/rosdep/sources.list.d ]; then
  sudo rosdep init
fi

echo "yaml file://$YAML" | sudo tee "$LIST" >/dev/null
echo "Registered $YAML in $LIST"

rosdep update
echo "Done. From your workspace root run: rosdep install --from-paths src --ignore-src -r -y"
