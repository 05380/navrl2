#!/usr/bin/env bash

# Start phase 1 in computation-only shadow mode. This script deliberately
# fixes output_enabled=false and refuses attempts to override it.

set -eo pipefail

NAVRL_WORKSPACE="${NAVRL_WORKSPACE:-/home/amov/navrl_ws}"
NAVRL_VENV="${NAVRL_VENV:-/home/amov/navrl_venv}"
PROMETHEUS_SETUP="${PROMETHEUS_SETUP:-/home/amov/su17_experiment/devel/setup.bash}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/noetic/setup.bash}"

for launch_arg in "$@"; do
  case "${launch_arg}" in
    output_enabled:=*)
      echo "ERROR: this launcher cannot enable control output" >&2
      exit 2
      ;;
  esac
done

for setup_file in \
  "${ROS_SETUP}" \
  "${PROMETHEUS_SETUP}" \
  "${NAVRL_VENV}/bin/activate" \
  "${NAVRL_WORKSPACE}/devel/setup.bash"; do
  if [[ ! -f "${setup_file}" ]]; then
    echo "ERROR: missing ${setup_file}; run setup_su17_phase1_onboard.sh first" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${setup_file}"
done

# An editable TensorDict install contains a small C++ extension compiled for
# the onboard Python/PyTorch ABI.  Copying or replacing the source tree does
# not provide that Linux binary, so fail before starting any ROS nodes and
# print the exact offline repair command if it is absent or stale.
if ! python -c \
  'import torch; import tensordict; import tensordict._tensordict; import torchrl' \
  >/dev/null 2>&1; then
  cat >&2 <<'EOF'
ERROR: the NavRL Python extensions are missing or incompatible.
Do not fly. Repair the local editable packages with:

  source /home/amov/navrl_venv/bin/activate
  export MAX_JOBS=2
  python -m pip install --no-build-isolation --no-deps --force-reinstall -e /home/amov/navrl2/isaac-training/third_party/tensordict

Then source the ROS overlays and run check_su17_phase1_env.py again.
EOF
  exit 4
fi

if [[ "$(rosparam get use_sim_time 2>/dev/null || true)" == "true" ]]; then
  echo "ERROR: use_sim_time=true; restore it with: rosparam set use_sim_time false" >&2
  exit 1
fi

active_navrl_nodes="$(
  rosnode list 2>/dev/null | grep -E \
    '^/(navrl_su17_occupancy_map|navigation_su17_phase1|navrl_su17_bridge)$' || true
)"
if [[ -n "${active_navrl_nodes}" ]]; then
  echo "ERROR: a NavRL phase-1 instance is already running:" >&2
  echo "${active_navrl_nodes}" >&2
  echo "Stop its roslaunch terminal with Ctrl-C before starting another instance." >&2
  exit 3
fi

exec roslaunch navigation_runner su17_phase1.launch \
  output_enabled:=false \
  enable_odom_consistency_check:=true \
  map_visualization:=false \
  "$@"
