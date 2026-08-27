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

if [[ "$(rosparam get use_sim_time 2>/dev/null || true)" == "true" ]]; then
  echo "ERROR: use_sim_time=true; restore it with: rosparam set use_sim_time false" >&2
  exit 1
fi

exec roslaunch navigation_runner su17_phase1.launch \
  output_enabled:=false \
  enable_odom_consistency_check:=true \
  map_visualization:=true \
  "$@"
