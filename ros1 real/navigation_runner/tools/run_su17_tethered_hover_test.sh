#!/usr/bin/env bash

# First powered-flight gate: low-altitude tethered hover with no policy/map.
# The only permitted active command is native Current_Pos_Hover.  This script
# is not a navigation-flight launcher and intentionally accepts no overrides.

set -euo pipefail

NAVRL_WORKSPACE="${NAVRL_WORKSPACE:-/home/amov/navrl_ws}"
NAVRL_VENV="${NAVRL_VENV:-/home/amov/navrl_venv}"
PROMETHEUS_SETUP="${PROMETHEUS_SETUP:-/home/amov/su17_experiment/devel/setup.bash}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/noetic/setup.bash}"
COMMAND_TOPIC="/uav1/prometheus/command"
DESIRED_TOPIC="/uav1/navrl/desired_setpoint"

if [[ $# -ne 0 ]]; then
  echo "ERROR: the tethered-hover launcher accepts no roslaunch overrides" >&2
  exit 2
fi

if [[ "${NAVRL_TETHERED_HOVER_ACK:-}" != "TETHER_SECURE_CLEAR_AREA_TWO_OPERATORS" ]]; then
  cat >&2 <<'EOF'
ERROR: tethered-hover safety interlock is not acknowledged.
Prepare a secure tether/protective facility, a clear 3-5 m area, calm air,
and two operators. Keep the UAV disarmed, then run exactly:

  NAVRL_TETHERED_HOVER_ACK=TETHER_SECURE_CLEAR_AREA_TWO_OPERATORS \
  bash /home/amov/navrl2/ros1_real/navigation_runner/tools/run_su17_tethered_hover_test.sh

This launcher is only for a no-goal hover handover at 0.9-1.0 m.
EOF
  exit 2
fi

for setup_file in \
  "${ROS_SETUP}" \
  "${PROMETHEUS_SETUP}" \
  "${NAVRL_VENV}/bin/activate" \
  "${NAVRL_WORKSPACE}/devel/setup.bash"; do
  if [[ ! -f "${setup_file}" ]]; then
    echo "ERROR: missing ${setup_file}" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${setup_file}"
done

if [[ "$(rosparam get use_sim_time 2>/dev/null || true)" == "true" ]]; then
  echo "ERROR: use_sim_time=true; run: rosparam set use_sim_time false" >&2
  exit 1
fi

active_navrl_nodes="$(
  rosnode list 2>/dev/null | grep -E \
    '^/(navrl_su17_occupancy_map|navigation_su17_phase1|navrl_su17_bridge)$' || true
)"
if [[ -n "${active_navrl_nodes}" ]]; then
  echo "ERROR: stop the existing NavRL instance first:" >&2
  echo "${active_navrl_nodes}" >&2
  exit 3
fi

required_topics=(
  /uav1/Odometry
  /uav1/mavros/local_position/odom
  /uav1/prometheus/state
  /uav1/prometheus/control_state
)
for topic in "${required_topics[@]}"; do
  if ! timeout 4 rostopic echo -n 1 "${topic}" >/dev/null 2>&1; then
    echo "ERROR: no live message on ${topic}" >&2
    exit 5
  fi
done

state_message="$(timeout 4 rostopic echo -n 1 /uav1/prometheus/state)"
if ! grep -Eqi '^connected:[[:space:]]+true$' <<<"${state_message}"; then
  echo "ERROR: Prometheus connected is not true" >&2
  exit 6
fi
if ! grep -Eqi '^armed:[[:space:]]+false$' <<<"${state_message}"; then
  echo "ERROR: UAV armed state cannot be verified as false" >&2
  exit 6
fi
if ! grep -Eqi '^location_source:[[:space:]]+10$' <<<"${state_message}"; then
  echo "ERROR: location_source is not MID360 (10)" >&2
  exit 6
fi
if ! grep -Eqi '^odom_valid:[[:space:]]+true$' <<<"${state_message}"; then
  echo "ERROR: Prometheus odom_valid is not true" >&2
  exit 6
fi

control_message="$(timeout 4 rostopic echo -n 1 /uav1/prometheus/control_state)"
if ! grep -Eqi '^control_state:[[:space:]]+0$' <<<"${control_message}"; then
  echo "ERROR: initial Prometheus control_state is not INIT (0)" >&2
  exit 6
fi
if ! grep -Eqi '^pos_controller:[[:space:]]+0$' <<<"${control_message}"; then
  echo "ERROR: Prometheus pos_controller is not PX4_ORIGIN (0)" >&2
  exit 6
fi
if ! grep -Eqi '^failsafe:[[:space:]]+false$' <<<"${control_message}"; then
  echo "ERROR: Prometheus failsafe is not false" >&2
  exit 6
fi

topic_publishers() {
  local topic="$1"
  local topic_info
  topic_info="$(rostopic info "${topic}" 2>/dev/null || true)"
  awk '
    /^Publishers:/ { in_publishers=1; next }
    /^Subscribers:/ { in_publishers=0 }
    in_publishers && /^[[:space:]]*\*/ { print }
  ' <<<"${topic_info}"
}

command_info="$(rostopic info "${COMMAND_TOPIC}" 2>/dev/null || true)"
if ! grep -q '/uav_control_main_1' <<<"${command_info}"; then
  echo "ERROR: ${COMMAND_TOPIC} is not subscribed by /uav_control_main_1" >&2
  exit 7
fi
existing_command_publishers="$(topic_publishers "${COMMAND_TOPIC}")"
if [[ -n "${existing_command_publishers}" ]]; then
  echo "ERROR: ${COMMAND_TOPIC} already has a publisher:" >&2
  echo "${existing_command_publishers}" >&2
  exit 7
fi

existing_desired_publishers="$(topic_publishers "${DESIRED_TOPIC}")"
if [[ -n "${existing_desired_publishers}" ]]; then
  echo "ERROR: ${DESIRED_TOPIC} has a publisher; this must remain a no-goal test:" >&2
  echo "${existing_desired_publishers}" >&2
  exit 7
fi

cat <<'EOF'
TETHERED HOVER ONLY: all startup interlocks passed.
No map or policy node will be started. Do not send a navigation goal.
Take off with the proven SU17 manual/RC procedure, stabilize at 0.9-1.0 m,
then test COMMAND_CONTROL for only 3-5 s before returning SWB to middle.
EOF

exec roslaunch navigation_runner su17_tethered_hover_test.launch
