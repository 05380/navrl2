#!/usr/bin/env bash

# One-shot, 0.10 m/s, 3 s maximum tethered NavRL Move launcher.

set -euo pipefail

NAVRL_WORKSPACE="${NAVRL_WORKSPACE:-/home/amov/navrl_ws}"
NAVRL_VENV="${NAVRL_VENV:-/home/amov/navrl_venv}"
PROMETHEUS_SETUP="${PROMETHEUS_SETUP:-/home/amov/su17_experiment/devel/setup.bash}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/noetic/setup.bash}"
COMMAND_TOPIC="/uav1/prometheus/command"
DESIRED_TOPIC="/uav1/navrl/desired_setpoint"

if [[ $# -ne 0 ]]; then
  echo "ERROR: limited-Move launcher accepts no overrides" >&2
  exit 2
fi
if [[ "${NAVRL_LIMITED_MOVE_ACK:-}" != "TETHER_SECURE_ONE_010MPS_MOVE" ]]; then
  cat >&2 <<'EOF'
ERROR: limited-Move interlock is not acknowledged.
Prepare the proven tether, clear area, calm air, and two operators. Keep the
UAV disarmed, then run exactly:

  NAVRL_LIMITED_MOVE_ACK=TETHER_SECURE_ONE_010MPS_MOVE \
  bash /home/amov/navrl2/ros1_real/navigation_runner/tools/run_su17_limited_move_test.sh
EOF
  exit 2
fi

for setup_file in \
  "${ROS_SETUP}" \
  "${PROMETHEUS_SETUP}" \
  "${NAVRL_VENV}/bin/activate" \
  "${NAVRL_WORKSPACE}/devel/setup.bash"; do
  [[ -f "${setup_file}" ]] || { echo "ERROR: missing ${setup_file}" >&2; exit 1; }
  # shellcheck disable=SC1090
  source "${setup_file}"
done

if ! python -c \
  'import torch; import tensordict; import tensordict._tensordict; import torchrl' \
  >/dev/null 2>&1; then
  echo "ERROR: NavRL Python extensions are unavailable" >&2
  exit 4
fi
if [[ "$(rosparam get use_sim_time 2>/dev/null || true)" == "true" ]]; then
  echo "ERROR: use_sim_time=true" >&2
  exit 1
fi

active_nodes="$(rosnode list 2>/dev/null | grep -E \
  '^/(navrl_su17_occupancy_map|navigation_su17_phase1|navrl_su17_bridge)$' || true)"
if [[ -n "${active_nodes}" ]]; then
  echo "ERROR: stop the existing NavRL instance first:" >&2
  echo "${active_nodes}" >&2
  exit 3
fi

for topic in \
  /uav1/cloud_mid360_body \
  /uav1/Odometry \
  /uav1/mavros/local_position/odom \
  /uav1/prometheus/state \
  /uav1/prometheus/control_state; do
  timeout 4 rostopic echo -n 1 "${topic}" >/dev/null 2>&1 || {
    echo "ERROR: no live message on ${topic}" >&2; exit 5;
  }
done

state_message="$(timeout 4 rostopic echo -n 1 /uav1/prometheus/state)"
grep -Eqi '^connected:[[:space:]]+true$' <<<"${state_message}" || { echo "ERROR: connected is not true" >&2; exit 6; }
grep -Eqi '^armed:[[:space:]]+false$' <<<"${state_message}" || { echo "ERROR: UAV is not verified disarmed" >&2; exit 6; }
grep -Eqi '^location_source:[[:space:]]+10$' <<<"${state_message}" || { echo "ERROR: location source is not MID360" >&2; exit 6; }
grep -Eqi '^odom_valid:[[:space:]]+true$' <<<"${state_message}" || { echo "ERROR: odom_valid is not true" >&2; exit 6; }

control_message="$(timeout 4 rostopic echo -n 1 /uav1/prometheus/control_state)"
grep -Eqi '^control_state:[[:space:]]+0$' <<<"${control_message}" || { echo "ERROR: initial control state is not INIT" >&2; exit 6; }
grep -Eqi '^pos_controller:[[:space:]]+0$' <<<"${control_message}" || { echo "ERROR: pos_controller is not PX4_ORIGIN" >&2; exit 6; }
grep -Eqi '^failsafe:[[:space:]]+false$' <<<"${control_message}" || { echo "ERROR: failsafe is not false" >&2; exit 6; }

topic_publishers() {
  local info
  info="$(rostopic info "$1" 2>/dev/null || true)"
  awk '/^Publishers:/ {p=1; next} /^Subscribers:/ {p=0} p && /^[[:space:]]*\*/ {print}' <<<"${info}"
}
command_info="$(rostopic info "${COMMAND_TOPIC}" 2>/dev/null || true)"
grep -q '/uav_control_main_1' <<<"${command_info}" || { echo "ERROR: Prometheus command subscriber missing" >&2; exit 7; }
[[ -z "$(topic_publishers "${COMMAND_TOPIC}")" ]] || { echo "ERROR: command publisher already exists" >&2; exit 7; }
[[ -z "$(topic_publishers "${DESIRED_TOPIC}")" ]] || { echo "ERROR: desired-setpoint publisher already exists" >&2; exit 7; }

cat <<'EOF'
LIMITED MOVE: all startup interlocks passed.
Limits are fixed: one private 0.55 m goal, 0.10 m/s XY, 3.0 s Move latch,
0.75 m home fence, Z position hold. RViz goals are not accepted.
EOF

exec roslaunch navigation_runner su17_limited_move_test.launch
