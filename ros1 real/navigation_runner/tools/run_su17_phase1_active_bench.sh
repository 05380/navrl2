#!/usr/bin/env bash

# Propellers-off validation of the NavRL -> Prometheus active command path.
# This is deliberately not a flight launcher.  It refuses to start unless the
# operator explicitly confirms that every propeller has been removed, the UAV
# is disarmed, the required live inputs are healthy, and no other node already
# publishes Prometheus UAVCommand.

set -euo pipefail

NAVRL_WORKSPACE="${NAVRL_WORKSPACE:-/home/amov/navrl_ws}"
NAVRL_VENV="${NAVRL_VENV:-/home/amov/navrl_venv}"
PROMETHEUS_SETUP="${PROMETHEUS_SETUP:-/home/amov/su17_experiment/devel/setup.bash}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/noetic/setup.bash}"
COMMAND_TOPIC="/uav1/prometheus/command"

if [[ $# -ne 0 ]]; then
  echo "ERROR: the active bench launcher accepts no roslaunch overrides" >&2
  exit 2
fi

if [[ "${NAVRL_ACTIVE_BENCH_ACK:-}" != "PROPELLERS_REMOVED" ]]; then
  cat >&2 <<'EOF'
ERROR: active bench interlock is not acknowledged.
Remove every propeller, secure the airframe, keep the UAV disarmed, then run:

  NAVRL_ACTIVE_BENCH_ACK=PROPELLERS_REMOVED \
  bash /home/amov/navrl2/ros1_real/navigation_runner/tools/run_su17_phase1_active_bench.sh

This launcher must not be used for flight.
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

if ! python -c \
  'import torch; import tensordict; import tensordict._tensordict; import torchrl' \
  >/dev/null 2>&1; then
  echo "ERROR: NavRL Python extensions are unavailable; do not continue" >&2
  exit 4
fi

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
  /uav1/cloud_mid360_body
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
if ! grep -Eqi '^armed:[[:space:]]+false$' <<<"${state_message}"; then
  echo "ERROR: UAV is armed or its armed state cannot be verified as false" >&2
  exit 6
fi
if ! grep -Eqi '^connected:[[:space:]]+true$' <<<"${state_message}"; then
  echo "ERROR: Prometheus reports connected=false" >&2
  exit 6
fi
if ! grep -Eqi '^odom_valid:[[:space:]]+true$' <<<"${state_message}"; then
  echo "ERROR: Prometheus reports odom_valid=false" >&2
  exit 6
fi

command_info="$(rostopic info "${COMMAND_TOPIC}" 2>/dev/null || true)"
if ! grep -q '/uav_control_main_1' <<<"${command_info}"; then
  echo "ERROR: ${COMMAND_TOPIC} is not subscribed by /uav_control_main_1" >&2
  exit 7
fi
existing_publishers="$(
  awk '
    /^Publishers:/ { in_publishers=1; next }
    /^Subscribers:/ { in_publishers=0 }
    in_publishers && /^[[:space:]]*\*/ { print }
  ' <<<"${command_info}"
)"
if [[ -n "${existing_publishers}" ]]; then
  echo "ERROR: ${COMMAND_TOPIC} already has a publisher; stop it first:" >&2
  echo "${existing_publishers}" >&2
  exit 7
fi

cat <<'EOF'
ACTIVE BENCH ONLY: all startup interlocks passed.
Do not send a NavRL goal. Keep all propellers removed.
Expected RC check: SWB middle -> RC_POS_CONTROL, SWB third -> COMMAND_CONTROL,
then immediately return SWB to middle to verify manual takeover.
EOF

exec roslaunch navigation_runner su17_phase1.launch \
  output_enabled:=true \
  enable_odom_consistency_check:=true \
  map_visualization:=false \
  max_xy_speed:=0.15 \
  max_z_speed:=0.15 \
  max_xy_from_home:=1.0 \
  emergency_stop_distance:=0.65
