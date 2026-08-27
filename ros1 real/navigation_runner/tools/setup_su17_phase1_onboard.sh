#!/usr/bin/env bash

# Build the SU17 NavRL phase-1 overlay without changing the working
# Prometheus installation. Run this after extracting/copying the deployment
# files and installing the apt prerequisites listed in the deployment guide.

set -eo pipefail

NAVRL_ROOT="${NAVRL_ROOT:-/home/amov/navrl2}"
NAVRL_WORKSPACE="${NAVRL_WORKSPACE:-/home/amov/navrl_ws}"
NAVRL_VENV="${NAVRL_VENV:-/home/amov/navrl_venv}"
PROMETHEUS_SETUP="${PROMETHEUS_SETUP:-/home/amov/su17_experiment/devel/setup.bash}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/noetic/setup.bash}"

ROS1_REAL="${NAVRL_ROOT}/ros1_real"
THIRD_PARTY="${NAVRL_ROOT}/isaac-training/third_party"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

for required_file in \
  "${ROS_SETUP}" \
  "${PROMETHEUS_SETUP}" \
  "${ROS1_REAL}/navigation_runner/package.xml" \
  "${ROS1_REAL}/map_manager/package.xml" \
  "${ROS1_REAL}/onboard_detector/package.xml" \
  "${THIRD_PARTY}/tensordict/setup.py" \
  "${THIRD_PARTY}/rl/setup.py"; do
  [[ -f "${required_file}" ]] || fail "missing ${required_file}"
done

for required_command in python3 catkin_make g++ ninja; do
  command -v "${required_command}" >/dev/null 2>&1 || \
    fail "${required_command} is missing; install the apt prerequisites first"
done

[[ -f /usr/include/python3.8/Python.h ]] || \
  fail "python3-dev is missing (/usr/include/python3.8/Python.h not found)"

# shellcheck disable=SC1090
source "${ROS_SETUP}"
# shellcheck disable=SC1090
source "${PROMETHEUS_SETUP}"
rospack find prometheus_msgs >/dev/null || \
  fail "prometheus_msgs is not visible after sourcing ${PROMETHEUS_SETUP}"
rospack find vision_msgs >/dev/null || \
  fail "vision_msgs is missing; install it with: sudo apt-get install -y ros-noetic-vision-msgs"

mkdir -p "${NAVRL_WORKSPACE}/src"

ensure_package_link() {
  local package_name="$1"
  local source_path="${ROS1_REAL}/${package_name}"
  local target_path="${NAVRL_WORKSPACE}/src/${package_name}"

  if [[ -L "${target_path}" ]]; then
    [[ "$(readlink -f "${target_path}")" == "$(readlink -f "${source_path}")" ]] || \
      fail "${target_path} already links to a different package"
    return
  fi
  [[ ! -e "${target_path}" ]] || \
    fail "${target_path} already exists and was not overwritten"
  ln -s "${source_path}" "${target_path}"
}

ensure_package_link map_manager
ensure_package_link onboard_detector
ensure_package_link navigation_runner

if [[ ! -d "${NAVRL_VENV}" ]]; then
  python3 -m venv --system-site-packages "${NAVRL_VENV}"
fi

# shellcheck disable=SC1090
source "${NAVRL_VENV}/bin/activate"
# Ubuntu 20.04 exposes an old system importlib_metadata inside the
# --system-site-packages venv.  Modern setuptools imports EntryPoints while
# preparing legacy setup.py packages (antlr4 is pulled in by Hydra), so shadow
# the system copy with the last Python-3.8-compatible release before resolving
# the remaining requirements.
python -m pip install --upgrade \
  "pip<25" setuptools wheel ninja "importlib-metadata==8.5.0"

if python -c "import torch" >/dev/null 2>&1; then
  TORCH_VERSION="$(python -c "import torch; print(torch.__version__)")"
  [[ "${TORCH_VERSION}" == 2.0.1* ]] || \
    fail "existing torch=${TORCH_VERSION}; phase 1 requires torch 2.0.1 CPU"
else
  python -m pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.0.1
fi

python -m pip install -r \
  "${ROS1_REAL}/navigation_runner/requirements_su17_phase1.txt"
python -m pip install --no-build-isolation --no-deps -e \
  "${THIRD_PARTY}/tensordict"
python -m pip install --no-build-isolation --no-deps -e \
  "${THIRD_PARTY}/rl"

cd "${NAVRL_WORKSPACE}"
catkin_make -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE="${NAVRL_VENV}/bin/python3"

# shellcheck disable=SC1090
source "${NAVRL_WORKSPACE}/devel/setup.bash"
rosrun navigation_runner check_su17_phase1_env.py

echo
echo "SU17 phase-1 overlay is ready."
echo "Run only the shadow-mode launcher first:"
echo "  ${ROS1_REAL}/navigation_runner/tools/run_su17_phase1_shadow.sh"
