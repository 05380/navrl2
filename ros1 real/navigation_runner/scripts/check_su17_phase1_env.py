#!/usr/bin/env python3

"""Offline dependency and checkpoint check for the SU17 phase-1 node."""

import os
import sys

import rospkg


def _version(module):
    return str(getattr(module, "__version__", "unknown"))


def main():
    failures = []
    print("python={}".format(sys.version.split()[0]))
    if sys.version_info < (3, 8):
        failures.append("Python 3.8 or newer is required")

    try:
        import numpy

        print("numpy={}".format(_version(numpy)))
    except Exception as exc:
        failures.append("numpy import failed: {}".format(exc))

    try:
        import torch

        print("torch={} cuda={}".format(_version(torch), torch.cuda.is_available()))
    except Exception as exc:
        torch = None
        failures.append("torch import failed: {}".format(exc))

    try:
        import tensordict

        print("tensordict={}".format(_version(tensordict)))
    except Exception as exc:
        failures.append("tensordict import failed: {}".format(exc))

    try:
        import torchrl

        print("torchrl={}".format(_version(torchrl)))
    except Exception as exc:
        failures.append("torchrl import failed: {}".format(exc))

    for module_name in ("omegaconf", "einops"):
        try:
            module = __import__(module_name)
            print("{}={}".format(module_name, _version(module)))
        except Exception as exc:
            failures.append("{} import failed: {}".format(module_name, exc))

    for ros_import in (
        ("mavros_msgs.msg", "PositionTarget"),
        ("prometheus_msgs.msg", "UAVCommand"),
        ("prometheus_msgs.msg", "UAVControlState"),
        ("map_manager.srv", "RayCast"),
    ):
        module_name, symbol = ros_import
        try:
            module = __import__(module_name, fromlist=[symbol])
            getattr(module, symbol)
            print("ros={}::{} OK".format(module_name, symbol))
        except Exception as exc:
            failures.append(
                "ROS import {}::{} failed: {}".format(module_name, symbol, exc)
            )

    package_scripts = os.path.join(
        rospkg.RosPack().get_path("navigation_runner"), "scripts"
    )
    checkpoint = os.path.join(package_scripts, "ckpts", "navrl_checkpoint.pt")
    if not os.path.isfile(checkpoint):
        failures.append("checkpoint not found: {}".format(checkpoint))
    elif torch is not None:
        try:
            state = torch.load(checkpoint, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if not isinstance(state, dict):
                raise TypeError("checkpoint is not a state dictionary")
            keys = tuple(state.keys())
            required_prefixes = ("feature_extractor.", "actor.", "critic.")
            missing = [p for p in required_prefixes if not any(k.startswith(p) for k in keys)]
            if missing:
                raise ValueError("missing parameter groups: {}".format(missing))
            print("checkpoint={} tensors={} OK".format(checkpoint, len(keys)))
        except Exception as exc:
            failures.append("checkpoint load failed: {}".format(exc))

    if failures:
        print("\nSU17 phase-1 environment: FAILED")
        for failure in failures:
            print("- {}".format(failure))
        return 1

    print("\nSU17 phase-1 environment: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
