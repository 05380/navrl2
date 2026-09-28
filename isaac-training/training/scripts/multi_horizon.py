import math

import torch
from tensordict.tensordict import TensorDict


def horizon_seconds_to_steps(seconds, dt):
    seconds = float(seconds)
    dt = float(dt)
    if seconds <= 0.0 or dt <= 0.0:
        raise ValueError("Outcome horizons and the simulator interval must be positive.")
    return max(1, int(math.floor(seconds / dt + 0.5)))


class MultiHorizonSequenceBuffer:
    """Build episode-aware future-outcome targets across PPO rollouts."""

    _OBSERVATION_KEYS = ("state", "lidar", "dynamic_obstacle")

    def __init__(
        self,
        horizons,
        dt,
        blockage_distance=1.5,
        blockage_ratio=0.8,
        max_net_speed=0.15,
        max_progress_speed=0.10,
    ):
        horizons = tuple(int(horizon) for horizon in horizons)
        if not horizons or horizons[0] < 1:
            raise ValueError("Multi-horizon target steps must be positive integers.")
        if tuple(sorted(set(horizons))) != horizons:
            raise ValueError("Multi-horizon target steps must be unique and increasing.")
        if dt <= 0.0:
            raise ValueError("The simulator interval must be positive.")

        self.horizons = horizons
        self.max_horizon = self.horizons[-1]
        self.dt = float(dt)
        self.blockage_distance = float(blockage_distance)
        self.blockage_ratio = float(blockage_ratio)
        self.max_net_speed = float(max_net_speed)
        self.max_progress_speed = float(max_progress_speed)
        self._pending = None

    @property
    def pending_count(self):
        if self._pending is None:
            return 0
        return int(self._pending["valid"].sum().item())

    def reset(self):
        self._pending = None

    def append(self, tensordict):
        """Append one [environment, time] rollout and return mature samples."""
        current = self._extract_records(tensordict)
        combined = self._concatenate(current)
        mature_mask, next_done = self._find_mature_sources(combined)
        mature_batch = self._build_mature_batch(combined, mature_mask, next_done)
        self._retain_unfinished_suffix(combined)
        return mature_batch

    def _extract_records(self, tensordict):
        if len(tensordict.batch_size) != 2:
            raise ValueError(
                "Multi-horizon buffering expects rollout batches shaped "
                "[num_envs, rollout_steps]."
            )

        records = {
            key: tensordict[("agents", "observation", key)].detach()
            for key in self._OBSERVATION_KEYS
        }
        records["action"] = tensordict[("agents", "action_normalized")].detach()
        records["collision"] = tensordict[("next", "stats", "collision")].detach().squeeze(-1)
        records["clearance"] = tensordict[("next", "stats", "front_clearance")].detach().squeeze(-1)
        records["progress"] = tensordict[("next", "stats", "goal_progress")].detach().squeeze(-1)
        records["done"] = tensordict[("next", "done")].detach().squeeze(-1).bool()

        current_state = tensordict[("info", "drone_state")].detach()
        next_state = tensordict[("next", "info", "drone_state")].detach()
        records["position_xy"] = current_state[..., 0, :2]
        records["next_position_xy"] = next_state[..., 0, :2]
        records["valid"] = torch.ones_like(records["done"], dtype=torch.bool)
        return records

    def _concatenate(self, current):
        if self._pending is None:
            return current
        if self._pending["valid"].shape[0] != current["valid"].shape[0]:
            raise RuntimeError(
                "The number of PPO environments changed while auxiliary "
                "future-outcome samples were pending."
            )
        return {
            key: torch.cat((self._pending[key], current[key]), dim=1)
            for key in current
        }

    def _find_mature_sources(self, records):
        valid = records["valid"]
        done = records["done"] & valid
        sequence_length = valid.shape[1]
        positions = torch.arange(sequence_length, device=valid.device).view(1, -1)
        positions = positions.expand_as(valid)

        no_done = torch.full_like(positions, sequence_length)
        done_positions = torch.where(done, positions, no_done)
        next_done = torch.flip(
            torch.cummin(torch.flip(done_positions, dims=[1]), dim=1).values,
            dims=[1],
        )
        remaining_valid = torch.flip(
            torch.cumsum(torch.flip(valid.long(), dims=[1]), dim=1),
            dims=[1],
        )
        mature = valid & (
            (next_done < sequence_length)
            | (remaining_valid >= self.max_horizon)
        )
        return mature, next_done

    @staticmethod
    def _range_sum(signal, environment_indices, starts, ends):
        prefix = torch.cat(
            (torch.zeros_like(signal[:, :1]), torch.cumsum(signal, dim=1)),
            dim=1,
        )
        return (
            prefix[environment_indices, ends + 1]
            - prefix[environment_indices, starts]
        )

    @staticmethod
    def _range_min(signal, environment_indices, starts, ends, horizon):
        offsets = torch.arange(horizon, device=signal.device).view(1, -1)
        indices = starts.unsqueeze(1) + offsets
        within_window = indices <= ends.unsqueeze(1)
        indices = indices.clamp_max(signal.shape[1] - 1)
        values = signal[environment_indices.unsqueeze(1), indices]
        values = torch.where(
            within_window,
            values,
            torch.full_like(values, float("inf")),
        )
        return values.amin(dim=1)

    def _build_mature_batch(self, records, mature_mask, next_done):
        environment_indices, starts = mature_mask.nonzero(as_tuple=True)
        if starts.numel() == 0:
            return None

        sequence_length = mature_mask.shape[1]
        source_next_done = next_done[environment_indices, starts]
        blocked = (records["clearance"] <= self.blockage_distance).float()

        collision_targets = []
        trap_targets = []
        clearance_targets = []
        progress_targets = []
        for horizon in self.horizons:
            nominal_ends = starts + horizon - 1
            ends = torch.minimum(nominal_ends, source_next_done)
            ends = ends.clamp_max(sequence_length - 1)
            window_steps = (ends - starts + 1).clamp_min(1)
            window_seconds = window_steps.float() * self.dt

            collision_count = self._range_sum(
                records["collision"].float(), environment_indices, starts, ends
            )
            progress = self._range_sum(
                records["progress"], environment_indices, starts, ends
            )
            blockage_count = self._range_sum(
                blocked, environment_indices, starts, ends
            )
            clearance = self._range_min(
                records["clearance"],
                environment_indices,
                starts,
                ends,
                horizon,
            )

            source_position = records["position_xy"][environment_indices, starts]
            end_position = records["next_position_xy"][environment_indices, ends]
            net_speed = (end_position - source_position).norm(dim=-1) / window_seconds
            progress_speed = progress / window_seconds
            blockage_fraction = blockage_count / window_steps.float()
            trapped = (
                (blockage_fraction >= self.blockage_ratio)
                & (net_speed <= self.max_net_speed)
                & (progress_speed <= self.max_progress_speed)
            )

            collision_targets.append((collision_count > 0.0).float())
            trap_targets.append(trapped.float())
            clearance_targets.append(clearance)
            progress_targets.append(progress)

        batch_size = starts.numel()
        device = records["state"].device
        observation = TensorDict(
            {
                key: records[key][environment_indices, starts].clone()
                for key in self._OBSERVATION_KEYS
            },
            batch_size=[batch_size],
            device=device,
        )
        agents = TensorDict(
            {
                "observation": observation,
                "action_normalized": records["action"][environment_indices, starts].clone(),
            },
            batch_size=[batch_size],
            device=device,
        )
        return TensorDict(
            {
                "agents": agents,
                "_aux_future_collision": torch.stack(collision_targets, dim=-1),
                "_aux_future_trap": torch.stack(trap_targets, dim=-1),
                "_aux_future_clearance": torch.stack(clearance_targets, dim=-1),
                "_aux_future_progress": torch.stack(progress_targets, dim=-1),
            },
            batch_size=[batch_size],
            device=device,
        )

    def _retain_unfinished_suffix(self, records):
        pending_capacity = self.max_horizon - 1
        if pending_capacity == 0:
            self._pending = None
            return

        valid = records["valid"]
        done = records["done"] & valid
        num_environments, sequence_length = valid.shape
        positions = torch.arange(sequence_length, device=valid.device).view(1, -1)
        positions = positions.expand_as(valid)
        last_done = torch.where(done, positions, torch.full_like(positions, -1)).amax(dim=1)
        valid_count = valid.long().sum(dim=1)
        unfinished_length = torch.where(
            last_done >= 0,
            sequence_length - last_done - 1,
            valid_count,
        ).clamp(max=pending_capacity)

        copied_length = min(sequence_length, pending_capacity)
        pending = {}
        for key, value in records.items():
            if key == "valid":
                continue
            shape = (num_environments, pending_capacity, *value.shape[2:])
            buffered = torch.zeros(shape, dtype=value.dtype, device=value.device)
            if copied_length > 0:
                buffered[:, -copied_length:] = value[:, -copied_length:].detach()
            pending[key] = buffered

        pending_positions = torch.arange(pending_capacity, device=valid.device).view(1, -1)
        pending["valid"] = pending_positions >= (
            pending_capacity - unfinished_length
        ).unsqueeze(1)
        self._pending = pending
