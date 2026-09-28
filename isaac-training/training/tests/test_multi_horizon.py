import pathlib
import sys
import unittest

import torch
from tensordict.tensordict import TensorDict


SCRIPT_DIR = pathlib.Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from multi_horizon import MultiHorizonSequenceBuffer, horizon_seconds_to_steps


def make_rollout(
    collision,
    clearance,
    progress,
    done,
    position_x,
    next_position_x,
):
    steps = len(collision)
    batch_size = [1, steps]
    device = torch.device("cpu")

    current_drone_state = torch.zeros(1, steps, 1, 13)
    next_drone_state = torch.zeros(1, steps, 1, 13)
    current_drone_state[0, :, 0, 0] = torch.tensor(position_x)
    next_drone_state[0, :, 0, 0] = torch.tensor(next_position_x)

    observation = TensorDict(
        {
            "state": torch.zeros(1, steps, 8),
            "lidar": torch.zeros(1, steps, 1, 36, 4),
            "dynamic_obstacle": torch.zeros(1, steps, 1, 5, 10),
        },
        batch_size=batch_size,
        device=device,
    )
    stats = TensorDict(
        {
            "collision": torch.tensor(collision).view(1, steps, 1),
            "front_clearance": torch.tensor(clearance).view(1, steps, 1),
            "goal_progress": torch.tensor(progress).view(1, steps, 1),
        },
        batch_size=batch_size,
        device=device,
    )
    return TensorDict(
        {
            "agents": TensorDict(
                {
                    "observation": observation,
                    "action_normalized": torch.full((1, steps, 1, 3), 0.5),
                },
                batch_size=batch_size,
                device=device,
            ),
            "info": TensorDict(
                {"drone_state": current_drone_state},
                batch_size=batch_size,
                device=device,
            ),
            "next": TensorDict(
                {
                    "stats": stats,
                    "done": torch.tensor(done, dtype=torch.bool).view(1, steps, 1),
                    "info": TensorDict(
                        {"drone_state": next_drone_state},
                        batch_size=batch_size,
                        device=device,
                    ),
                },
                batch_size=batch_size,
                device=device,
            ),
        },
        batch_size=batch_size,
        device=device,
    )


class MultiHorizonSequenceBufferTest(unittest.TestCase):
    def test_physical_horizons_convert_to_paper_step_counts(self):
        self.assertEqual(
            [horizon_seconds_to_steps(value, 0.016) for value in (0.25, 1.0, 2.0)],
            [16, 63, 125],
        )

    def test_long_horizon_target_matures_across_rollouts(self):
        buffer = MultiHorizonSequenceBuffer(
            horizons=(2, 4),
            dt=1.0,
            max_progress_speed=0.11,
        )
        first = make_rollout(
            collision=[0.0, 0.0],
            clearance=[1.0, 1.0],
            progress=[0.1, 0.1],
            done=[False, False],
            position_x=[0.0, 0.1],
            next_position_x=[0.1, 0.2],
        )
        second = make_rollout(
            collision=[1.0, 0.0],
            clearance=[0.5, 2.0],
            progress=[0.1, 0.1],
            done=[False, False],
            position_x=[0.2, 0.3],
            next_position_x=[0.3, 0.4],
        )

        self.assertIsNone(buffer.append(first))
        mature = buffer.append(second)

        self.assertIsNotNone(mature)
        self.assertEqual(int(mature.batch_size[0]), 1)
        torch.testing.assert_close(
            mature["_aux_future_collision"],
            torch.tensor([[0.0, 1.0]]),
        )
        torch.testing.assert_close(
            mature["_aux_future_clearance"],
            torch.tensor([[1.0, 0.5]]),
        )
        torch.testing.assert_close(
            mature["_aux_future_progress"],
            torch.tensor([[0.2, 0.4]]),
        )
        torch.testing.assert_close(
            mature["_aux_future_trap"],
            torch.tensor([[1.0, 0.0]]),
        )
        self.assertEqual(buffer.pending_count, 3)

    def test_paper_horizons_are_not_truncated_by_32_step_rollouts(self):
        buffer = MultiHorizonSequenceBuffer(
            horizons=(16, 63, 125),
            dt=0.016,
        )

        mature = None
        for rollout_index in range(4):
            first_step = rollout_index * 32
            collision = [0.0] * 32
            if first_step <= 100 < first_step + 32:
                collision[100 - first_step] = 1.0
            position_x = [0.01 * (first_step + step) for step in range(32)]
            next_position_x = [value + 0.01 for value in position_x]
            mature = buffer.append(
                make_rollout(
                    collision=collision,
                    clearance=[2.0] * 32,
                    progress=[0.1] * 32,
                    done=[False] * 32,
                    position_x=position_x,
                    next_position_x=next_position_x,
                )
            )
            if rollout_index < 3:
                self.assertIsNone(mature)

        self.assertIsNotNone(mature)
        self.assertEqual(int(mature.batch_size[0]), 4)
        torch.testing.assert_close(
            mature["_aux_future_collision"][0],
            torch.tensor([0.0, 0.0, 1.0]),
        )
        torch.testing.assert_close(
            mature["_aux_future_progress"][0],
            torch.tensor([1.6, 6.3, 12.5]),
        )
        self.assertEqual(buffer.pending_count, 124)

    def test_terminal_window_does_not_cross_episode_reset(self):
        buffer = MultiHorizonSequenceBuffer(horizons=(2, 4), dt=1.0)
        rollout = make_rollout(
            collision=[0.0, 1.0, 0.0],
            clearance=[2.0, 1.0, 0.01],
            progress=[0.1, 0.1, -5.0],
            done=[False, True, False],
            position_x=[0.0, 0.1, 20.0],
            next_position_x=[0.1, 0.2, 15.0],
        )

        mature = buffer.append(rollout)

        self.assertIsNotNone(mature)
        self.assertEqual(int(mature.batch_size[0]), 2)
        torch.testing.assert_close(
            mature["_aux_future_collision"],
            torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
        )
        torch.testing.assert_close(
            mature["_aux_future_clearance"],
            torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
        )
        torch.testing.assert_close(
            mature["_aux_future_progress"],
            torch.tensor([[0.2, 0.2], [0.1, 0.1]]),
        )
        self.assertEqual(buffer.pending_count, 1)


if __name__ == "__main__":
    unittest.main()
