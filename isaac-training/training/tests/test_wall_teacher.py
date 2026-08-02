import pathlib
import sys
import unittest
from types import SimpleNamespace

import torch


SCRIPT_DIR = pathlib.Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from demonstration_buffer import DemonstrationBuffer
from wall_follow_teacher import TeacherRolloutPolicy, WallFollowTeacher


def make_line_fitter():
    teacher = WallFollowTeacher.__new__(WallFollowTeacher)
    teacher.device = torch.device("cpu")
    teacher.fit_min_inliers = 5
    teacher.fit_residual_threshold = 0.06
    teacher.fit_min_pair_separation = 0.2
    teacher.fit_max_hypotheses = 256
    teacher.fit_min_span = 1.4
    return teacher


class WallGeometryTest(unittest.TestCase):
    def test_long_wall_is_fit_but_compact_face_is_rejected(self):
        teacher = make_line_fitter()
        y = torch.linspace(-2.0, 2.0, 17)
        wall = torch.stack([1.2 + 0.01 * torch.sin(3.0 * y), y], dim=-1)
        wall = torch.cat([wall, torch.tensor([[0.1, 2.7], [-0.4, -2.8]])], dim=0)

        fit = teacher._fit_one_line(wall)
        self.assertIsNotNone(fit)
        self.assertGreater(float(fit.span), 3.5)
        self.assertGreater(float(torch.abs(fit.normal[0])), 0.98)

        compact_y = torch.linspace(-0.4, 0.4, 7)
        compact_face = torch.stack([torch.ones_like(compact_y), compact_y], dim=-1)
        self.assertIsNone(teacher._fit_one_line(compact_face))

    def test_l_shape_produces_two_dominant_boundaries(self):
        teacher = make_line_fitter()
        vertical_y = torch.linspace(-2.2, 1.0, 15)
        horizontal_x = torch.linspace(1.0, 3.4, 12)
        vertical = torch.stack([torch.ones_like(vertical_y), vertical_y], dim=-1)
        horizontal = torch.stack([horizontal_x, torch.ones_like(horizontal_x)], dim=-1)

        fits = teacher.fit_lines_xy(torch.cat([vertical, horizontal], dim=0), max_lines=2)
        self.assertEqual(len(fits), 2)
        alignment = float(torch.abs(fits[0].tangent.dot(fits[1].tangent)))
        self.assertLess(alignment, 0.15)


class DynamicSafetyFallbackTest(unittest.TestCase):
    def test_vo_check_distinguishes_approaching_and_departing_obstacle(self):
        teacher = WallFollowTeacher.__new__(WallFollowTeacher)
        teacher.device = torch.device("cpu")
        teacher.dynamic_width_resolution = 0.25
        teacher.dynamic_spanning_height = 4.0
        teacher.vo_xy_margin = 0.3
        teacher.vo_z_margin = 0.3
        teacher.vo_horizon = 2.1
        teacher.vo_tau = 0.75
        dynamic = torch.zeros(1, 10)
        dynamic[0, 0] = 1.0
        dynamic[0, 3] = 2.0
        dynamic[0, 8] = 1.0
        dynamic[0, 9] = 1.0

        dynamic[0, 5] = -1.0
        approaching_risk = teacher._vo_risk(torch.zeros(3), dynamic)
        dynamic[0, 5] = 1.0
        departing_risk = teacher._vo_risk(torch.zeros(3), dynamic)

        self.assertGreater(float(approaching_risk), 0.0)
        self.assertEqual(float(departing_risk), 0.0)

    def test_unsafe_teacher_candidates_fall_back_to_policy(self):
        teacher = WallFollowTeacher.__new__(WallFollowTeacher)
        teacher.device = torch.device("cpu")
        teacher.action_limit = 2.0
        teacher.dynamic_risk_limit = 0.0
        teacher._vo_risk = lambda action, dynamic: torch.tensor(0.5)
        teacher_action = torch.tensor([0.8, 0.4, 0.0])
        policy_action = torch.tensor([-0.2, 0.3, 0.1])

        action, _, safe, is_teacher = teacher._select_dynamic_safe_action(
            teacher_action,
            policy_action,
            torch.empty(0),
        )
        self.assertTrue(torch.equal(action, policy_action))
        self.assertFalse(safe)
        self.assertFalse(is_teacher)


class DemonstrationBufferTest(unittest.TestCase):
    def test_masked_ring_buffer_and_sampling(self):
        buffer = DemonstrationBuffer(capacity=4, balanced_sampling=True)
        count = 6
        observation = {
            "state": torch.arange(count * 8, dtype=torch.float32).reshape(count, 8),
            "lidar": torch.zeros(count, 1, 36, 4),
            "direction": torch.zeros(count, 1, 3),
            "dynamic_obstacle": torch.zeros(count, 1, 5, 10),
        }
        action = torch.full((count, 3), 0.5)
        confidence = torch.linspace(0.4, 0.9, count)
        phase = torch.tensor([0, 1, 1, 2, 2, 3])

        written = buffer.add(
            observation,
            action,
            confidence,
            phase,
            torch.tensor([True, False, True, True, False, True]),
        )
        self.assertEqual(written, 4)
        self.assertEqual(len(buffer), 4)

        batch = buffer.sample(8, "cpu")
        self.assertEqual(tuple(batch.batch_size), (8,))
        self.assertEqual(tuple(batch["teacher_action_normalized"].shape), (8, 3))
        self.assertEqual(tuple(batch["agents", "observation", "lidar"].shape), (8, 1, 36, 4))
        self.assertEqual(sum(buffer.phase_counts().values()), 4)

    def test_pending_segment_is_written_only_when_committed(self):
        buffer = DemonstrationBuffer(capacity=16)
        wrapper = TeacherRolloutPolicy(
            base_policy=torch.nn.Linear(2, 2),
            teacher=SimpleNamespace(num_envs=1),
            demonstration_buffer=buffer,
        )
        self.assertTrue(any(name.startswith("base_policy.") for name, _ in wrapper.named_parameters()))
        observation = {
            "state": torch.zeros(1, 8),
            "lidar": torch.zeros(1, 1, 36, 4),
            "direction": torch.zeros(1, 1, 3),
            "dynamic_obstacle": torch.zeros(1, 1, 5, 10),
        }
        result = {
            "teacher_action_normalized": torch.full((1, 3), 0.5),
            "confidence": torch.ones(1),
            "phase": torch.zeros(1, dtype=torch.long),
        }

        wrapper._append_pending(0, observation, result)
        self.assertEqual(len(buffer), 0)
        self.assertEqual(wrapper._commit_pending(0), 1)
        self.assertEqual(len(buffer), 1)


if __name__ == "__main__":
    unittest.main()
