"""Native action-interface regressions. Run from the repository root:

    python -m unittest discover -s tests -p test_drive_continuous_actions.py -v

Requires the Drive C extension. A bundled sanity scene is serialized for testing.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pufferlib.models import LSTMWrapper
from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive, save_map_binary
from pufferlib.ocean.torch import Drive as DrivePolicy


ROOT = Path(__file__).resolve().parents[1]


class TestDriveContinuousActions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.maps = tempfile.TemporaryDirectory(prefix="drive_actions_")
        cls.addClassCleanup(cls.maps.cleanup)
        source = ROOT / "pufferlib/resources/drive/sanity/sanity_jsons/two_agent_forward_goal_in_front.json"
        scene = json.loads(source.read_text())
        scene["scenario_id"] = str(scene["scenario_id"])
        save_map_binary(scene, Path(cls.maps.name) / "map_000.bin", 0)

    def make_env(self, dynamics_model="delta_local", action_type="continuous"):
        env = Drive(
            dynamics_model=dynamics_model,
            action_type=action_type,
            num_agents=8,
            num_maps=1,
            episode_length=110,
            termination_mode=1,
            map_dir=self.maps.name,
            ini_file_path=str(ROOT / "pufferlib/config/ocean/drive.ini"),
            control_mode="control_agents",
            async_resets=False,
            resample_frequency=0,
            fix_rewards=True,
            fix_lambdas=True,
            reg_mode="None",
        )
        self.addCleanup(env.close)
        env.reset(seed=0)
        return env

    def test_all_action_spaces_and_native_steps(self):
        discrete_bins = {
            "classic": [binding.NUM_ACCEL_BINS * binding.NUM_STEER_BINS],
            "jerk": [12],
            "delta_local": [binding.NUM_DX_BINS, binding.NUM_DY_BINS, binding.NUM_YAW_BINS],
        }
        for model in discrete_bins:
            for action_type in ("discrete", "continuous"):
                with self.subTest(model=model, action_type=action_type):
                    env = self.make_env(model, action_type)
                    if action_type == "continuous":
                        dim = 3 if model == "delta_local" else 2
                        self.assertEqual(env.single_action_space.shape, (dim,))
                        self.assertEqual(env.actions.dtype, np.dtype(np.float32))
                        np.testing.assert_array_equal(env.single_action_space.low, -np.ones(dim))
                        np.testing.assert_array_equal(env.single_action_space.high, np.ones(dim))
                    else:
                        np.testing.assert_array_equal(env.single_action_space.nvec, discrete_bins[model])
                        dim = len(discrete_bins[model])
                        self.assertEqual(env.actions.dtype, np.dtype(np.int32))
                    self.assertEqual(env.actions.shape, (env.num_agents, dim))
                    self.assertTrue(env.actions.flags.c_contiguous)
                    for _ in range(3):
                        obs, rewards, _, _, _ = env.step(np.zeros_like(env.actions))
                        self.assertTrue(np.isfinite(obs).all())
                        self.assertTrue(np.isfinite(rewards).all())

    def test_delta_local_yaw_is_per_agent_and_survives_resampling(self):
        env = self.make_env()
        # Fail before stepping C if the old two-float buffer regression returns.
        self.assertEqual(env.actions.shape, (env.num_agents, 3))
        for resample in (False, True):
            with self.subTest(resample=resample):
                if resample:
                    env.resample_maps()
                before = env.get_global_agent_state()
                actions = np.zeros_like(env.actions)
                actions[:, 2] = np.linspace(-0.3, 0.3, env.num_agents)
                env.step(actions)
                after = env.get_global_agent_state()
                valid = (before["x"] > -9999) & (after["x"] > -9999)
                self.assertGreater(np.count_nonzero(valid), 1)
                yaw_delta = after["heading"] - before["heading"]
                yaw_delta = (yaw_delta + np.pi) % (2 * np.pi) - np.pi
                np.testing.assert_allclose(yaw_delta[valid], actions[valid, 2] * np.pi / 6, atol=1e-6)

    def test_existing_lstm_policy_outputs_three_continuous_actions(self):
        env = self.make_env()
        self.assertEqual(env.single_action_space.shape, (3,))
        policy = LSTMWrapper(env, DrivePolicy(env, input_size=64, hidden_size=256), 256, 256)
        state = {"lstm_h": None, "lstm_c": None}
        with torch.no_grad():
            for _ in range(2):
                dist, value = policy.forward_eval(torch.from_numpy(env.observations.copy()), state)
                self.assertEqual(dist.loc.shape, (env.num_agents, 3))
                self.assertEqual(state["lstm_h"].shape, (env.num_agents, 256))
                self.assertTrue(torch.isfinite(value).all())
                env.step(dist.loc.clamp(-1, 1).numpy())


if __name__ == "__main__":
    unittest.main()
