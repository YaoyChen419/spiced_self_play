"""Small CPU/native checks; no driving dataset or long training run required."""
import ast
import configparser
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from pufferlib.fasttd3 import EpisodeReplay, Learner, RecurrentActor
from pufferlib.fasttd3_train import train, transition_observations, validate
from pufferlib.ocean.drive.drive import Drive, save_map_binary

ROOT = Path(__file__).resolve().parents[1]


def config():
    parser = configparser.ConfigParser()
    parser.read([ROOT / 'pufferlib/config/default.ini', ROOT / 'pufferlib/config/ocean/drive.ini'])
    args = {}
    for section in parser.sections():
        values = {}
        for key, value in parser[section].items():
            try:
                values[key] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                values[key] = value
        if section == 'base':
            args.update(values)
        else:
            args[section] = values
    args.update(algorithm='fasttd3', wandb=False, neptune=False, load_model_path=None, load_id=None)
    args['env'].update(action_type='continuous', num_agents=8, num_maps=1,
                       episode_length=4, termination_mode=0, resample_frequency=0,
                       reg_mode='None', ini_file_path=str(ROOT / 'pufferlib/config/ocean/drive.ini'))
    args['policy'].update(input_size=8, hidden_size=16)
    args['rnn'].update(input_size=16, hidden_size=16)
    args['train'].update(device='cpu', use_rnn=True, rollout_horizon=2, minibatch_size=8,
                         batch_size=16, total_timesteps=64)
    args['fasttd3'].update(actor_hidden_dim=16, critic_hidden_dim=32, num_atoms=11,
                           v_min=-10., v_max=10., replay_capacity=64,
                           microbatch_sequences=2, learning_starts=1, num_updates=1)
    args['vec'] = dict(backend='Serial', num_envs=1)
    args['eval'].update(human_replay_eval=False, self_play_eval=False)
    return args


class TestFastTD3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.maps = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.maps.cleanup)
        scene = json.loads((ROOT / 'pufferlib/resources/drive/sanity/sanity_jsons/two_agent_forward_goal_in_front.json').read_text())
        scene['scenario_id'] = str(scene['scenario_id'])
        save_map_binary(scene, Path(cls.maps.name) / 'map_000.bin', 0)

    def make_env(self, **overrides):
        args = config()
        args['env'].update(map_dir=self.maps.name, **overrides)
        env = Drive(**args['env'])
        self.addCleanup(env.close)
        env.reset(seed=0)
        return env, args

    def test_final_observation_and_resample(self):
        automatic, _ = self.make_env(capture_final_observations=True)
        manual, _ = self.make_env(capture_final_observations=True, async_resets=False)
        # The original native boundary is (timestep + 1) >= episode_length.
        for _ in range(3):
            actions = np.zeros_like(automatic.actions)
            actions[:, 0] = .1
            obs, r, d, t, info = automatic.step(actions)
            reference, rr, rd, rt, ri = manual.step(actions)
        self.assertTrue(t.all())
        final, true_d = transition_observations(obs, d, info, True)
        np.testing.assert_allclose(final, reference)
        np.testing.assert_array_equal(true_d, rd)
        self.assertFalse(np.allclose(final, obs))
        np.testing.assert_array_equal(r, rr)
        automatic.resample_maps()  # Verify the optional native buffer survives re-creation.
        for _ in range(3):
            obs, r, d, t, info = automatic.step(actions)
        self.assertTrue(np.isfinite(transition_observations(obs, d, info, True)[0]).all())

    def test_resampling_keeps_true_terminal_mask(self):
        env, _ = self.make_env(capture_final_observations=True, resample_frequency=1)
        env.needs_resampling = True  # Exercise resampling even with the one-map fixture.
        reference, _ = self.make_env(capture_final_observations=True)
        action = np.zeros_like(env.actions)
        obs, _, d, t, info = env.step(action)
        expected, _, expected_d, _, _ = reference.step(action)
        final, true_d = transition_observations(obs, d, info, True)
        self.assertTrue(d.all() and t.all())
        np.testing.assert_allclose(final, expected)
        np.testing.assert_array_equal(true_d, expected_d)

    def test_replay_boundaries_and_prefix(self):
        replay = EpisodeReplay(2, 4, 1, 1, 8, 42)
        for step in range(3):
            obs = np.array([[step], [step + 100]], np.float32)
            replay.add(np.array([0, 1]), obs, np.zeros((2, 1)), np.zeros(2), obs + 1,
                       np.array([step == 2, False]), np.array([step == 2, step == 2]))
        self.assertEqual(replay.size, 6)
        self.assertTrue(replay.episodes[0][3][-1])
        self.assertFalse(replay.episodes[1][3][-1])  # Truncation still bootstraps.
        batch = replay.sample(20, 2, 'cpu')
        for row in range(20):
            n = int(batch['mask'][row].sum())
            values = batch['obs'][row, :n + 1, 0]
            torch.testing.assert_close(values[1:] - values[:-1], torch.ones(n))
            start = int(batch['lengths'][row])
            if start:
                self.assertEqual(batch['prefix'][row, start - 1, 0] + 1, values[0])

    def test_learning_memory_and_checkpoint(self):
        env, args = self.make_env()
        learner = Learner(env, args)
        replay = EpisodeReplay(8, 4, env.observations.shape[1], 3, 64, 42)
        obs = env.observations.copy()
        for step in range(4):
            action = np.zeros_like(env.actions)
            successor, reward, done, trunc, _ = env.step(action)
            replay.add(np.arange(8), obs, action, reward, successor, done, done | trunc)
            obs = successor.copy()
        before = [p.detach().clone() for p in learner.actor.parameters()]
        first = learner.update(replay)
        self.assertEqual(first['actor_updated'], 0)
        for old, new in zip(before, learner.actor.parameters()):
            torch.testing.assert_close(old, new)
        second = learner.update(replay)
        self.assertEqual(second['actor_updated'], 1)
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, learner.actor.parameters())))
        self.assertTrue(all(np.isfinite(v) for v in second.values()))
        b = replay.sample(2, 2, 'cpu')
        # Full-prefix state reconstruction agrees with ordinary sequential inference.
        for row in range(2):
            length = int(b['lengths'][row])
            full = torch.cat((b['prefix'][row:row + 1, :length], b['obs'][row:row + 1]), dim=1)
            with torch.no_grad():
                expected = learner.actor.memory(full)[0][:, length:]
                actual = learner.actor.memory.sequence(b['obs'][row:row + 1],
                    b['prefix'][row:row + 1], b['lengths'][row:row + 1])
            torch.testing.assert_close(actual, expected)
        state = {}
        with torch.no_grad():
            action, _ = learner.actor.forward_eval(torch.from_numpy(obs), state)
        self.assertEqual(state['lstm_h'].shape, (8, 16))
        self.assertTrue((action.abs() <= 1).all())
        restored = RecurrentActor(env, args)
        restored.load_state_dict(learner.actor.state_dict())
        with torch.no_grad():
            torch.testing.assert_close(restored.forward_eval(torch.from_numpy(obs), {})[0], action)

    def test_native_training_and_reload(self):
        from pufferlib.pufferl import load_policy
        args = config()
        args['env']['map_dir'] = self.maps.name
        with tempfile.TemporaryDirectory() as output:
            args['train']['data_dir'] = output
            logs = train('puffer_drive', args)
            self.assertGreater(logs[-1]['updates'], 0)
            self.assertGreaterEqual(logs[-1]['global_step'], 64)
            checkpoint = next(Path(output).glob('*.pt'))
            args['load_model_path'] = str(checkpoint)
            env, _ = self.make_env()
            class VectorView:
                driver_env = env
            policy = load_policy(args, VectorView(), 'puffer_drive')
            with torch.no_grad():
                self.assertTrue(torch.isfinite(policy.forward_eval(torch.from_numpy(env.observations), {})[0]).all())

    def test_multiprocessing_collector(self):
        args = config()
        args['env']['map_dir'] = self.maps.name
        args['vec'] = dict(backend='Multiprocessing', num_envs=2, num_workers=2,
                           batch_size=1, zero_copy=True, overwork=True)
        args['train']['total_timesteps'] = 96
        with tempfile.TemporaryDirectory() as output:
            args['train']['data_dir'] = output
            logs = train('puffer_drive', args)
            self.assertGreater(logs[-1]['updates'], 0)
            self.assertEqual(logs[-1]['global_step'], 96)

    def test_reject_nonzero_regularization(self):
        args = config()
        args['env']['lambda_value'] = .03
        with self.assertRaisesRegex(ValueError, 'lambda=0'):
            validate(args)


if __name__ == '__main__':
    unittest.main()
