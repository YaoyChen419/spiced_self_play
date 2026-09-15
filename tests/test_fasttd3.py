"""Regression checks for SPiCED / FastTD3 integration (no map data required)."""
import ast
from functools import partial
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import gymnasium as gym
import numpy as np
import torch
from tensordict import from_module

from pufferlib.fast_td3 import Actor, Critic
from pufferlib.fast_td3_encoder import DriveEncoder
from pufferlib.fast_td3_train import PufferDriveEnv
from pufferlib.ocean.torch import Drive


def layout():
    return SimpleNamespace(
        single_observation_space=gym.spaces.Box(-1, 1, (61,)),
        single_action_space=gym.spaces.Box(-1, 1, (3,)),
        ego_features=12, max_partner_objects=3, partner_features=7,
        max_road_objects=4, road_features=7,
    )


def observations(n=8):
    x = torch.randn(n, 61)
    x[:, 39::7] = torch.arange(4) % 7
    return x


class FakeVector:
    num_agents = 8
    observation_space = SimpleNamespace(shape=(4, 61))
    single_observation_space = SimpleNamespace(shape=(61,))
    single_action_space = SimpleNamespace(shape=(3,))

    def __init__(self):
        self.ids = np.arange(4)
        self.obs = np.ones((4, 61), dtype=np.float32)
        self.term = np.zeros(4, dtype=bool)
        self.trunc = np.zeros(4, dtype=bool)
        self.original_term = self.term.copy()

    def async_reset(self, seed):
        pass

    def send(self, actions):
        self.sent = actions.copy()

    def recv(self):
        metadata = dict(count=4, indices=np.arange(4),
                        observations=self.obs.copy(), terminals=self.original_term.copy())
        return (self.obs.copy(), np.arange(4, dtype=np.float32),
                self.term.copy(), self.trunc.copy(),
                [{"_fasttd3_transition": metadata}], self.ids.copy(), np.ones(4, dtype=bool))


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_original_encoder_output_and_initialization(self):
        original = Drive(layout(), input_size=16, hidden_size=16)
        torch.manual_seed(42)
        encoder = DriveEncoder(layout(), input_size=16, hidden_size=16)
        x = observations()
        torch.testing.assert_close(encoder(x), original.encode_observations(x), rtol=0, atol=0)
        self.assertFalse(any(k.startswith(('actor.', 'value_fn.')) for k in encoder.state_dict()))

    def test_original_object_permutation_invariance(self):
        encoder = DriveEncoder(layout(), input_size=16, hidden_size=16)
        x = observations()
        permuted = x.clone()
        permuted[:, 12:33] = x[:, 12:33].reshape(-1, 3, 7).flip(1).flatten(1)
        permuted[:, 33:] = x[:, 33:].reshape(-1, 4, 7).flip(1).flatten(1)
        torch.testing.assert_close(encoder(x), encoder(permuted), rtol=0, atol=0)

    def test_terminal_priority_and_pure_timeout(self):
        vec = FakeVector()
        env = PufferDriveEnv(vec, 'cpu', 42)
        env.reset()
        vec.term[:] = [True, True, False, False]
        vec.original_term[:] = vec.term
        vec.trunc[:] = [False, True, True, False]
        vec.obs[:] = 50
        _, _, dones, info = env.step(torch.zeros(4, 3))
        bootstrap = info['time_outs'] | ~dones
        self.assertEqual(bootstrap.tolist(), [False, False, True, True])
        self.assertTrue(info['valid'].all())  # Keep the successful transition.
        self.assertEqual(env.agent_dead[:4].tolist(), [True, False, False, False])
        self.assertTrue((info['observations']['raw']['obs'][:2] == 0).all())
        self.assertTrue((info['observations']['raw']['obs'][2:] == 50).all())
        _, _, _, info = env.step(torch.zeros(4, 3))
        self.assertFalse(info['valid'][0])  # Drop post-terminal padding.

    def test_map_resample_preserves_true_terminal(self):
        vec = FakeVector()
        env = PufferDriveEnv(vec, 'cpu', 42)
        env.reset()
        vec.term[:] = True  # resample_maps overwrites these flags
        vec.trunc[:] = True
        vec.original_term[:] = [False, True, False, False]
        _, _, dones, info = env.step(torch.zeros(4, 3))
        self.assertEqual(info['time_outs'].tolist(), [True, False, True, True])
        self.assertTrue(dones.all())
        self.assertFalse(env.agent_dead.any())

    def test_async_transition_uses_matching_agent_ids(self):
        vec = FakeVector()
        env = PufferDriveEnv(vec, 'cpu', 42)
        env.reset()
        vec.ids = np.arange(4, 8)
        vec.obs[:] = 2
        _, _, _, info = env.step(torch.ones(4, 3) * .25)
        self.assertIsNone(info['transition'])
        self.assertFalse(info['valid'].any())
        vec.ids = np.arange(4)
        vec.obs[:] = 3
        _, _, _, info = env.step(torch.ones(4, 3) * .75)
        torch.testing.assert_close(info['transition'][1], torch.ones(4, 61))
        torch.testing.assert_close(info['transition'][2], torch.ones(4, 3) * .25)

    def test_eval_loader_new_and_legacy_checkpoints(self):
        # Load the actual loader without importing optional rendering dependencies.
        tree = ast.parse((Path(__file__).parents[1] / "pufferlib/pufferl.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "load_fasttd3_policy")
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "pufferl.py", "exec"), namespace)
        for structured in (False, True):
            with self.subTest(structured=structured), tempfile.TemporaryDirectory() as tmp:
                factory = partial(DriveEncoder, layout(), input_size=16, hidden_size=16) if structured else None
                actor = Actor(n_obs=61, n_act=3, num_envs=8, init_scale=.01,
                              hidden_dim=32, device="cpu", encoder_factory=factory)
                checkpoint_args = dict(num_envs=8, init_scale=.01, actor_hidden_dim=32,
                                       std_min=.05, std_max=.8, sim_type="", sim_dimension=64,
                                       actor_seq_len=8, obs_normalization=False)
                path = str(Path(tmp) / "model.pt")
                torch.save(dict(actor_state_dict=actor.state_dict(), args=checkpoint_args), path)
                restored, normalizer = namespace["load_fasttd3_policy"](
                    dict(train={"device": "cpu"}, load_model_path=path, load_id=None),
                    SimpleNamespace(driver_env=layout()))
                x = observations()
                torch.testing.assert_close(actor(x), restored(normalizer(x)), rtol=0, atol=0)

    def test_actor_critic_gradients_targets_and_checkpoint(self):
        factory = partial(DriveEncoder, layout(), input_size=16, hidden_size=16)
        kwargs = dict(n_obs=61, n_act=3, num_envs=8, init_scale=.01,
                      hidden_dim=32, device='cpu', encoder_factory=factory)
        actor = Actor(**kwargs)
        detached = Actor(**kwargs)
        from_module(actor).data.to_module(detached)
        self.assertEqual(actor.encoder.ego_encoder[0].weight.data_ptr(),
                         detached.encoder.ego_encoder[0].weight.data_ptr())
        critic_args = dict(n_obs=61, n_act=3, num_atoms=101, v_min=-250., v_max=250.,
                           hidden_dim=32, sim_type='', sim_dimension=64, seq_len=8,
                           device='cpu', encoder_factory=factory)
        critic = Critic(**critic_args)
        target = Critic(**critic_args)
        target.load_state_dict(critic.state_dict())
        self.assertNotEqual(next(critic.qnet1.encoder.parameters()).data_ptr(),
                            next(critic.qnet2.encoder.parameters()).data_ptr())
        self.assertNotEqual(next(target.qnet1.encoder.parameters()).data_ptr(),
                            next(critic.qnet1.encoder.parameters()).data_ptr())
        x = observations()
        with torch.no_grad():
            projected, _ = target.projection(x, actor(x), torch.ones(8),
                                            torch.zeros(8), torch.ones(8) * .99)
            torch.testing.assert_close(projected.sum(1), torch.ones(8))
        q1, q2 = critic(x, actor(x))
        loss = -(projected * q1.log_softmax(1)).sum(1).mean()
        loss.backward()
        for model in (actor.encoder, critic.qnet1.encoder):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        optimizer = torch.optim.AdamW(actor.parameters(), lr=3e-4)
        optimizer.step()
        torch.testing.assert_close(actor(x), detached(x))
        restored = Actor(**kwargs)
        restored.load_state_dict(actor.state_dict())
        torch.testing.assert_close(actor(x), restored(x), rtol=0, atol=0)
        compiled = torch.compile(actor, backend='eager', fullgraph=True)
        torch.testing.assert_close(actor(x), compiled(x))


if __name__ == '__main__':
    unittest.main()
