"""FastTD3 heads with the driving encoder and recurrent, episodic replay."""
from collections import deque
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from fast_td3.fast_td3 import Actor, Critic
from pufferlib.ocean.torch import Drive
from pufferlib.models import LSTMWrapper


class Memory(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        original = LSTMWrapper(env, Drive(env, **args['policy']), **args['rnn'])
        self.encoder = original.policy
        del self.encoder.actor, self.encoder.value_fn
        self.hidden_size = args['rnn']['hidden_size']
        self.lstm = original.lstm
        self.lstm.batch_first = True

    def forward(self, obs, state=None):
        batch, steps = obs.shape[:2]
        features = self.encoder.encode_observations(obs.reshape(-1, obs.shape[-1]))
        return self.lstm(features.reshape(batch, steps, -1), state)

    def sequence(self, obs, prefix, lengths):
        # Rebuild memory with current weights from the actual episode start.
        # Prefix has no gradients; BPTT is limited to the learning window.
        h = obs.new_zeros(1, len(obs), self.hidden_size)
        c = torch.zeros_like(h)
        active = lengths > 0
        if active.any():
            with torch.no_grad():
                x = prefix[active]
                features = self.encoder.encode_observations(x.reshape(-1, x.shape[-1]))
                features = features.reshape(len(x), x.shape[1], -1)
                packed = pack_padded_sequence(features, lengths[active].cpu(),
                                              batch_first=True, enforce_sorted=False)
                _, (hp, cp) = self.lstm(packed)
                h[:, active], c[:, active] = hp, cp
        return self(obs, (h, c))[0]


class RecurrentActor(nn.Module):
    is_deterministic = True

    def __init__(self, env, args):
        super().__init__()
        self.memory = Memory(env, args)
        self.hidden_size = self.memory.hidden_size
        cfg = args['fasttd3']
        # Exploration is maintained by the collector, indexed by global agent ID.
        self.head = Actor(self.hidden_size, env.single_action_space.shape[0], 1,
                          cfg['init_scale'], cfg['actor_hidden_dim'],
                          cfg['std_min'], cfg['std_max'])

    def forward_eval(self, observations, state):
        recurrent = None
        if state.get('lstm_h') is not None:
            recurrent = (state['lstm_h'].unsqueeze(0), state['lstm_c'].unsqueeze(0))
        features, (h, c) = self.memory(observations[:, None], recurrent)
        state['lstm_h'], state['lstm_c'] = h[0], c[0]
        actions = self.head(features[:, 0])
        return actions, observations.new_zeros(len(observations), 1)


class RecurrentCritic(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.memory = Memory(env, args)
        cfg = args['fasttd3']
        self.head = Critic(self.memory.hidden_size, env.single_action_space.shape[0],
                           cfg['num_atoms'], cfg['v_min'], cfg['v_max'],
                           cfg['critic_hidden_dim'], '', 64, 8)


class EpisodeReplay:
    """CPU replay. Never join different vehicles or reset boundaries.

    Live episodes are vectorized; completed episodes retain their whole prefix.
    Capacity counts completed transitions, in addition to live episode storage.
    """
    def __init__(self, agents, episode_length, obs_dim, act_dim, capacity, seed):
        if capacity < episode_length:
            raise ValueError('replay_capacity must hold at least one whole episode')
        self.capacity, self.size = capacity, 0
        self.episodes = deque()
        self.rng = np.random.default_rng(seed)
        self.lengths = np.zeros(agents, dtype=np.int32)
        self.obs = np.empty((agents, episode_length + 1, obs_dim), np.float32)
        self.actions = np.empty((agents, episode_length, act_dim), np.float32)
        self.rewards = np.empty((agents, episode_length), np.float32)
        self.terminals = np.empty((agents, episode_length), bool)

    def add(self, ids, obs, actions, rewards, next_obs, terminals, boundaries):
        pos = self.lengths[ids]
        if np.any(pos >= self.actions.shape[1]):
            raise RuntimeError('Episode exceeded configured length; refusing to mix histories')
        self.obs[ids, pos] = obs
        self.obs[ids, pos + 1] = next_obs
        self.actions[ids, pos] = actions
        self.rewards[ids, pos] = rewards
        self.terminals[ids, pos] = terminals
        self.lengths[ids] += 1
        for agent in ids[boundaries]:
            length = int(self.lengths[agent])
            self.episodes.append((self.obs[agent, :length + 1].copy(),
                                  self.actions[agent, :length].copy(),
                                  self.rewards[agent, :length].copy(),
                                  self.terminals[agent, :length].copy()))
            self.size += length
            self.lengths[agent] = 0
        while self.size > self.capacity:
            self.size -= len(self.episodes.popleft()[1])

    def sample(self, sequences, horizon, device):
        episodes = list(self.episodes)
        # Uniform start transitions; padded suffixes do not contribute to losses.
        lengths = np.asarray([len(e[1]) for e in episodes])
        ends = lengths.cumsum()
        draws = self.rng.integers(0, ends[-1], sequences)
        selected = np.searchsorted(ends, draws, side='right')
        starts = draws - np.r_[0, ends[:-1]][selected]
        obs_dim, act_dim = self.obs.shape[-1], self.actions.shape[-1]
        prefix = np.zeros((sequences, max(1, starts.max()), obs_dim), np.float32)
        obs = np.zeros((sequences, horizon + 1, obs_dim), np.float32)
        actions = np.zeros((sequences, horizon, act_dim), np.float32)
        rewards = np.zeros((sequences, horizon), np.float32)
        terminals = np.ones((sequences, horizon), bool)
        mask = np.zeros((sequences, horizon), bool)
        for row, (idx, start) in enumerate(zip(selected, starts)):
            eo, ea, er, et = episodes[idx]
            n = min(horizon, len(ea) - start)
            prefix[row, :start] = eo[:start]
            obs[row, :n + 1] = eo[start:start + n + 1]
            actions[row, :n] = ea[start:start + n]
            rewards[row, :n] = er[start:start + n]
            terminals[row, :n] = et[start:start + n]
            mask[row, :n] = True
        return {key: torch.as_tensor(value, device=device) for key, value in
                dict(obs=obs, prefix=prefix, lengths=starts, actions=actions,
                     rewards=rewards, terminals=terminals, mask=mask).items()}


class Learner:
    def __init__(self, env, args):
        self.args, self.cfg = args, args['fasttd3']
        self.device = args['train']['device']
        self.actor = RecurrentActor(env, args).to(self.device)
        self.critic = RecurrentCritic(env, args).to(self.device)
        self.target = deepcopy(self.critic).requires_grad_(False)
        train = args['train']
        options = dict(betas=(train['adam_beta1'], train['adam_beta2']), eps=train['adam_eps'])
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg['actor_learning_rate'], **options)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg['critic_learning_rate'], **options)
        self.updates = 0

    @staticmethod
    def features(memory, batch):
        return memory.sequence(batch['obs'], batch['prefix'], batch['lengths'])

    def update(self, replay):
        cfg, train = self.cfg, self.args['train']
        horizon = train['rollout_horizon']
        sequences = train['minibatch_size'] // horizon
        micro = cfg['microbatch_sequences']
        # Sample on CPU first so gradient accumulation has the exact valid denominator.
        batches = [replay.sample(min(micro, sequences - i), horizon, 'cpu')
                   for i in range(0, sequences, micro)]
        count = sum(int(b['mask'].sum()) for b in batches)
        self.updates += 1
        actor_step = self.updates % cfg['policy_frequency'] == 0
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss = 0.0
        for cpu_batch in batches:
            b = {k: v.to(self.device) for k, v in cpu_batch.items()}
            mask = b['mask']
            with torch.no_grad():
                # FastTD3 uses the current actor and a target critic (no target actor).
                af = self.features(self.actor.memory, b)[:, 1:][mask]
                next_actions = self.actor.head(af)
                noise = (torch.randn_like(next_actions) * cfg['policy_noise']).clamp(
                    -cfg['noise_clip'], cfg['noise_clip'])
                next_actions = (next_actions + noise).clamp(-1, 1)
                tf = self.features(self.target.memory, b)[:, 1:][mask]
                q1, q2 = self.target.head.projection(tf, next_actions, b['rewards'][mask],
                    (~b['terminals'][mask]).float(),
                    torch.full_like(b['rewards'][mask], train['gamma']))
                if cfg['use_cdq']:
                    take1 = self.target.head.get_value(q1) < self.target.head.get_value(q2)
                    q1 = q2 = torch.where(take1[:, None], q1, q2)
            cf = self.features(self.critic.memory, b)[:, :-1][mask]
            logits1, logits2 = self.critic.head(cf, b['actions'][mask])
            loss = -(q1 * F.log_softmax(logits1, -1) + q2 * F.log_softmax(logits2, -1)).sum() / count
            loss.backward()
            critic_loss += loss.detach().item()
        nn.utils.clip_grad_norm_(self.critic.parameters(), train['max_grad_norm'])
        self.critic_opt.step()
        actor_loss = 0.0
        if actor_step:
            self.actor_opt.zero_grad(set_to_none=True)
            self.critic.requires_grad_(False)
            try:
                for cpu_batch in batches:
                    b = {k: v.to(self.device) for k, v in cpu_batch.items()}
                    mask = b['mask']
                    af = self.features(self.actor.memory, b)[:, :-1][mask]
                    with torch.no_grad():
                        cf = self.features(self.critic.memory, b)[:, :-1][mask]
                    q1, q2 = self.critic.head(cf, self.actor.head(af))
                    v1 = self.critic.head.get_value(q1.softmax(-1))
                    v2 = self.critic.head.get_value(q2.softmax(-1))
                    value = torch.minimum(v1, v2) if cfg['use_cdq'] else (v1 + v2) / 2
                    loss = -value.sum() / count
                    loss.backward()
                    actor_loss += loss.detach().item()
                nn.utils.clip_grad_norm_(self.actor.parameters(), train['max_grad_norm'])
                self.actor_opt.step()
            finally:
                self.critic.requires_grad_(True)
        with torch.no_grad():
            for target, source in zip(self.target.parameters(), self.critic.parameters()):
                target.lerp_(source, cfg['tau'])
        return dict(critic_loss=critic_loss, actor_loss=actor_loss,
                    actor_updated=int(actor_step), valid_samples=count)
