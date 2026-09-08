"""FastTD3 heads with the driving encoder and recurrent, episodic replay."""
from contextlib import nullcontext
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from fast_td3.fast_td3 import Actor, Critic
from fast_td3.fast_td3_utils import EmpiricalNormalization
from pufferlib.ocean.torch import Drive
from pufferlib.models import LSTMWrapper
from pufferlib.fasttd3_replay import DeviceEpisodeReplay as EpisodeReplay


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

    def sequence(self, obs, prefix, lengths, cpu_lengths=None):
        # Rebuild memory with current weights from the actual episode start.
        # Prefix has no gradients; BPTT is limited to the learning window.
        h = obs.new_zeros(1, len(obs), self.hidden_size)
        c = torch.zeros_like(h)
        with torch.no_grad():
            features = self.encoder.encode_observations(prefix.reshape(-1, prefix.shape[-1]))
            features = features.reshape(len(prefix), prefix.shape[1], -1)
            # Packed cuDNN requires CPU lengths. Zero-prefix rows get one dummy
            # step whose state is discarded; encoder batch shapes stay fixed.
            packed = pack_padded_sequence(features, lengths.clamp_min(1).cpu() if cpu_lengths is None else cpu_lengths,
                                          batch_first=True, enforce_sorted=False)
            _, (hp, cp) = self.lstm(packed)
            active = (lengths > 0)[None, :, None]
            h, c = torch.where(active, hp, h), torch.where(active, cp, c)
        return self(obs, (h, c))[0]


class RecurrentActor(nn.Module):
    is_deterministic = True

    def __init__(self, env, args):
        super().__init__()
        self.memory = Memory(env, args)
        self.hidden_size = self.memory.hidden_size
        cfg = args['fasttd3']
        self.obs_normalizer = (EmpiricalNormalization(env.single_observation_space.shape[0], 'cpu')
                               if cfg['obs_normalization'] else None)
        normalize_mask = torch.ones(env.single_observation_space.shape[0], dtype=torch.bool)
        road_start = env.ego_features + env.max_partner_objects * env.partner_features
        normalize_mask[road_start + env.road_features - 1::env.road_features] = False
        self.register_buffer('normalize_mask', normalize_mask)
        # Exploration is maintained by the collector, indexed by global agent ID.
        self.head = Actor(self.hidden_size, env.single_action_space.shape[0], 1,
                          cfg['init_scale'], cfg['actor_hidden_dim'],
                          cfg['std_min'], cfg['std_max'])

    def forward_eval(self, observations, state):
        observations = self.normalize(observations)
        recurrent = None
        if state.get('lstm_h') is not None:
            recurrent = (state['lstm_h'].unsqueeze(0), state['lstm_c'].unsqueeze(0))
        features, (h, c) = self.memory(observations[:, None], recurrent)
        state['lstm_h'], state['lstm_c'] = h[0], c[0]
        actions = self.head(features[:, 0])
        return actions, observations.new_zeros(len(observations), 1)

    def normalize(self, observations, update=False):
        if self.obs_normalizer is None:
            return observations
        flat = observations.reshape(-1, observations.shape[-1])
        normalized = self.obs_normalizer(flat, update=update).reshape_as(observations)
        # Road object type is a categorical ID consumed by the original one-hot
        # encoder. Standardization must never alter that ID.
        return torch.where(self.normalize_mask, normalized, observations)


class RecurrentCritic(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.memory = Memory(env, args)
        cfg = args['fasttd3']
        self.head = Critic(self.memory.hidden_size, env.single_action_space.shape[0],
                           cfg['num_atoms'], cfg['v_min'], cfg['v_max'],
                           cfg['critic_hidden_dim'], '', 64, 8)


class Learner:
    def __init__(self, env, args):
        self.args, self.cfg = args, args['fasttd3']
        self.device = args['train']['device']
        self.actor = RecurrentActor(env, args).to(self.device)
        self.critic = RecurrentCritic(env, args).to(self.device)
        self.target = deepcopy(self.critic).requires_grad_(False)
        self.amp_enabled = self.cfg['amp'] and torch.device(self.device).type == 'cuda'
        self.amp_dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16}[self.cfg['amp_dtype']]
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp_enabled and self.amp_dtype == torch.float16)
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(),
            lr=torch.tensor(self.cfg['actor_learning_rate'], device=self.device), weight_decay=self.cfg['weight_decay'])
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(),
            lr=torch.tensor(self.cfg['critic_learning_rate'], device=self.device), weight_decay=self.cfg['weight_decay'])
        if self.cfg['compile']:
            # cuDNN LSTM/packed sequences stay eager. Compile tensor-only regions
            # without wrapping modules, so checkpoint keys remain unchanged.
            for model in (self.actor, self.critic, self.target):
                model.memory.encoder.encode_observations = torch.compile(
                    model.memory.encoder.encode_observations, mode=self.cfg['compile_mode'])
            self.actor.head.forward = torch.compile(self.actor.head.forward, mode=self.cfg['compile_mode'])
            for model in (self.critic, self.target):
                for qnet in (model.head.qnet1, model.head.qnet2):
                    qnet.forward = torch.compile(qnet.forward, mode=self.cfg['compile_mode'])
        self.updates = 0

    def autocast(self):
        return torch.autocast('cuda', dtype=self.amp_dtype) if self.amp_enabled else nullcontext()

    def schedule(self, fraction):
        # CosineAnnealingLR expressed in environment progress, preserving the
        # original driving budget despite asynchronous partial vector batches.
        for opt, key in ((self.actor_opt, 'actor'), (self.critic_opt, 'critic')):
            start, end = self.cfg[key + '_learning_rate'], self.cfg[key + '_learning_rate_end']
            lr = end + (start - end) * (1 + np.cos(np.pi * min(1., fraction))) / 2
            opt.param_groups[0]['lr'].fill_(lr)

    def optimizer_step(self, optimizer, model):
        if self.cfg['use_grad_norm_clipping']:
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), self.cfg['max_grad_norm'])
        self.scaler.step(optimizer)

    @staticmethod
    def features(memory, batch):
        return memory.sequence(batch['obs'], batch['prefix'], batch['lengths'], batch['cpu_lengths'])

    def update(self, replay):
        cfg, train = self.cfg, self.args['train']
        horizon = train['rollout_horizon']
        sequences = train['minibatch_size'] // horizon
        micro = cfg['microbatch_sequences']
        # Divide weighted losses by sampled starts, not the random sum of weights.
        # Each transition then contributes equally in expectation across windows.
        sample_device = getattr(replay, 'device', 'cpu')
        batches = [replay.sample(min(micro, sequences - i), horizon, sample_device)
                   for i in range(0, sequences, micro)]
        count = int(torch.stack([b['mask'].sum() for b in batches]).sum())
        batches = [{k: v.to(self.device) for k, v in b.items()} for b in batches]
        for b in batches:
            b['cpu_lengths'] = b['lengths'].clamp_min(1).cpu()
        if self.actor.obs_normalizer is not None:
            # Update statistics on real transitions only, never padded windows.
            # Freeze one shared statistics snapshot for all three networks.
            with torch.no_grad():
                for b in batches:
                    self.actor.obs_normalizer.update(b['obs'][:, :-1][b['mask']])
                    self.actor.obs_normalizer.update(b['obs'][:, 1:][b['mask']])
                for b in batches:
                    b['obs'] = self.actor.normalize(b['obs'])
                    b['prefix'] = self.actor.normalize(b['prefix'])
        self.updates += 1
        actor_step = self.updates % cfg['policy_frequency'] == 0
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss = 0.0
        for b in batches:
            with torch.no_grad(), self.autocast():
                # FastTD3 uses the current actor and a target critic (no target actor).
                af = self.features(self.actor.memory, b)[:, 1:].reshape(-1, self.actor.hidden_size)
                next_actions = self.actor.head(af)
                noise = (torch.randn_like(next_actions) * cfg['policy_noise']).clamp(
                    -cfg['noise_clip'], cfg['noise_clip'])
                next_actions = (next_actions + noise).clamp(-1, 1)
                tf = self.features(self.target.memory, b)[:, 1:].reshape(-1, self.actor.hidden_size)
                # C51's index_add requires float32 source and destination; keep
                # projection out of autocast (the recurrent trunk still uses AMP).
                with torch.autocast('cuda', enabled=False):
                    q1, q2 = self.target.head.projection(tf.float(), next_actions.float(), b['rewards'].flatten(),
                        (~b['terminals']).float().flatten(),
                        torch.full_like(b['rewards'].flatten(), cfg['gamma']))
                if cfg['use_cdq']:
                    take1 = self.target.head.get_value(q1) < self.target.head.get_value(q2)
                    q1 = q2 = torch.where(take1[:, None], q1, q2)
            with self.autocast():
                cf = self.features(self.critic.memory, b)[:, :-1].reshape(-1, self.actor.hidden_size)
                logits1, logits2 = self.critic.head(cf, b['actions'].reshape(-1, b['actions'].shape[-1]))
            per_transition = -(q1 * F.log_softmax(logits1.float(), -1) +
                               q2 * F.log_softmax(logits2.float(), -1)).sum(-1)
            loss = (per_transition * b['weights'].flatten()).sum() / sequences
            self.scaler.scale(loss).backward()
            critic_loss = critic_loss + loss.detach()
        self.optimizer_step(self.critic_opt, self.critic)
        actor_loss = 0.0
        if actor_step:
            self.actor_opt.zero_grad(set_to_none=True)
            self.critic.requires_grad_(False)
            try:
                for b in batches:
                    with self.autocast():
                        af = self.features(self.actor.memory, b)[:, :-1].reshape(-1, self.actor.hidden_size)
                        with torch.no_grad():
                            cf = self.features(self.critic.memory, b)[:, :-1].reshape(-1, self.actor.hidden_size)
                        q1, q2 = self.critic.head(cf, self.actor.head(af))
                    v1 = self.critic.head.get_value(q1.float().softmax(-1))
                    v2 = self.critic.head.get_value(q2.float().softmax(-1))
                    value = torch.minimum(v1, v2) if cfg['use_cdq'] else (v1 + v2) / 2
                    loss = -(value * b['weights'].flatten()).sum() / sequences
                    self.scaler.scale(loss).backward()
                    actor_loss = actor_loss + loss.detach()
                self.optimizer_step(self.actor_opt, self.actor)
            finally:
                self.critic.requires_grad_(True)
        self.scaler.update()
        with torch.no_grad():
            torch._foreach_mul_(list(self.target.parameters()), 1 - cfg['tau'])
            torch._foreach_add_(list(self.target.parameters()), list(self.critic.parameters()), alpha=cfg['tau'])
        return dict(critic_loss=float(critic_loss), actor_loss=float(actor_loss),
                    actor_updated=int(actor_step), valid_samples=count)
