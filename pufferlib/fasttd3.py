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


def float_logits(forward):
    """Keep official C51 projection in FP32 without disabling AMP in its MLP."""
    def wrapped(obs, actions):
        return forward(obs, actions).float()
    return wrapped


class Memory(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        original = LSTMWrapper(env, Drive(env, **args['policy']), **args['rnn'])
        self.encoder = original.policy
        del self.encoder.actor, self.encoder.value_fn
        self.hidden_size = args['rnn']['hidden_size']
        self.lstm = original.lstm
        self.lstm.batch_first = True
        self.encode_prefix = self.encoder.encode_observations

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
            # Pack before the expensive Drive encoder: padding must not consume
            # road/partner embedding work. LayerNorm is per observation, so
            # encoding packed observations preserves the original features.
            packed = pack_padded_sequence(prefix, lengths.clamp_min(1).cpu() if cpu_lengths is None else cpu_lengths,
                                          batch_first=True, enforce_sorted=False)
            packed = packed._replace(data=self.encode_prefix(packed.data))
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
        self.conditioning_indices = [env.lambda_obs_idx, env.reward_veh_obs_idx]
        self.conditioning = None
        self.critic = RecurrentCritic(env, args).to(self.device)
        self.target = deepcopy(self.critic).requires_grad_(False)
        for qnet in (self.target.head.qnet1, self.target.head.qnet2):
            qnet.forward = float_logits(qnet.forward)
        self.amp_enabled = self.cfg['amp'] and torch.device(self.device).type == 'cuda'
        self.amp_dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16}[self.cfg['amp_dtype']]
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp_enabled and self.amp_dtype == torch.float16)
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(),
            lr=torch.tensor(self.cfg['actor_learning_rate'], device=self.device), weight_decay=self.cfg['weight_decay'])
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(),
            lr=torch.tensor(self.cfg['critic_learning_rate'], device=self.device), weight_decay=self.cfg['weight_decay'])
        self.stable_grad_buffers = self.cfg['compile'] and self.cfg['compile_mode'] == 'reduce-overhead'
        if self.stable_grad_buffers:
            # Recurrent microbatches accumulate gradients. CUDA graphs require
            # their destination buffers to exist outside graph capture.
            for model in (self.actor, self.critic):
                for parameter in model.parameters():
                    parameter.grad = torch.zeros_like(parameter)
        if self.cfg['compile']:
            # cuDNN LSTM/packed sequences stay eager. Compile tensor-only regions
            # without wrapping modules, so checkpoint keys remain unchanged.
            for model in (self.actor, self.critic, self.target):
                # Episode prefixes have varying numbers of valid observations.
                # Compile their tensor work, but never record one CUDA graph
                # for every prefix length. Fixed rollout/loss paths keep graphs.
                model.memory.encode_prefix = torch.compile(
                    model.memory.encode_prefix, dynamic=True,
                    options={'triton.cudagraphs': False})
                model.memory.encoder.encode_observations = torch.compile(
                    model.memory.encoder.encode_observations, mode=self.cfg['compile_mode'], dynamic=False)
            self.actor.head.forward = torch.compile(self.actor.head.forward, mode=self.cfg['compile_mode'])
            for model in (self.critic, self.target):
                for qnet in (model.head.qnet1, model.head.qnet2):
                    qnet.forward = torch.compile(qnet.forward, mode=self.cfg['compile_mode'])
            self.target.head.projection = torch.compile(self.target.head.projection, mode=self.cfg['compile_mode'])
            self.critic_objective = torch.compile(self.critic_objective, mode=self.cfg['compile_mode'])
            self.actor_objective = torch.compile(self.actor_objective, mode=self.cfg['compile_mode'])
        self.updates = 0

    def critic_objective(self, features, actions, q1, q2, weights):
        logits1, logits2 = self.critic.head(features, actions)
        per_transition = -(q1 * F.log_softmax(logits1.float(), -1) +
                           q2 * F.log_softmax(logits2.float(), -1)).sum(-1)
        sequences = self.cfg['batch_size'] // self.args['train']['rollout_horizon']
        return (per_transition * weights).sum() / sequences

    def actor_objective(self, critic_features, actor_features, weights):
        q1, q2 = self.critic.head(critic_features, self.actor.head(actor_features))
        v1 = self.critic.head.get_value(q1.float().softmax(-1))
        v2 = self.critic.head.get_value(q2.float().softmax(-1))
        value = torch.minimum(v1, v2) if self.cfg['use_cdq'] else (v1 + v2) / 2
        sequences = self.cfg['batch_size'] // self.args['train']['rollout_horizon']
        return -(value * weights).sum() / sequences

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
        self.scaler.unscale_(optimizer)
        if self.cfg['use_grad_norm_clipping']:
            norm = nn.utils.clip_grad_norm_(model.parameters(),
                self.cfg['max_grad_norm'] if self.cfg['max_grad_norm'] > 0 else float('inf'))
        else:
            norm = torch.zeros((), device=self.device)
        self.scaler.step(optimizer)
        self.scaler.update()
        return norm

    @staticmethod
    def features(memory, batch):
        return memory.sequence(batch['obs'], batch['prefix'], batch['lengths'], batch['cpu_lengths'])

    def update(self, replay, actor_step=None):
        cfg, train = self.cfg, self.args['train']
        horizon = train['rollout_horizon']
        sequences = cfg['batch_size'] // horizon
        micro = cfg['microbatch_sequences']
        # Divide weighted losses by sampled starts, not the random sum of weights.
        # Each transition then contributes equally in expectation across windows.
        sample_device = getattr(replay, 'device', 'cpu')
        batches = [replay.sample(min(micro, sequences - i), horizon, sample_device)
                   for i in range(0, sequences, micro)]
        count = int(torch.stack([b['mask'].sum() for b in batches]).sum())
        batches = [{k: v.to(self.device) for k, v in b.items()} for b in batches]
        # Match the platform's data/* meaning: raw observations actually used
        # by the latest training batch, not a new collection snapshot.
        self.conditioning = torch.cat([
            b['obs'][:, :-1, self.conditioning_indices][b['mask']] for b in batches]).detach()
        # Packed cuDNN needs CPU lengths. Transfer once per optimizer update,
        # rather than synchronizing the GPU separately for every microbatch.
        cpu_lengths = torch.cat([b['lengths'] for b in batches]).clamp_min(1).cpu()
        offset = 0
        for b in batches:
            size = len(b['lengths'])
            b['cpu_lengths'] = cpu_lengths[offset:offset + size]
            offset += size
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
        if actor_step is None:
            actor_step = self.updates % cfg['policy_frequency'] == 0
        self.critic_opt.zero_grad(set_to_none=not self.stable_grad_buffers)
        critic_loss = 0.0
        qf_min = torch.full((), float('inf'), device=self.device)
        qf_max = -qf_min
        for b in batches:
            with torch.no_grad(), self.autocast():
                # FastTD3 uses the current actor and a target critic (no target actor).
                af = self.features(self.actor.memory, b)[:, 1:].reshape(-1, self.actor.hidden_size)
                next_actions = self.actor.head(af)
                noise = (torch.randn_like(next_actions) * cfg['policy_noise']).clamp(
                    -cfg['noise_clip'], cfg['noise_clip'])
                next_actions = (next_actions + noise).clamp(-1, 1)
                tf = self.features(self.target.memory, b)[:, 1:].reshape(-1, self.actor.hidden_size)
                # Target logits are promoted before softmax/index_add; the
                # target MLP itself still runs under the official AMP setting.
                q1, q2 = self.target.head.projection(tf, next_actions, b['rewards'].flatten(),
                    (~b['terminals']).float().flatten(),
                    torch.full_like(b['rewards'].flatten(), cfg['gamma']))
                values = self.target.head.get_value(q1)
                valid = b['mask'].flatten()
                qf_min = torch.minimum(qf_min, values.masked_fill(~valid, float('inf')).min())
                qf_max = torch.maximum(qf_max, values.masked_fill(~valid, -float('inf')).max())
                if cfg['use_cdq']:
                    take1 = self.target.head.get_value(q1) < self.target.head.get_value(q2)
                    q1 = q2 = torch.where(take1[:, None], q1, q2)
            with self.autocast():
                cf = self.features(self.critic.memory, b)[:, :-1].reshape(-1, self.actor.hidden_size)
                loss = self.critic_objective(cf, b['actions'].reshape(-1, b['actions'].shape[-1]),
                                            q1, q2, b['weights'].flatten())
            self.scaler.scale(loss).backward()
            critic_loss = critic_loss + loss.detach()
        critic_grad_norm = self.optimizer_step(self.critic_opt, self.critic)
        actor_loss = 0.0
        actor_grad_norm = torch.zeros((), device=self.device)
        if actor_step:
            self.actor_opt.zero_grad(set_to_none=not self.stable_grad_buffers)
            self.critic.requires_grad_(False)
            try:
                for b in batches:
                    with self.autocast():
                        af = self.features(self.actor.memory, b)[:, :-1].reshape(-1, self.actor.hidden_size)
                        with torch.no_grad():
                            cf = self.features(self.critic.memory, b)[:, :-1].reshape(-1, self.actor.hidden_size)
                        loss = self.actor_objective(cf, af, b['weights'].flatten())
                    self.scaler.scale(loss).backward()
                    actor_loss = actor_loss + loss.detach()
                actor_grad_norm = self.optimizer_step(self.actor_opt, self.actor)
            finally:
                self.critic.requires_grad_(True)
        with torch.no_grad():
            torch._foreach_mul_(list(self.target.parameters()), 1 - cfg['tau'])
            torch._foreach_add_(list(self.target.parameters()), list(self.critic.parameters()), alpha=cfg['tau'])
        buffer_reward = sum((b['rewards'] * b['mask']).sum() for b in batches) / max(count, 1)
        return dict(critic_loss=float(critic_loss), actor_loss=float(actor_loss),
                    qf_loss=float(critic_loss), qf_min=float(qf_min), qf_max=float(qf_max),
                    critic_grad_norm=float(critic_grad_norm), actor_grad_norm=float(actor_grad_norm),
                    buffer_rewards=float(buffer_reward),
                    actor_updated=int(actor_step), valid_samples=count, sampled_slots=cfg['batch_size'])
