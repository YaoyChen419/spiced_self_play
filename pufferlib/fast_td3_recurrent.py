"""ICML 2022 sequence replay, adapted to FastTD3's vector/GPU storage.

Source: twni2016/pomdp-baselines, commit e7c19c32a20033d75414b29fbc466c77c211e968,
buffers/seq_replay_buffer_vanilla.py and policies/models/policy_rnn.py.
Copyright (c) 2021-22 Tianwei Ni. MIT license: fast_td3_recurrent.LICENSE.
"""

from typing import NamedTuple

import torch
from tensordict import TensorDict

from pufferlib.fast_td3_utils import SimpleReplayBuffer


class SequenceObservations(NamedTuple):
    sequence: torch.Tensor  # (B, T+1, obs), native SPiCED LSTMWrapper layout
    indices: torch.Tensor  # valid steps in the wrapper's flattened (B * (T+1), H) output


def sequence_batch(data, mask, initial_observations, obs_normalizer=None):
    """Prepare layout and valid-step indices once; keep native FastTD3 loss means."""
    length, batch_size = mask.shape
    steps = mask.flatten().nonzero(as_tuple=True)[0]
    # Loss tensors retain (T, B) order; LSTMWrapper returns (B, T+1) order.
    memory_steps = (steps % batch_size) * (length + 1) + steps // batch_size
    indices = torch.stack((steps, memory_steps, memory_steps + 1)).to(data.device)
    sequence = torch.cat((initial_observations[:, None],
                          data['next', 'observations'].transpose(0, 1)), dim=1)
    if obs_normalizer is not None:
        # Statistics are collected from live observations, not repeatedly from replay.
        sequence = obs_normalizer(sequence.flatten(0, 1), update=False).view_as(sequence)

    def select(value):
        return value.flatten(0, 1).index_select(0, indices[0])

    return {
        'observations': SequenceObservations(sequence, indices[1]),
        'actions': select(data['actions']),
        'next': {
            'observations': SequenceObservations(sequence, indices[2]),
            **{key: select(value) for key, value in data['next'].items() if key != 'observations'},
        },
    }


class SequenceReplayBuffer(SimpleReplayBuffer):
    """Completed trajectories, valid starts and masked subsequences from ICML 2022.

    Keep FastTD3's storage; vectorize episode indexing on CPU instead of copying
    each vehicle's trajectory to a separate NumPy buffer.
    """

    def __init__(self, sampled_seq_len, sample_weight_baseline=0.0, **kwargs):
        if kwargs.get('n_steps', 1) != 1 or kwargs.get('asymmetric_obs', False):
            raise ValueError('Recurrent Drive replay requires one-step symmetric transitions')
        if not 2 <= sampled_seq_len <= kwargs['buffer_size'] or sample_weight_baseline < 0:
            raise ValueError('Invalid sequence length or sample weight baseline')
        kwargs['valid_device'] = 'cpu'
        super().__init__(**kwargs)
        self._sampled_seq_len = sampled_seq_len
        self._sample_weight_baseline = sample_weight_baseline
        self._valid_starts = torch.zeros(self.n_env, self.buffer_size)
        self._ends = torch.zeros(self.n_env, self.buffer_size, dtype=torch.bool)
        self._episode_start = torch.zeros(self.n_env, dtype=torch.long)
        self._offsets = torch.arange(self.buffer_size)

    @property
    def ready(self):
        return bool(self._valid_starts.any())

    def _compute_valid_starts(self, seq_len):
        # SeqReplayBuffer._compute_valid_starts, batched over completed vehicles.
        num_valid_starts = (seq_len - self._sampled_seq_len + 1).clamp(min=1)
        total_weights = self._sample_weight_baseline + num_valid_starts
        weights = total_weights / num_valid_starts
        return weights[:, None] * (self._offsets[None] < num_valid_starts[:, None])

    @torch.no_grad()
    def extend(self, tensor_dict, valid=None, ends=None):
        valid = tensor_dict['valid'].cpu() if valid is None else valid
        ends = tensor_dict['next', 'dones'].bool().cpu() if ends is None else ends
        ptr = self.ptr
        super().extend(tensor_dict, valid=valid)
        self._valid_starts[:, ptr % self.buffer_size] = 0
        self._ends[:, ptr % self.buffer_size] = ends | ~valid

        rows = (ends & valid).nonzero(as_tuple=True)[0]
        lengths = ptr + 1 - self._episode_start[rows]
        # An overwritten episode is not a complete trajectory.
        rows, lengths = rows[lengths <= self.buffer_size], lengths[lengths <= self.buffer_size]
        starts = self._compute_valid_starts(lengths)
        slots = (self._episode_start[rows, None] + self._offsets[None]) % self.buffer_size
        usable = starts > 0
        self._valid_starts[rows[:, None].expand_as(slots)[usable], slots[usable]] = starts[usable]
        self._episode_start[ends | ~valid] = ptr + 1

    def _sample_indices(self, batch_size):
        # SeqReplayBuffer._sample_indices: probability proportional to valid-start weights.
        return torch.multinomial(self._valid_starts.flatten(), batch_size, replacement=True)

    def _generate_masks(self, rows, slots, absolute):
        # Upstream episode masks; absolute indices distinguish retained data from overwritten slots.
        ends = self._ends[rows[:, None], slots]
        before_end = torch.cat((torch.ones_like(ends[:, :1]), ~ends[:, :-1]), dim=1).cumprod(1).bool()
        return (before_end & self.valid[rows[:, None], slots]
                & (absolute < self.ptr) & (absolute >= max(0, self.ptr - self.buffer_size)))

    @torch.no_grad()
    def sample(self, batch_size):
        starts = self._sample_indices(batch_size)
        rows, positions = starts // self.buffer_size, starts % self.buffer_size
        absolute = self.ptr - 1 - (self.ptr - 1 - positions) % self.buffer_size
        absolute = absolute[:, None] + self._offsets[None, :self._sampled_seq_len]
        slots = absolute % self.buffer_size
        mask = self._generate_masks(rows, slots, absolute).transpose(0, 1)
        rows, slots = rows.to(self.device), slots.to(self.device)

        def gather(storage):
            return storage[rows[:, None], slots].transpose(0, 1)

        initial_observations = self.observations[rows, slots[:, 0]]
        dones = gather(self.dones)
        return TensorDict({
            'actions': gather(self.actions),
            'next': {
                'observations': gather(self.next_observations),
                'rewards': gather(self.rewards),
                'dones': dones,
                'truncations': gather(self.truncations),
                'effective_n_steps': torch.ones_like(dones),
            },
        }, batch_size=[self._sampled_seq_len, batch_size], device=self.device), mask, initial_observations
