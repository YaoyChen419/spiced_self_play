"""SPiCED observation and memory adapters for native FastTD3 networks."""

import torch
from torch import nn

from pufferlib.fast_td3 import Actor, Critic
from pufferlib.ocean.torch import Drive
from pufferlib.models import LSTMWrapper
from pufferlib.fast_td3_recurrent import SequenceObservations


class DriveEncoder(Drive):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        del self.actor, self.value_fn

    def forward(self, observations):
        return self.encode_observations(observations)


class DriveQInput(nn.Module):
    def __init__(self, encoder, n_act):
        super().__init__()
        self.encoder = encoder
        self.n_act = n_act

    def forward(self, inputs):
        observations, actions = inputs[:, :-self.n_act], inputs[:, -self.n_act:]
        return torch.cat((self.encoder(observations), actions), dim=1)


class DriveActor(Actor):
    def __init__(self, env, policy_kwargs, **kwargs):
        kwargs["n_obs"] = policy_kwargs.get("hidden_size", 128)
        super().__init__(**kwargs)
        encoder = DriveEncoder(env, **policy_kwargs).to(self.device)
        self.net = nn.Sequential(encoder, *self.net)


class DriveCritic(Critic):
    def __init__(self, env, policy_kwargs, **kwargs):
        kwargs["n_obs"] = policy_kwargs.get("hidden_size", 128)
        super().__init__(**kwargs)
        for qnet in (self.qnet1, self.qnet2):
            encoder = DriveEncoder(env, **policy_kwargs).to(self.device)
            qnet.net = nn.Sequential(DriveQInput(encoder, kwargs["n_act"]), *qnet.net)


class MemoryPolicy(DriveEncoder):
    """Expose native Drive features through LSTMWrapper's policy interface."""

    def decode_actions(self, hidden):
        return hidden, hidden.new_zeros(hidden.shape[0])


class DriveMemory(LSTMWrapper):
    def __init__(self, env, policy_kwargs, rnn_kwargs):
        super().__init__(env, MemoryPolicy(env, **policy_kwargs), **rnn_kwargs)

    def forward(self, observations):
        state = dict(lstm_h=None, lstm_c=None)
        super().forward(observations.sequence.transpose(0, 1), state)
        hidden = state['hidden'].transpose(0, 1)
        start = observations.offset
        return hidden[start:start + observations.mask.shape[0]][observations.mask]


class RecurrentDriveActor(Actor):
    is_recurrent = True

    def __init__(self, env, policy_kwargs, rnn_kwargs, **kwargs):
        kwargs['n_obs'] = rnn_kwargs['hidden_size']
        super().__init__(**kwargs)
        self.encoder = DriveMemory(env, policy_kwargs, rnn_kwargs).to(self.device)
        self.hidden_size = rnn_kwargs['hidden_size']

    def forward(self, observations):
        if isinstance(observations, SequenceObservations):
            observations = self.encoder(observations)
        return super().forward(observations)

    def _encode_step(self, observations, state, dones):
        if dones is not None:
            for key in ('lstm_h', 'lstm_c'):
                if state[key] is not None:
                    state[key] = state[key].masked_fill(dones.reshape(-1, 1).bool(), 0)
        return self.encoder.forward_eval(observations, state)

    def forward_eval(self, observations, state):
        hidden, values = self._encode_step(observations, state, state.get('done'))
        return super().forward(hidden), values

    def explore(self, obs, dones=None, state=None):
        if state is None:
            raise ValueError('Recurrent exploration requires per-vehicle LSTM state')
        hidden, _ = self._encode_step(obs, state, dones)
        # Native exploration, including noise-scale resampling, stays in Actor.explore.
        return super().explore(hidden, dones)


class RecurrentDriveCritic(Critic):
    def __init__(self, env, policy_kwargs, rnn_kwargs, **kwargs):
        kwargs['n_obs'] = rnn_kwargs['hidden_size']
        super().__init__(**kwargs)
        self.encoders = nn.ModuleList(
            DriveMemory(env, policy_kwargs, rnn_kwargs).to(self.device) for _ in range(2)
        )

    def forward(self, observations, actions):
        return (self.qnet1(self.encoders[0](observations), actions),
                self.qnet2(self.encoders[1](observations), actions))

    def projection(self, observations, actions, rewards, bootstrap, discount):
        return tuple(qnet.projection(
            encoder(observations), actions, rewards, bootstrap, discount,
            self.q_support, self.q_support.device,
        ) for qnet, encoder in zip((self.qnet1, self.qnet2), self.encoders))
