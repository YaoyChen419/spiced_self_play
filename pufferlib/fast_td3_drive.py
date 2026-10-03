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
        if isinstance(observations, SequenceObservations):
            hidden, _ = LSTMWrapper.forward(self, observations.sequence, state)
            return hidden.index_select(0, observations.indices)
        hidden, _ = LSTMWrapper.forward(self, observations, state)
        return hidden


class RecurrentDriveActor(Actor):
    is_recurrent = True

    def __init__(self, env, policy_kwargs, rnn_kwargs, **kwargs):
        kwargs['n_obs'] = rnn_kwargs['hidden_size']
        super().__init__(**kwargs)
        self.encoder = DriveMemory(env, policy_kwargs, rnn_kwargs).to(self.device)
        self.hidden_size = rnn_kwargs['hidden_size']

    def forward(self, observations, state=None):
        if isinstance(observations, SequenceObservations):
            return Actor.forward(self, self.encoder(observations))
        if state is None:
            state = dict(lstm_h=None, lstm_c=None)
        return self.forward_eval(observations, state)[0]

    def _encode_step(self, observations, state, dones):
        if dones is not None:
            for key in ('lstm_h', 'lstm_c'):
                if state[key] is not None:
                    state[key] = state[key].masked_fill(dones.reshape(-1, 1).bool(), 0)
        return LSTMWrapper.forward_eval(self.encoder, observations, state)

    def forward_eval(self, observations, state):
        hidden, values = self._encode_step(observations, state, state.get('done'))
        return Actor.forward(self, hidden), values

    def explore(self, obs, dones=None, deterministic=False, state=None):
        if state is None:
            raise ValueError('Recurrent exploration requires per-vehicle LSTM state')
        # Actor.explore source, with only its forward call adapted to explicit memory.
        if dones is not None and dones.sum() > 0:
            new_scales = (
                torch.rand(self.n_envs, 1, device=obs.device)
                * (self.std_max - self.std_min)
                + self.std_min
            )
            dones_view = dones.view(-1, 1) > 0
            self.noise_scales.copy_(
                torch.where(dones_view, new_scales, self.noise_scales)
            )

        hidden, _ = self._encode_step(obs, state, dones)
        act = Actor.forward(self, hidden)
        if deterministic:
            return act
        noise = torch.randn_like(act) * self.noise_scales
        return act + noise


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
