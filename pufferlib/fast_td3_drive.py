"""SPiCED observation adapters for native FastTD3 networks."""

import torch
from torch import nn

from pufferlib.fast_td3 import Actor, Critic
from pufferlib.ocean.torch import Drive


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
