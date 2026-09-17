"""Recurrent FastTD3 networks for partially observed continuous control.

The recurrent layout follows the separate Actor/Critic architecture from
Ni, Eysenbach and Salakhutdinov (ICML 2022):
https://github.com/twni2016/pomdp-baselines

Adapted from the MIT-licensed implementation by Tianwei Ni (2021-2022).
"""

import torch
import torch.nn as nn

from pufferlib.fast_td3 import (
    DistributionalQNetwork,
    SimNormLinear,
    _validate_sim_config,
)


class RecurrentHistoryEncoder(nn.Module):
    """Encode (previous action, previous reward, current observation) histories."""

    def __init__(
        self,
        encoder_factory,
        n_act,
        action_embedding_size,
        reward_embedding_size,
        hidden_size,
        device,
    ):
        super().__init__()
        self.observation_encoder = encoder_factory().to(device)
        self.action_embedder = nn.Sequential(
            nn.Linear(n_act, action_embedding_size, device=device),
            nn.ReLU(),
        )
        self.reward_embedder = nn.Sequential(
            nn.Linear(1, reward_embedding_size, device=device),
            nn.ReLU(),
        )
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            self.observation_encoder.hidden_size
            + action_embedding_size
            + reward_embedding_size,
            hidden_size,
        ).to(device)

        for name, parameter in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(parameter, 0)
            elif "weight" in name:
                nn.init.orthogonal_(parameter)

    def encode_observations(self, observations):
        shape = observations.shape[:-1]
        encoded = self.observation_encoder(
            observations.reshape(-1, observations.shape[-1])
        )
        return encoded.reshape(*shape, -1)

    def forward(self, prev_actions, rewards, observations, state=None):
        encoded_observations = self.encode_observations(observations)
        inputs = torch.cat(
            (
                self.action_embedder(prev_actions),
                self.reward_embedder(rewards),
                encoded_observations,
            ),
            dim=-1,
        )
        hidden, state = self.lstm(inputs, state)
        return hidden, encoded_observations, state

    def initial_state(self, batch_size, device):
        shape = (1, batch_size, self.hidden_size)
        return torch.zeros(shape, device=device), torch.zeros(shape, device=device)


class RecurrentActor(nn.Module):
    def __init__(
        self,
        n_obs,
        n_act,
        num_envs,
        init_scale,
        hidden_dim,
        std_min=0.001,
        std_max=0.4,
        sim_type="",
        sim_dimension=64,
        seq_len=8,
        device=None,
        encoder_factory=None,
        rnn_hidden_size=256,
        action_embedding_size=32,
        reward_embedding_size=8,
    ):
        super().__init__()
        del n_obs
        _validate_sim_config(sim_type, sim_dimension, seq_len)
        self.n_act = n_act
        self.hidden_size = rnn_hidden_size
        self.history = RecurrentHistoryEncoder(
            encoder_factory,
            n_act,
            action_embedding_size,
            reward_embedding_size,
            rnn_hidden_size,
            device,
        )
        encoded_obs = self.history.observation_encoder.hidden_size
        self.net = nn.Sequential(
            nn.Linear(rnn_hidden_size + encoded_obs, hidden_dim, device=device),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.ReLU(),
        )
        if sim_type in ["sim_both", "sim_actor"]:
            self.fc_head = SimNormLinear(
                hidden_dim // 2,
                seq_len=seq_len,
                simnorm_dim=sim_dimension,
                device=device,
            )
            head_size = seq_len * sim_dimension
        else:
            self.fc_head = nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
                nn.ReLU(),
            )
            head_size = hidden_dim // 4
        self.fc_mu = nn.Sequential(nn.Linear(head_size, n_act, device=device), nn.Tanh())
        nn.init.normal_(self.fc_mu[0].weight, 0.0, init_scale)
        nn.init.constant_(self.fc_mu[0].bias, 0.0)

        noise_scales = torch.rand(num_envs, 1, device=device) * (
            std_max - std_min
        ) + std_min
        self.register_buffer("noise_scales", noise_scales)
        self.register_buffer("std_min", torch.as_tensor(std_min, device=device))
        self.register_buffer("std_max", torch.as_tensor(std_max, device=device))
        self.n_envs = num_envs
        self.device = device

    def decode(self, hidden, encoded_observations):
        features = torch.cat((hidden, encoded_observations), dim=-1)
        shape = features.shape[:-1]
        features = features.reshape(-1, features.shape[-1])
        actions = self.fc_mu(self.fc_head(self.net(features)))
        return actions.reshape(*shape, self.n_act)

    def forward_sequence(self, prev_actions, rewards, observations):
        hidden, encoded_observations, _ = self.history(
            prev_actions, rewards, observations
        )
        return self.decode(hidden, encoded_observations)

    def recurrent_step(self, observations, prev_actions, rewards, state):
        hidden, encoded_observations, state = self.history(
            prev_actions.unsqueeze(0),
            rewards.unsqueeze(0),
            observations.unsqueeze(0),
            state,
        )
        actions = self.decode(hidden, encoded_observations).squeeze(0)
        return actions, state

    def forward(self, observations):
        batch_size = observations.shape[0]
        prev_actions = torch.zeros(
            batch_size, self.n_act, device=observations.device
        )
        rewards = torch.zeros(batch_size, 1, device=observations.device)
        state = self.history.initial_state(batch_size, observations.device)
        actions, _ = self.recurrent_step(
            observations, prev_actions, rewards, state
        )
        return actions

    def explore_step(
        self,
        observations,
        prev_actions,
        rewards,
        state,
        dones=None,
        deterministic=False,
    ):
        if dones is not None and dones.any():
            new_scales = torch.rand(self.n_envs, 1, device=observations.device) * (
                self.std_max - self.std_min
            ) + self.std_min
            self.noise_scales.copy_(
                torch.where(dones.view(-1, 1), new_scales, self.noise_scales)
            )
        actions, state = self.recurrent_step(
            observations, prev_actions, rewards, state
        )
        if not deterministic:
            actions = actions + torch.randn_like(actions) * self.noise_scales
        return actions, state


class RecurrentCritic(nn.Module):
    def __init__(
        self,
        n_obs,
        n_act,
        num_atoms,
        v_min,
        v_max,
        hidden_dim,
        sim_type,
        sim_dimension,
        seq_len,
        device=None,
        encoder_factory=None,
        rnn_hidden_size=256,
        action_embedding_size=32,
        reward_embedding_size=8,
    ):
        super().__init__()
        del n_obs
        self.n_act = n_act
        self.history = RecurrentHistoryEncoder(
            encoder_factory,
            n_act,
            action_embedding_size,
            reward_embedding_size,
            rnn_hidden_size,
            device,
        )
        encoded_obs = self.history.observation_encoder.hidden_size
        q_obs = rnn_hidden_size + encoded_obs
        q_kwargs = dict(
            n_obs=q_obs,
            n_act=n_act,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            hidden_dim=hidden_dim,
            sim_type=sim_type,
            sim_dimension=sim_dimension,
            seq_len=seq_len,
            device=device,
        )
        self.qnet1 = DistributionalQNetwork(**q_kwargs)
        self.qnet2 = DistributionalQNetwork(**q_kwargs)
        self.register_buffer(
            "q_support", torch.linspace(v_min, v_max, num_atoms, device=device)
        )
        self.device = device

    def _sequence_features(self, prev_actions, rewards, observations):
        hidden, encoded_observations, _ = self.history(
            prev_actions, rewards, observations
        )
        return torch.cat((hidden, encoded_observations), dim=-1)

    def forward_sequence(
        self, prev_actions, rewards, observations, current_actions
    ):
        features = self._sequence_features(prev_actions, rewards, observations)
        if current_actions.shape[0] != observations.shape[0]:
            features = features[:-1]
        shape = current_actions.shape[:-1]
        flat_features = features.reshape(-1, features.shape[-1])
        flat_actions = current_actions.reshape(-1, current_actions.shape[-1])
        q1 = self.qnet1(flat_features, flat_actions)
        q2 = self.qnet2(flat_features, flat_actions)
        return q1.reshape(*shape, -1), q2.reshape(*shape, -1)

    def projection_sequence(
        self,
        prev_actions,
        previous_rewards,
        observations,
        next_actions,
        rewards,
        bootstrap,
        discount,
    ):
        features = self._sequence_features(
            prev_actions, previous_rewards, observations
        )[1:]
        shape = next_actions.shape[:-1]
        flat_features = features.reshape(-1, features.shape[-1])
        flat_actions = next_actions.reshape(-1, next_actions.shape[-1])
        flat_rewards = rewards.reshape(-1)
        flat_bootstrap = bootstrap.reshape(-1)
        flat_discount = discount.reshape(-1)
        q1 = self.qnet1.projection(
            flat_features,
            flat_actions,
            flat_rewards,
            flat_bootstrap,
            flat_discount,
            self.q_support,
            self.q_support.device,
        )
        q2 = self.qnet2.projection(
            flat_features,
            flat_actions,
            flat_rewards,
            flat_bootstrap,
            flat_discount,
            self.q_support,
            self.q_support.device,
        )
        return q1.reshape(*shape, -1), q2.reshape(*shape, -1)

    def forward(self, observations, actions):
        batch_size = observations.shape[0]
        prev_actions = torch.zeros(
            1, batch_size, self.n_act, device=observations.device
        )
        rewards = torch.zeros(1, batch_size, 1, device=observations.device)
        q1, q2 = self.forward_sequence(
            prev_actions, rewards, observations.unsqueeze(0), actions.unsqueeze(0)
        )
        return q1.squeeze(0), q2.squeeze(0)

    def get_value(self, probabilities):
        return torch.sum(probabilities * self.q_support, dim=-1)
