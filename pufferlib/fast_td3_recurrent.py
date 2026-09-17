"""Separate recurrent Actor/Critic adaptation for FastTD3.

The recurrent layout and sequence semantics are adapted directly from the
MIT-licensed pomdp-baselines implementation:
https://github.com/twni2016/pomdp-baselines
Copyright (c) 2021-2022 Tianwei Ni.

Only the policy/Q heads and observation encoder are replaced with FastTD3's
C51 heads and SPiCED's native Drive encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pufferlib.fast_td3 import (
    DistributionalQNetwork,
    SimNormLinear,
    _validate_sim_config,
)


class FeatureExtractor(nn.Module):
    """Official one-layer ReLU feature extractor, with device adaptation."""

    def __init__(self, input_size, output_size, device):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size, device=device)

    def forward(self, inputs):
        return F.relu(self.fc(inputs))


def _orthogonal_recurrent_init(module):
    for name, parameter in module.named_parameters():
        if "bias" in name:
            nn.init.constant_(parameter, 0)
        elif "weight" in name:
            nn.init.orthogonal_(parameter)


class RecurrentActor(nn.Module):
    """Actor_RNN from pomdp-baselines with FastTD3's deterministic head."""

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
        rnn_hidden_size=128,
        action_embedding_size=16,
        reward_embedding_size=16,
    ):
        super().__init__()
        del n_obs
        _validate_sim_config(sim_type, sim_dimension, seq_len)

        self.n_act = n_act
        self.hidden_size = rnn_hidden_size
        self.observation_encoder = encoder_factory().to(device)
        observation_embedding_size = self.observation_encoder.hidden_size
        self.action_embedder = FeatureExtractor(
            n_act, action_embedding_size, device
        )
        self.reward_embedder = FeatureExtractor(
            1, reward_embedding_size, device
        )
        self.rnn = nn.LSTM(
            action_embedding_size
            + observation_embedding_size
            + reward_embedding_size,
            rnn_hidden_size,
        ).to(device)
        _orthogonal_recurrent_init(self.rnn)

        self.net = nn.Sequential(
            nn.Linear(
                rnn_hidden_size + observation_embedding_size,
                hidden_dim,
                device=device,
            ),
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
        self.fc_mu = nn.Sequential(
            nn.Linear(head_size, n_act, device=device),
            nn.Tanh(),
        )
        nn.init.normal_(self.fc_mu[0].weight, 0.0, init_scale)
        nn.init.constant_(self.fc_mu[0].bias, 0.0)

        noise_scales = torch.rand(num_envs, 1, device=device) * (
            std_max - std_min
        ) + std_min
        self.register_buffer("noise_scales", noise_scales)
        self.register_buffer("std_min", torch.as_tensor(std_min, device=device))
        self.register_buffer("std_max", torch.as_tensor(std_max, device=device))
        self.n_envs = num_envs

    def _encode_observations(self, observations):
        shape = observations.shape[:-1]
        encoded = self.observation_encoder(
            observations.reshape(-1, observations.shape[-1])
        )
        return encoded.reshape(*shape, -1)

    def get_hidden_states(
        self,
        prev_actions,
        rewards,
        observations,
        initial_internal_state=None,
    ):
        encoded_observations = self._encode_observations(observations)
        inputs = torch.cat(
            (
                self.action_embedder(prev_actions),
                self.reward_embedder(rewards),
                encoded_observations,
            ),
            dim=-1,
        )
        hidden_states, internal_state = self.rnn(
            inputs, initial_internal_state
        )
        return hidden_states, encoded_observations, internal_state

    def _decode(self, hidden_states, encoded_observations):
        joint_embeds = torch.cat(
            (hidden_states, encoded_observations), dim=-1
        )
        shape = joint_embeds.shape[:-1]
        joint_embeds = joint_embeds.reshape(-1, joint_embeds.shape[-1])
        actions = self.fc_mu(self.fc_head(self.net(joint_embeds)))
        return actions.reshape(*shape, self.n_act)

    def forward_sequence(self, prev_actions, rewards, observations):
        hidden_states, encoded_observations, _ = self.get_hidden_states(
            prev_actions, rewards, observations
        )
        return self._decode(hidden_states, encoded_observations)

    def recurrent_step(self, observations, prev_actions, rewards, state):
        hidden_states, encoded_observations, state = self.get_hidden_states(
            prev_actions.unsqueeze(0),
            rewards.unsqueeze(0),
            observations.unsqueeze(0),
            state,
        )
        actions = self._decode(
            hidden_states, encoded_observations
        ).squeeze(0)
        return actions, state

    def initial_state(self, batch_size, device):
        shape = (1, batch_size, self.hidden_size)
        return torch.zeros(shape, device=device), torch.zeros(
            shape, device=device
        )

    def forward(self, observations):
        batch_size = observations.shape[0]
        actions, _ = self.recurrent_step(
            observations,
            torch.zeros(
                batch_size, self.n_act, device=observations.device
            ),
            torch.zeros(batch_size, 1, device=observations.device),
            self.initial_state(batch_size, observations.device),
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
            new_scales = torch.rand(
                self.n_envs, 1, device=observations.device
            ) * (self.std_max - self.std_min) + self.std_min
            self.noise_scales.copy_(
                torch.where(
                    dones.view(-1, 1), new_scales, self.noise_scales
                )
            )
        actions, state = self.recurrent_step(
            observations, prev_actions, rewards, state
        )
        if not deterministic:
            actions = actions + torch.randn_like(actions) * self.noise_scales
        return actions, state


class RecurrentCritic(nn.Module):
    """Critic_RNN from pomdp-baselines with twin FastTD3 C51 heads."""

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
        rnn_hidden_size=128,
        action_embedding_size=16,
        reward_embedding_size=16,
    ):
        super().__init__()
        del n_obs
        self.n_act = n_act
        self.observation_encoder = encoder_factory().to(device)
        observation_embedding_size = self.observation_encoder.hidden_size
        self.action_embedder = FeatureExtractor(
            n_act, action_embedding_size, device
        )
        self.reward_embedder = FeatureExtractor(
            1, reward_embedding_size, device
        )
        rnn_input_size = (
            action_embedding_size
            + observation_embedding_size
            + reward_embedding_size
        )
        self.rnn = nn.LSTM(rnn_input_size, rnn_hidden_size).to(device)
        _orthogonal_recurrent_init(self.rnn)

        # Official continuous-action shortcut: current observation plus a
        # separate embedding of the current action.
        self.current_action_embedder = FeatureExtractor(
            n_act, rnn_input_size, device
        )
        joint_size = (
            rnn_hidden_size
            + observation_embedding_size
            + rnn_input_size
        )
        q_kwargs = dict(
            n_obs=joint_size,
            n_act=0,
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
            "q_support",
            torch.linspace(v_min, v_max, num_atoms, device=device),
        )

    def _encode_observations(self, observations):
        shape = observations.shape[:-1]
        encoded = self.observation_encoder(
            observations.reshape(-1, observations.shape[-1])
        )
        return encoded.reshape(*shape, -1)

    def get_hidden_states(self, prev_actions, rewards, observations):
        encoded_observations = self._encode_observations(observations)
        inputs = torch.cat(
            (
                self.action_embedder(prev_actions),
                self.reward_embedder(rewards),
                encoded_observations,
            ),
            dim=-1,
        )
        hidden_states, _ = self.rnn(inputs)
        return hidden_states, encoded_observations

    def _joint_embeddings(
        self,
        prev_actions,
        rewards,
        observations,
        current_actions,
    ):
        hidden_states, encoded_observations = self.get_hidden_states(
            prev_actions, rewards, observations
        )
        if current_actions.shape[0] != observations.shape[0]:
            hidden_states = hidden_states[:-1]
            encoded_observations = encoded_observations[:-1]
        shortcut = torch.cat(
            (
                encoded_observations,
                self.current_action_embedder(current_actions),
            ),
            dim=-1,
        )
        return torch.cat((hidden_states, shortcut), dim=-1)

    @staticmethod
    def _flatten_joint(joint_embeddings):
        shape = joint_embeddings.shape[:-1]
        flat = joint_embeddings.reshape(-1, joint_embeddings.shape[-1])
        no_actions = flat.new_empty((flat.shape[0], 0))
        return shape, flat, no_actions

    def forward_sequence(
        self, prev_actions, rewards, observations, current_actions
    ):
        joint = self._joint_embeddings(
            prev_actions,
            rewards,
            observations,
            current_actions,
        )
        shape, flat_joint, no_actions = self._flatten_joint(joint)
        q1 = self.qnet1(flat_joint, no_actions)
        q2 = self.qnet2(flat_joint, no_actions)
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
        hidden_states, encoded_observations = self.get_hidden_states(
            prev_actions, previous_rewards, observations
        )
        shortcut = torch.cat(
            (
                encoded_observations[1:],
                self.current_action_embedder(next_actions),
            ),
            dim=-1,
        )
        joint = torch.cat((hidden_states[1:], shortcut), dim=-1)
        shape, flat_joint, no_actions = self._flatten_joint(joint)
        flat_rewards = rewards.reshape(-1)
        flat_bootstrap = bootstrap.reshape(-1)
        flat_discount = discount.reshape(-1)
        q1 = self.qnet1.projection(
            flat_joint,
            no_actions,
            flat_rewards,
            flat_bootstrap,
            flat_discount,
            self.q_support,
            self.q_support.device,
        )
        q2 = self.qnet2.projection(
            flat_joint,
            no_actions,
            flat_rewards,
            flat_bootstrap,
            flat_discount,
            self.q_support,
            self.q_support.device,
        )
        return q1.reshape(*shape, -1), q2.reshape(*shape, -1)

    def forward(self, observations, actions):
        batch_size = observations.shape[0]
        q1, q2 = self.forward_sequence(
            torch.zeros(
                1, batch_size, self.n_act, device=observations.device
            ),
            torch.zeros(1, batch_size, 1, device=observations.device),
            observations.unsqueeze(0),
            actions.unsqueeze(0),
        )
        return q1.squeeze(0), q2.squeeze(0)

    def get_value(self, probabilities):
        return torch.sum(probabilities * self.q_support, dim=-1)
