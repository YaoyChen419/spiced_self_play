import os

from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist

from tensordict import TensorDict


class SimpleReplayBuffer(nn.Module):
    def __init__(
        self,
        n_env: int,
        buffer_size: int,
        n_obs: int,
        n_act: int,
        n_critic_obs: int,
        asymmetric_obs: bool = False,
        playground_mode: bool = False,
        n_steps: int = 1,
        gamma: float = 0.99,
        device=None,
    ):
        super().__init__()

        if n_steps != 1:
            raise ValueError(
                "SPiCED FastTD3 replay buffer currently requires num_steps = 1"
            )

        self.n_env = n_env
        self.buffer_size = buffer_size
        self.n_obs = n_obs
        self.n_act = n_act
        self.n_critic_obs = n_critic_obs
        self.asymmetric_obs = asymmetric_obs
        self.playground_mode = playground_mode and asymmetric_obs
        self.n_steps = n_steps
        self.gamma = gamma
        self.device = device

        self.observations = torch.zeros(
            (buffer_size, n_obs),
            device=device,
            dtype=torch.float,
        )
        self.actions = torch.zeros(
            (buffer_size, n_act),
            device=device,
            dtype=torch.float,
        )
        self.rewards = torch.zeros(
            buffer_size,
            device=device,
            dtype=torch.float,
        )
        self.dones = torch.zeros(
            buffer_size,
            device=device,
            dtype=torch.long,
        )
        self.truncations = torch.zeros(
            buffer_size,
            device=device,
            dtype=torch.long,
        )
        self.next_observations = torch.zeros(
            (buffer_size, n_obs),
            device=device,
            dtype=torch.float,
        )

        if asymmetric_obs:
            self.critic_observations = torch.zeros(
                (buffer_size, n_critic_obs),
                device=device,
                dtype=torch.float,
            )
            self.next_critic_observations = torch.zeros(
                (buffer_size, n_critic_obs),
                device=device,
                dtype=torch.float,
            )

        self.ptr = 0
        self.size = 0

    @torch.no_grad()
    def extend(self, tensor_dict: TensorDict):
        batch_size = tensor_dict["observations"].shape[0]

        if batch_size >= self.buffer_size:
            tensor_dict = tensor_dict[-self.buffer_size :]
            batch_size = self.buffer_size

        indices = (
            torch.arange(batch_size, device=self.device) + self.ptr
        ) % self.buffer_size

        self.observations[indices] = tensor_dict["observations"]
        self.actions[indices] = tensor_dict["actions"]
        self.rewards[indices] = tensor_dict["next"]["rewards"]
        self.dones[indices] = tensor_dict["next"]["dones"]
        self.truncations[indices] = tensor_dict["next"]["truncations"]
        self.next_observations[indices] = tensor_dict["next"]["observations"]

        if self.asymmetric_obs:
            self.critic_observations[indices] = tensor_dict[
                "critic_observations"
            ]
            self.next_critic_observations[indices] = tensor_dict["next"][
                "critic_observations"
            ]

        self.ptr = (self.ptr + batch_size) % self.buffer_size
        self.size = min(self.size + batch_size, self.buffer_size)

    @torch.no_grad()
    def sample(self, batch_size: int):
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer")

        indices = torch.randint(
            0,
            self.size,
            (batch_size,),
            device=self.device,
        )

        observations = self.observations[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        dones = self.dones[indices]
        truncations = self.truncations[indices]
        next_observations = self.next_observations[indices]

        out = TensorDict(
            {
                "observations": observations,
                "actions": actions,
                "next": {
                    "rewards": rewards,
                    "dones": dones,
                    "truncations": truncations,
                    "observations": next_observations,
                    "effective_n_steps": torch.ones_like(dones),
                },
            },
            batch_size=batch_size,
            device=self.device,
        )

        if self.asymmetric_obs:
            out["critic_observations"] = self.critic_observations[indices]
            out["next"]["critic_observations"] = (
                self.next_critic_observations[indices]
            )

        return out


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, shape, device, eps=1e-2, until=None):
        """Initialize EmpiricalNormalization module.

        Args:
            shape (int or tuple of int): Shape of input values except batch axis.
            eps (float): Small value for stability.
            until (int or None): If this arg is specified, the link learns input values until the sum of batch sizes
            exceeds it.
        """
        super().__init__()
        self.eps = eps
        self.until = until
        self.device = device
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0).to(device))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long).to(device))

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, center: bool = True, update: bool = True
    ) -> torch.Tensor:
        if x.shape[1:] != self._mean.shape[1:]:
            raise ValueError(
                f"Expected input of shape (*,{self._mean.shape[1:]}), got {x.shape}"
            )

        if self.training and update:
            self.update(x)
        if center:
            return (x - self._mean) / (self._std + self.eps)
        else:
            return x / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x):
        if self.until is not None and self.count >= self.until:
            return

        if dist.is_available() and dist.is_initialized():
            # Calculate global batch size arithmetically
            local_batch_size = x.shape[0]
            world_size = dist.get_world_size()
            global_batch_size = world_size * local_batch_size

            # Calculate the stats
            x_shifted = x - self._mean
            local_sum_shifted = torch.sum(x_shifted, dim=0, keepdim=True)
            local_sum_sq_shifted = torch.sum(x_shifted.pow(2), dim=0, keepdim=True)

            # Sync the stats across all processes
            stats_to_sync = torch.cat([local_sum_shifted, local_sum_sq_shifted], dim=0)
            dist.all_reduce(stats_to_sync, op=dist.ReduceOp.SUM)
            global_sum_shifted, global_sum_sq_shifted = stats_to_sync

            # Calculate the mean and variance of the global batch
            batch_mean_shifted = global_sum_shifted / global_batch_size
            batch_var = (
                global_sum_sq_shifted / global_batch_size - batch_mean_shifted.pow(2)
            )
            batch_mean = batch_mean_shifted + self._mean

        else:
            global_batch_size = x.shape[0]
            batch_mean = torch.mean(x, dim=0, keepdim=True)
            batch_var = torch.var(x, dim=0, keepdim=True, unbiased=False)

        new_count = self.count + global_batch_size

        # Update mean
        delta = batch_mean - self._mean
        self._mean.copy_(self._mean + delta * (global_batch_size / new_count))

        # Update variance
        delta2 = batch_mean - self._mean
        m_a = self._var * self.count
        m_b = batch_var * global_batch_size
        M2 = m_a + m_b + delta2.pow(2) * (self.count * global_batch_size / new_count)
        self._var.copy_(M2 / new_count)
        self._std.copy_(self._var.sqrt())
        self.count.copy_(new_count)

    @torch.jit.unused
    def inverse(self, y):
        return y * (self._std + self.eps) + self._mean


class RewardNormalizer(nn.Module):
    def __init__(
        self,
        gamma: float,
        device: torch.device,
        g_max: float = 10.0,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.register_buffer(
            "G", torch.zeros(1, device=device)
        )  # running estimate of the discounted return
        self.register_buffer("G_r_max", torch.zeros(1, device=device))  # running-max
        self.G_rms = EmpiricalNormalization(shape=1, device=device)
        self.gamma = gamma
        self.g_max = g_max
        self.epsilon = epsilon

    def _scale_reward(self, rewards: torch.Tensor) -> torch.Tensor:
        var_denominator = self.G_rms.std[0] + self.epsilon
        min_required_denominator = self.G_r_max / self.g_max
        denominator = torch.maximum(var_denominator, min_required_denominator)

        return rewards / denominator

    def update_stats(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ):
        self.G = self.gamma * (1 - dones) * self.G + rewards
        self.G_rms.update(self.G.view(-1, 1))

        local_max = torch.max(torch.abs(self.G))

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local_max, op=dist.ReduceOp.MAX)

        self.G_r_max = max(self.G_r_max, local_max)

    def forward(self, rewards: torch.Tensor) -> torch.Tensor:
        return self._scale_reward(rewards)


class PerTaskEmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values for each task."""

    def __init__(
        self,
        num_tasks: int,
        shape: tuple,
        device: torch.device,
        eps: float = 1e-2,
        until: int = None,
    ):
        """
        Initialize PerTaskEmpiricalNormalization module.

        Args:
            num_tasks (int): The total number of tasks.
            shape (int or tuple of int): Shape of input values except batch axis.
            eps (float): Small value for stability.
            until (int or None): If specified, learns until the sum of batch sizes
                                 for a specific task exceeds this value.
        """
        super().__init__()
        if not isinstance(shape, tuple):
            shape = (shape,)
        self.num_tasks = num_tasks
        self.shape = shape
        self.eps = eps
        self.until = until
        self.device = device

        # Buffers now have a leading dimension for tasks
        self.register_buffer("_mean", torch.zeros(num_tasks, *shape).to(device))
        self.register_buffer("_var", torch.ones(num_tasks, *shape).to(device))
        self.register_buffer("_std", torch.ones(num_tasks, *shape).to(device))
        self.register_buffer(
            "count", torch.zeros(num_tasks, dtype=torch.long).to(device)
        )

    def forward(
        self, x: torch.Tensor, task_ids: torch.Tensor, center: bool = True
    ) -> torch.Tensor:
        """
        Normalize the input tensor `x` using statistics for the given `task_ids`.

        Args:
            x (torch.Tensor): Input tensor of shape [num_envs, *shape].
            task_ids (torch.Tensor): Tensor of task indices, shape [num_envs].
            center (bool): If True, center the data by subtracting the mean.
        """
        if x.shape[1:] != self.shape:
            raise ValueError(f"Expected input shape (*, {self.shape}), got {x.shape}")
        if x.shape[0] != task_ids.shape[0]:
            raise ValueError("Batch size of x and task_ids must match.")

        # Gather the stats for the tasks in the current batch
        # Reshape task_ids for broadcasting: [num_envs] -> [num_envs, 1, ...]
        view_shape = (task_ids.shape[0],) + (1,) * len(self.shape)
        task_ids_expanded = task_ids.view(view_shape).expand_as(x)

        mean = self._mean.gather(0, task_ids_expanded)
        std = self._std.gather(0, task_ids_expanded)

        if self.training:
            self.update(x, task_ids)

        if center:
            return (x - mean) / (std + self.eps)
        else:
            return x / (std + self.eps)

    @torch.jit.unused
    def update(self, x: torch.Tensor, task_ids: torch.Tensor):
        """Update running statistics for the tasks present in the batch."""
        unique_tasks = torch.unique(task_ids)

        for task_id in unique_tasks:
            if self.until is not None and self.count[task_id] >= self.until:
                continue

            # Create a mask to select data for the current task
            mask = task_ids == task_id
            x_task = x[mask]
            batch_size = x_task.shape[0]

            if batch_size == 0:
                continue

            # Update count for this task
            old_count = self.count[task_id].clone()
            new_count = old_count + batch_size

            # Update mean
            task_mean = self._mean[task_id]
            batch_mean = torch.mean(x_task, dim=0)
            delta = batch_mean - task_mean
            self._mean[task_id].copy_(task_mean + (batch_size / new_count) * delta)

            # Update variance using Chan's parallel algorithm
            if old_count > 0:
                batch_var = torch.var(x_task, dim=0, unbiased=False)
                m_a = self._var[task_id] * old_count
                m_b = batch_var * batch_size
                M2 = m_a + m_b + (delta**2) * (old_count * batch_size / new_count)
                self._var[task_id].copy_(M2 / new_count)
            else:
                # For the first batch of this task
                self._var[task_id].copy_(torch.var(x_task, dim=0, unbiased=False))

            self._std[task_id].copy_(torch.sqrt(self._var[task_id]))
            self.count[task_id].copy_(new_count)


class PerTaskRewardNormalizer(nn.Module):
    def __init__(
        self,
        num_tasks: int,
        gamma: float,
        device: torch.device,
        g_max: float = 10.0,
        epsilon: float = 1e-8,
    ):
        """
        Per-task reward normalizer, motivation comes from BRC (https://arxiv.org/abs/2505.23150v1)
        """
        super().__init__()
        self.num_tasks = num_tasks
        self.gamma = gamma
        self.g_max = g_max
        self.epsilon = epsilon
        self.device = device

        # Per-task running estimate of the discounted return
        self.register_buffer("G", torch.zeros(num_tasks, device=device))
        # Per-task running-max of the discounted return
        self.register_buffer("G_r_max", torch.zeros(num_tasks, device=device))
        # Use the new per-task normalizer for the statistics of G
        self.G_rms = PerTaskEmpiricalNormalization(
            num_tasks=num_tasks, shape=(1,), device=device
        )

    def _scale_reward(
        self, rewards: torch.Tensor, task_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Scales rewards using per-task statistics.

        Args:
            rewards (torch.Tensor): Reward tensor, shape [num_envs].
            task_ids (torch.Tensor): Task indices, shape [num_envs].
        """
        # Gather stats for the tasks in the batch
        std_for_batch = self.G_rms._std.gather(0, task_ids.unsqueeze(-1)).squeeze(-1)
        g_r_max_for_batch = self.G_r_max.gather(0, task_ids)

        var_denominator = std_for_batch + self.epsilon
        min_required_denominator = g_r_max_for_batch / self.g_max
        denominator = torch.maximum(var_denominator, min_required_denominator)

        # Add a small epsilon to the final denominator to prevent division by zero
        # in case g_r_max is also zero.
        return rewards / (denominator + self.epsilon)

    def update_stats(
        self, rewards: torch.Tensor, dones: torch.Tensor, task_ids: torch.Tensor
    ):
        """
        Updates the running discounted return and its statistics for each task.

        Args:
            rewards (torch.Tensor): Reward tensor, shape [num_envs].
            dones (torch.Tensor): Done tensor, shape [num_envs].
            task_ids (torch.Tensor): Task indices, shape [num_envs].
        """
        if not (rewards.shape == dones.shape == task_ids.shape):
            raise ValueError("rewards, dones, and task_ids must have the same shape.")

        # === Update G (running discounted return) ===
        # Gather the previous G values for the tasks in the batch
        prev_G = self.G.gather(0, task_ids)
        # Update G for each environment based on its own reward and done signal
        new_G = self.gamma * (1 - dones.float()) * prev_G + rewards
        # Scatter the updated G values back to the main buffer
        self.G.scatter_(0, task_ids, new_G)

        # === Update G_rms (statistics of G) ===
        # The update function handles the per-task logic internally
        self.G_rms.update(new_G.unsqueeze(-1), task_ids)

        # === Update G_r_max (running max of |G|) ===
        prev_G_r_max = self.G_r_max.gather(0, task_ids)
        # Update the max for each environment
        updated_G_r_max = torch.maximum(prev_G_r_max, torch.abs(new_G))
        # Scatter the new maxes back to the main buffer
        self.G_r_max.scatter_(0, task_ids, updated_G_r_max)

    def forward(self, rewards: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        """
        Normalizes rewards. During training, it also updates the running statistics.

        Args:
            rewards (torch.Tensor): Reward tensor, shape [num_envs].
            task_ids (torch.Tensor): Task indices, shape [num_envs].
        """
        return self._scale_reward(rewards, task_ids)


def cpu_state(sd):
    # detach & move to host without locking the compute stream
    return {k: v.detach().to("cpu", non_blocking=True) for k, v in sd.items()}


def save_params(
    global_step,
    actor,
    qnet,
    qnet_target,
    obs_normalizer,
    critic_obs_normalizer,
    args,
    save_path,
):
    """Save model parameters and training configuration to disk."""

    def get_ddp_state_dict(model):
        """Get state dict from model, handling DDP wrapper if present."""
        if hasattr(model, "module"):
            return model.module.state_dict()
        return model.state_dict()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_dict = {
        "actor_state_dict": cpu_state(get_ddp_state_dict(actor)),
        "qnet_state_dict": cpu_state(get_ddp_state_dict(qnet)),
        "qnet_target_state_dict": cpu_state(get_ddp_state_dict(qnet_target)),
        "obs_normalizer_state": (
            cpu_state(obs_normalizer.state_dict())
            if hasattr(obs_normalizer, "state_dict")
            else None
        ),
        "critic_obs_normalizer_state": (
            cpu_state(critic_obs_normalizer.state_dict())
            if hasattr(critic_obs_normalizer, "state_dict")
            else None
        ),
        "args": dict(args) if isinstance(args, dict) else vars(args),  # Save all arguments
        "global_step": global_step,
    }
    torch.save(save_dict, save_path, _use_new_zipfile_serialization=True)
    print(f"Saved parameters and configuration to {save_path}")


def get_ddp_state_dict(model):
    """Get state dict from model, handling DDP wrapper if present."""
    if hasattr(model, "module"):
        return model.module.state_dict()
    return model.state_dict()


def load_ddp_state_dict(model, state_dict):
    """Load state dict into model, handling DDP wrapper if present."""
    if hasattr(model, "module"):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)


@torch.no_grad()
def mark_step():
    # call this once per iteration *before* any compiled function
    torch.compiler.cudagraph_mark_step_begin()