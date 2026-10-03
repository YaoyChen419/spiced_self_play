import os
import sys

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"
else:
    os.environ["MUJOCO_GL"] = "glfw"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"

import random
import time
import math
from collections import defaultdict
from types import SimpleNamespace

import tqdm
import wandb
import numpy as np

try:
    # Required for avoiding IsaacGym import error
    import isaacgym
except ImportError:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler

from tensordict import TensorDict

import pufferlib
from pufferlib.fast_td3_utils import (
    EmpiricalNormalization,
    RewardNormalizer,
    PerTaskRewardNormalizer,
    SimpleReplayBuffer,
    save_params,
    mark_step,
)

torch.set_float32_matmul_precision("high")

try:
    import jax.numpy as jnp
except ImportError:
    pass


def _collect_environment_stats(stats, infos):
    """Collect PufferDrive info using the native PuffeRL aggregation semantics."""
    for info in infos:
        for key, value in pufferlib.unroll_nested_dict(info):
            if key.startswith("_"):
                continue
            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, (list, tuple)):
                stats[key].extend(value)
                continue
            stats[key].append(value)


def _mean_environment_stats(stats):
    logs = {}
    for key, values in stats.items():
        try:
            logs[f"environment/{key}"] = np.mean(values)
        except (TypeError, ValueError):
            pass
    stats.clear()
    return logs


class PufferDriveEnv:
    """Expose SPiCED's PufferLib vector API through FastTD3's env contract."""

    def __init__(self, vecenv, device, seed):
        self.vecenv = vecenv
        self.device = device
        self.seed = seed
        self.num_envs = vecenv.observation_space.shape[0]
        self.num_obs = vecenv.single_observation_space.shape[0]
        if vecenv.single_action_space.dtype != np.float32:
            raise pufferlib.APIUsageError("FastTD3 requires a continuous float32 action space")
        self.num_actions = vecenv.single_action_space.shape[0]
        self.asymmetric_obs = False
        self.agents_per_worker = vecenv.num_agents // getattr(vecenv, "num_workers", 1)
        self.agent_dead = np.zeros(vecenv.num_agents, dtype=bool)
        self.pending = {}

    def reset(self):
        self.vecenv.async_reset(self.seed)
        observations, _, _, _, _, agent_ids, _ = self.vecenv.recv()
        self.agent_ids = agent_ids.copy()
        self.pending.clear()
        self.agent_dead.fill(False)
        self.has_dones = False
        self.observations = torch.as_tensor(observations.copy(), device=self.device, dtype=torch.float)
        return self.observations

    def step(self, actions):
        actions = torch.clamp(actions.float(), -1.0, 1.0)
        for start in range(0, self.num_envs, self.agents_per_worker):
            end = start + self.agents_per_worker
            self.pending[int(self.agent_ids[start])] = (
                self.agent_ids[start:end], self.observations[start:end], actions[start:end]
            )
        self.vecenv.send(actions.detach().cpu().numpy())
        observations, rewards, terminals, truncations, infos, agent_ids, masks = self.vecenv.recv()

        previous_obs = torch.zeros_like(self.observations)
        previous_actions = torch.zeros_like(actions)
        transition_ready = np.zeros(self.num_envs, dtype=bool)
        for start in range(0, self.num_envs, self.agents_per_worker):
            end = start + self.agents_per_worker
            worker_previous = self.pending.pop(int(agent_ids[start]), None)
            if worker_previous is None:
                continue  # Initial observations have no preceding action.
            if not np.array_equal(worker_previous[0], agent_ids[start:end]):
                raise RuntimeError("PufferDrive agent IDs changed within a worker")
            previous_obs[start:end] = worker_previous[1]
            previous_actions[start:end] = worker_previous[2]
            transition_ready[start:end] = True
        previous = (
            (agent_ids.copy(), previous_obs, previous_actions)
            if transition_ready.any() else None
        )
        received_mask = masks & transition_ready
        received_count = int(np.count_nonzero(received_mask))
        valid_cpu = (~self.agent_dead[agent_ids]) & received_mask
        received_mask = torch.as_tensor(
            received_mask, device=self.device, dtype=torch.bool
        )
        valid = torch.as_tensor(valid_cpu, device=self.device, dtype=torch.bool)

        final_rows, final_observations = [], []
        transition_terminals = terminals.copy()
        covered = np.zeros(self.num_envs, dtype=bool)
        transitions = []
        for info in infos:
            if "_fasttd3_transition" in info:
                transitions.append(info["_fasttd3_transition"])

        for transition in transitions:
            start = transition.get("offset", 0)
            end = start + transition["count"]
            transition_terminals[start:end] = transition["terminals"]
            indices = start + transition["indices"]
            if indices.size:
                final_rows.append(indices)
                final_observations.append(transition["observations"])
            covered[start:end] = True

        if np.any(truncations & transition_ready & ~covered):
            raise RuntimeError(
                "Drive truncated without providing FastTD3 final observations"
            )
        self.has_dones = bool(transition_terminals.any() or truncations.any())

        # Keep validity on the host, where the environment already produces done flags.
        self.agent_dead[agent_ids] |= terminals & ~truncations
        self.agent_dead[agent_ids] &= ~truncations
        # recv buffers stay unchanged until the next send; CUDA conversion is blocking.
        observations = torch.as_tensor(observations, device=self.device, dtype=torch.float)
        if observations.device.type == "cpu":
            observations = observations.clone()
        raw_observations = observations
        if final_rows:
            raw_observations = observations.clone()
            raw_observations.index_copy_(
                0,
                torch.as_tensor(np.concatenate(final_rows), device=self.device),
                torch.as_tensor(
                    np.concatenate(final_observations), device=self.device, dtype=torch.float
                ),
            )
        rewards = torch.as_tensor(rewards.copy(), device=self.device, dtype=torch.float)
        rewards = torch.clamp(rewards, -1, 1)
        terminals = torch.as_tensor(transition_terminals, device=self.device, dtype=torch.bool)
        truncations = torch.as_tensor(truncations.copy(), device=self.device, dtype=torch.bool)
        dones = terminals | truncations
        self.agent_ids = agent_ids.copy()
        self.observations = observations
        info = {
            "time_outs": truncations & ~terminals,
            "observations": {"raw": {"obs": raw_observations}},
            "transition": previous,
            "mask": received_mask,
            "received_count": received_count,
            "valid": valid,
            "valid_cpu": valid_cpu,
            "environment_infos": infos,
        }
        return observations, rewards, dones, info


def train(env_name, args=None, vecenv=None, policy=None, logger=None):
    from pufferlib.pufferl import (
        NoLogger,
        NeptuneLogger,
        WandbLogger,
        load_config,
        load_env,
        resolve_fasttd3_checkpoint_path,
    )

    full_args = args or load_config(env_name)
    full_args["env"]["capture_final_observations"] = True

    train_config = full_args["train"]
    requested_device = str(train_config["device"])
    cuda = requested_device.startswith("cuda")
    device_rank = int(requested_device.split(":", 1)[1]) if ":" in requested_device else 0

    args = SimpleNamespace(**train_config)
    args.env_name = env_name
    args.cuda = cuda
    args.device_rank = device_rank
    args.checkpoint_path = (
        resolve_fasttd3_checkpoint_path(full_args, env_name)
        if full_args.get("load_model_path") or full_args.get("load_id") else None
    )
    args.eval_interval_agent_steps = full_args.get("eval", {}).get("eval_interval_agent_steps", 0)
    args.save_interval = train_config["checkpoint_interval"]

    if logger is None:
        if full_args.get("neptune"):
            logger = NeptuneLogger(full_args)
        elif full_args.get("wandb"):
            logger = WandbLogger(full_args)
        else:
            logger = NoLogger(full_args)

    vecenv = vecenv or load_env(env_name, full_args)
    args.num_envs = vecenv.observation_space.shape[0]
    args.save_interval_agent_steps = args.save_interval * args.num_envs
    total_agent_timesteps = args.total_timesteps
    args.total_timesteps = math.ceil(total_agent_timesteps / args.num_envs)

    print(args)

    amp_enabled = args.amp and args.cuda and torch.cuda.is_available()
    amp_device_type = (
        "cuda"
        if args.cuda and torch.cuda.is_available()
        else "mps" if args.cuda and torch.backends.mps.is_available() else "cpu"
    )
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    scaler = GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if not args.cuda:
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.device_rank}")
        elif torch.backends.mps.is_available():
            device = torch.device(f"mps:{args.device_rank}")
        else:
            raise ValueError("No GPU available")
    print(f"Using device: {device}")

    env_type = "puffer_drive"
    envs = PufferDriveEnv(vecenv, device, args.seed)

    n_act = envs.num_actions
    n_obs = envs.num_obs if type(envs.num_obs) == int else envs.num_obs[0]
    if envs.asymmetric_obs:
        n_critic_obs = (
            envs.num_privileged_obs
            if type(envs.num_privileged_obs) == int
            else envs.num_privileged_obs[0]
        )
    else:
        n_critic_obs = n_obs
    action_low, action_high = -1.0, 1.0

    if args.obs_normalization:
        obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)
        critic_obs_normalizer = EmpiricalNormalization(
            shape=n_critic_obs, device=device
        )
    else:
        obs_normalizer = nn.Identity()
        critic_obs_normalizer = nn.Identity()

    if args.reward_normalization:
        if env_type in ["mtbench"]:
            reward_normalizer = PerTaskRewardNormalizer(
                num_tasks=envs.num_tasks,
                gamma=args.gamma,
                device=device,
                g_max=min(abs(args.v_min), abs(args.v_max)),
            )
        else:
            reward_normalizer = RewardNormalizer(
                gamma=args.gamma,
                device=device,
                g_max=min(abs(args.v_min), abs(args.v_max)),
                num_envs=vecenv.num_agents,
            )
    else:
        reward_normalizer = nn.Identity()

    actor_kwargs = {
        "n_obs": n_obs,
        "n_act": n_act,
        "num_envs": args.num_envs,
        "device": device,
        "init_scale": args.init_scale,
        "hidden_dim": args.actor_hidden_dim,
        "std_min": args.std_min,
        "std_max": args.std_max,
    }
    critic_kwargs = {
        "n_obs": n_critic_obs,
        "n_act": n_act,
        "num_atoms": args.num_atoms,
        "v_min": args.v_min,
        "v_max": args.v_max,
        "hidden_dim": args.critic_hidden_dim,
        "device": device,
    }

    if env_type == "mtbench":
        actor_kwargs["n_obs"] = n_obs - envs.num_tasks + args.task_embedding_dim
        critic_kwargs["n_obs"] = n_critic_obs - envs.num_tasks + args.task_embedding_dim
        actor_kwargs["num_tasks"] = envs.num_tasks
        actor_kwargs["task_embedding_dim"] = args.task_embedding_dim
        critic_kwargs["num_tasks"] = envs.num_tasks
        critic_kwargs["task_embedding_dim"] = args.task_embedding_dim

    if args.agent == "fasttd3":
        if env_type in ["mtbench"]:
            from pufferlib.fast_td3 import MultiTaskActor, MultiTaskCritic

            actor_cls = MultiTaskActor
            critic_cls = MultiTaskCritic
        else:
            from pufferlib.fast_td3 import Actor, Critic

            actor_cls = Actor
            critic_cls = Critic

        actor_kwargs.update(
            {
                "sim_type": args.sim_type,
                "sim_dimension": args.sim_dimension,
                "seq_len": args.actor_seq_len,
            }
        )
        critic_kwargs.update(
            {
                "sim_type": args.sim_type,
                "sim_dimension": args.sim_dimension,
                "seq_len": args.critic_seq_len,
            }
        )

        print("Using FastTD3")
    elif args.agent == "fasttd3_simbav2":
        if args.sim_type:
            raise ValueError("SimNorm options are only supported with agent='fasttd3'")

        if env_type in ["mtbench"]:
            from fast_td3_simbav2 import MultiTaskActor, MultiTaskCritic

            actor_cls = MultiTaskActor
            critic_cls = MultiTaskCritic
        else:
            from fast_td3_simbav2 import Actor, Critic

            actor_cls = Actor
            critic_cls = Critic

        print("Using FastTD3 + SimbaV2")
        actor_kwargs.pop("init_scale")
        actor_kwargs.update(
            {
                "scaler_init": math.sqrt(2.0 / args.actor_hidden_dim),
                "scaler_scale": math.sqrt(2.0 / args.actor_hidden_dim),
                "alpha_init": 1.0 / (args.actor_num_blocks + 1),
                "alpha_scale": 1.0 / math.sqrt(args.actor_hidden_dim),
                "expansion": 4,
                "c_shift": 3.0,
                "num_blocks": args.actor_num_blocks,
            }
        )
        critic_kwargs.update(
            {
                "scaler_init": math.sqrt(2.0 / args.critic_hidden_dim),
                "scaler_scale": math.sqrt(2.0 / args.critic_hidden_dim),
                "alpha_init": 1.0 / (args.critic_num_blocks + 1),
                "alpha_scale": 1.0 / math.sqrt(args.critic_hidden_dim),
                "num_blocks": args.critic_num_blocks,
                "expansion": 4,
                "c_shift": 3.0,
            }
        )
    else:
        raise ValueError(f"Agent {args.agent} not supported")

    if isinstance(policy, tuple):
        policy, obs_normalizer = policy
        obs_normalizer = obs_normalizer.to(device)
    if policy is not None and not isinstance(policy, actor_cls):
        raise pufferlib.APIUsageError("FastTD3 training requires a native FastTD3 actor")
    actor = policy.to(device) if policy is not None else actor_cls(**actor_kwargs)

    if env_type in ["mtbench"]:
        # Python 3.8 doesn't support 'from_module' in tensordict
        policy = actor.explore
    else:
        from tensordict import from_module

        actor_detach = actor_cls(**actor_kwargs)
        # Copy params to actor_detach without grad
        from_module(actor).data.to_module(actor_detach)
        policy = actor_detach.explore

    qnet = critic_cls(**critic_kwargs)
    qnet_target = critic_cls(**critic_kwargs)
    qnet_target.load_state_dict(qnet.state_dict())

    q_optimizer = optim.AdamW(
        list(qnet.parameters()),
        lr=torch.tensor(args.critic_learning_rate, device=device),
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    actor_optimizer = optim.AdamW(
        list(actor.parameters()),
        lr=torch.tensor(args.actor_learning_rate, device=device),
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )

    # Add learning rate schedulers
    q_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        q_optimizer,
        T_max=args.total_timesteps,
        eta_min=torch.tensor(args.critic_learning_rate_end, device=device),
    )
    actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        actor_optimizer,
        T_max=args.total_timesteps,
        eta_min=torch.tensor(args.actor_learning_rate_end, device=device),
    )

    # Each PufferLib worker owns continuous trajectories for its agent IDs.
    num_workers = getattr(vecenv, "num_workers", 1)
    agents_per_worker = vecenv.num_agents // num_workers
    replay_buffers = [
        SimpleReplayBuffer(
            n_env=agents_per_worker,
            buffer_size=args.buffer_size,
            n_obs=n_obs,
            n_act=n_act,
            n_critic_obs=n_critic_obs,
            asymmetric_obs=envs.asymmetric_obs,
            playground_mode=env_type == "mujoco_playground",
            n_steps=args.num_steps,
            gamma=args.gamma,
            device=device,
            valid_device="cpu",
        )
        for _ in range(num_workers)
    ]

    samples_per_agent = max(1, args.batch_size // args.num_envs)
    replay_batch = None
    if args.num_steps == 1 and not envs.asymmetric_obs:
        capacity = args.num_envs * samples_per_agent
        replay_batch = TensorDict({
            "observations": torch.empty(capacity, n_obs, device=device),
            "actions": torch.empty(capacity, n_act, device=device),
            "next": {
                "observations": torch.empty(capacity, n_obs, device=device),
                "rewards": torch.empty(capacity, device=device),
                "dones": torch.empty(capacity, device=device, dtype=torch.long),
                "truncations": torch.empty(capacity, device=device, dtype=torch.long),
                "effective_n_steps": torch.empty(capacity, device=device, dtype=torch.long),
            },
        }, batch_size=[capacity], device=device)

    use_cuda_graphs = (
        args.compile and args.compile_mode == "reduce-overhead" and device.type == "cuda"
    )
    full_batch_size = args.num_envs * samples_per_agent

    policy_noise = args.policy_noise
    noise_clip = args.noise_clip

    def evaluate():
        from pufferlib.ocean.benchmark.evaluator import Evaluator

        evaluator = Evaluator(full_args, logger)

        if full_args["eval"]["human_replay_eval"]:
            evaluator.hr_env = load_env("puffer_drive", evaluator.hr_eval_config)
            try:
                evaluator.rollout(
                    actor,
                    mode="human_replay",
                    obs_normalizer=obs_normalizer,
                )
            except Exception as error:
                print(f"Render failed (non-fatal): {error}")
            evaluator.hr_env.driver_env.stop_recorder(0)
            evaluator.hr_env.close()
            evaluator.log_videos(eval_mode="human_replay", epoch=global_step)

        if full_args["eval"]["self_play_eval"]:
            evaluator.sp_env = load_env("puffer_drive", evaluator.sp_eval_config)
            try:
                evaluator.rollout(
                    actor,
                    mode="self_play",
                    obs_normalizer=obs_normalizer,
                )
            except Exception as error:
                print(f"Render failed (non-fatal): {error}")
            evaluator.sp_env.driver_env.stop_recorder(0)
            evaluator.sp_env.close()
            evaluator.log_videos(eval_mode="self_play", epoch=global_step)

        return evaluator.collect_stats()

    def update_main(data):
        logs_dict = {}
        with autocast(
            device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            observations = data["observations"]
            next_observations = data["next"]["observations"]
            if envs.asymmetric_obs:
                critic_observations = data["critic_observations"]
                next_critic_observations = data["next"]["critic_observations"]
            else:
                critic_observations = observations
                next_critic_observations = next_observations
            actions = data["actions"]
            rewards = data["next"]["rewards"]
            dones = data["next"]["dones"].bool()
            truncations = data["next"]["truncations"].bool()
            if args.disable_bootstrap:
                bootstrap = (~dones).float()
            else:
                bootstrap = (truncations | ~dones).float()

            clipped_noise = torch.randn_like(actions)
            clipped_noise = clipped_noise.mul(policy_noise).clamp(
                -noise_clip, noise_clip
            )

            with torch.no_grad():
                next_state_actions = (actor(next_observations) + clipped_noise).clamp(
                    action_low, action_high
                )
                discount = args.gamma ** data["next"]["effective_n_steps"]
                qf1_next_target_projected, qf2_next_target_projected = (
                    qnet_target.projection(
                        next_critic_observations,
                        next_state_actions,
                        rewards,
                        bootstrap,
                        discount,
                    )
                )
                qf1_next_target_value = qnet_target.get_value(qf1_next_target_projected)
                qf2_next_target_value = qnet_target.get_value(qf2_next_target_projected)
                if args.use_cdq:
                    qf_next_target_dist = torch.where(
                        qf1_next_target_value.unsqueeze(1)
                        < qf2_next_target_value.unsqueeze(1),
                        qf1_next_target_projected,
                        qf2_next_target_projected,
                    )
                    qf1_next_target_dist = qf2_next_target_dist = qf_next_target_dist
                else:
                    qf1_next_target_dist, qf2_next_target_dist = (
                        qf1_next_target_projected,
                        qf2_next_target_projected,
                    )

            qf1, qf2 = qnet(critic_observations, actions)
            qf1_loss = -torch.sum(
                qf1_next_target_dist * F.log_softmax(qf1, dim=1), dim=1
            ).mean()
            qf2_loss = -torch.sum(
                qf2_next_target_dist * F.log_softmax(qf2, dim=1), dim=1
            ).mean()
            qf_loss = qf1_loss + qf2_loss

        q_optimizer.zero_grad(set_to_none=True)
        scaler.scale(qf_loss).backward()
        scaler.unscale_(q_optimizer)

        if args.use_grad_norm_clipping:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                qnet.parameters(),
                max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
            )
        else:
            critic_grad_norm = torch.tensor(0.0, device=device)
        scaler.step(q_optimizer)
        scaler.update()

        logs_dict["critic_grad_norm"] = critic_grad_norm.detach()
        logs_dict["qf_loss"] = qf_loss.detach()
        logs_dict["qf_max"] = qf1_next_target_value.max().detach()
        logs_dict["qf_min"] = qf1_next_target_value.min().detach()
        return logs_dict

    def update_pol(data):
        logs_dict = {}
        with autocast(
            device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            critic_observations = (
                data["critic_observations"]
                if envs.asymmetric_obs
                else data["observations"]
            )

            qf1, qf2 = qnet(critic_observations, actor(data["observations"]))
            qf1_value = qnet.get_value(F.softmax(qf1, dim=1))
            qf2_value = qnet.get_value(F.softmax(qf2, dim=1))
            if args.use_cdq:
                qf_value = torch.minimum(qf1_value, qf2_value)
            else:
                qf_value = (qf1_value + qf2_value) / 2.0
            actor_loss = -qf_value.mean()

        actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward()
        scaler.unscale_(actor_optimizer)
        if args.use_grad_norm_clipping:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                actor.parameters(),
                max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
            )
        else:
            actor_grad_norm = torch.tensor(0.0, device=device)
        scaler.step(actor_optimizer)
        scaler.update()
        logs_dict["actor_grad_norm"] = actor_grad_norm.detach()
        logs_dict["actor_loss"] = actor_loss.detach()
        return logs_dict

    @torch.no_grad()
    def soft_update(src, tgt, tau: float):
        src_ps = [p.data for p in src.parameters()]
        tgt_ps = [p.data for p in tgt.parameters()]

        torch._foreach_mul_(tgt_ps, 1.0 - tau)
        torch._foreach_add_(tgt_ps, src_ps, alpha=tau)

    if args.compile:
        # Skip native projection frames while allowing target MLP forwards to compile.
        for module in (qnet_target, qnet_target.qnet1, qnet_target.qnet2):
            projection = torch.compiler.disable(
                module.projection.__func__, recursive=False
            )
            module.projection = projection.__get__(module, type(module))
        # Native FastTD3 graphs apply only to unchanged, full replay batches.
        if use_cuda_graphs:
            graph_update_main = torch.compile(update_main, mode="reduce-overhead", dynamic=False)
            graph_update_pol = torch.compile(update_pol, mode="reduce-overhead", dynamic=False)
        compile_mode = "default" if use_cuda_graphs else args.compile_mode
        update_main = torch.compile(update_main, mode=compile_mode, dynamic=True)
        update_pol = torch.compile(update_pol, mode=compile_mode, dynamic=True)
        policy = torch.compile(
            policy, mode="reduce-overhead" if use_cuda_graphs else None, dynamic=False
        )
        normalize_obs = torch.compile(obs_normalizer.forward, mode=None, dynamic=True)
        normalize_critic_obs = torch.compile(critic_obs_normalizer.forward, mode=None, dynamic=True)
        if args.reward_normalization:
            update_stats = torch.compile(reward_normalizer.update_stats, mode=None, dynamic=True)
        normalize_reward = torch.compile(reward_normalizer.forward, mode=None, dynamic=True)
    else:
        normalize_obs = obs_normalizer.forward
        normalize_critic_obs = critic_obs_normalizer.forward
        if args.reward_normalization:
            update_stats = reward_normalizer.update_stats
        normalize_reward = reward_normalizer.forward

    if envs.asymmetric_obs:
        obs, critic_obs = envs.reset_with_critic_obs()
        critic_obs = torch.as_tensor(critic_obs, device=device, dtype=torch.float)
    else:
        obs = envs.reset()
    noise_scales_by_id = (
        torch.rand(vecenv.num_agents, 1, device=device)
        * (actor_detach.std_max - actor_detach.std_min)
        + actor_detach.std_min
    )
    if args.checkpoint_path:
        # Load checkpoint if specified
        torch_checkpoint = torch.load(
            f"{args.checkpoint_path}", map_location=device, weights_only=False
        )
        actor.load_state_dict(torch_checkpoint["actor_state_dict"])
        obs_normalizer.load_state_dict(torch_checkpoint["obs_normalizer_state"])
        critic_obs_normalizer.load_state_dict(
            torch_checkpoint["critic_obs_normalizer_state"]
        )
        qnet.load_state_dict(torch_checkpoint["qnet_state_dict"])
        qnet_target.load_state_dict(torch_checkpoint["qnet_target_state_dict"])
        global_step = torch_checkpoint["global_step"]
        agent_steps = torch_checkpoint.get(
            "agent_steps", global_step * args.num_envs
        )
    else:
        global_step = 0
        agent_steps = 0

    next_eval_step = (
        (agent_steps // args.eval_interval_agent_steps + 1) * args.eval_interval_agent_steps
        if args.eval_interval_agent_steps > 0 else None
    )
    next_checkpoint_step = (
        (agent_steps // args.save_interval_agent_steps + 1) * args.save_interval_agent_steps
        if args.save_interval_agent_steps > 0 else None
    )
    dones = None
    pbar = tqdm.tqdm(total=args.total_timesteps, initial=global_step)
    run_start_time = time.time()
    start_time = None
    last_log_time = run_start_time
    last_log_agent_steps = agent_steps
    interval_received = 0
    interval_valid = 0
    last_eval_agent_steps = None
    environment_stats = defaultdict(list)
    desc = ""
    all_logs = []
    model_dir = os.path.join(args.data_dir, f"{env_name}_{logger.run_id}")

    def checkpoint_path(step):
        return os.path.join(model_dir, f"model_{env_name}_{step:06d}.pt")

    def save_checkpoint():
        model_path = checkpoint_path(global_step)
        save_params(
            global_step, actor, qnet, qnet_target,
            obs_normalizer, critic_obs_normalizer, args, model_path,
            agent_steps=agent_steps, full_args=full_args,
        )
        # Reuse SPiCED's periodic checkpoint artifact upload.
        if isinstance(logger, WandbLogger):
            artifact = wandb.Artifact(
                f"checkpoint-{logger.run_id}-epoch{global_step:06d}",
                type="checkpoint",
                metadata={"epoch": global_step, "global_step": agent_steps},
            )
            artifact.add_file(model_path)
            logger.wandb.run.log_artifact(artifact)
        if full_args["eval"]["wosac_realism_eval"]:
            pufferlib.utils.run_wosac_eval_in_subprocess(
                dict(**train_config, env=env_name, eval=full_args["eval"]),
                logger, agent_steps, full_args=full_args,
            )
        return model_path

    while agent_steps < total_agent_timesteps:
        mark_step()
        logs_dict = {}
        if (
            start_time is None
            and global_step >= args.measure_burnin + args.learning_starts
        ):
            start_time = time.time()
            last_log_time = start_time
            last_log_agent_steps = agent_steps

        with torch.no_grad(), autocast(
            device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            current_ids = torch.as_tensor(envs.agent_ids, device=device, dtype=torch.long)
            if args.obs_normalization:
                active_indices = np.flatnonzero(~envs.agent_dead[envs.agent_ids])
                if active_indices.size == obs.shape[0]:
                    obs_normalizer.update(obs)
                elif active_indices.size:
                    active_indices = torch.as_tensor(active_indices, device=device)
                    obs_normalizer.update(obs[active_indices])
                norm_obs = normalize_obs(obs, update=False)
            else:
                norm_obs = normalize_obs(obs)
            actor_detach.noise_scales.copy_(noise_scales_by_id[current_ids])
            if args.agent == "fasttd3":
                actions = policy(obs=norm_obs, dones=dones, has_dones=envs.has_dones)
            else:
                actions = policy(obs=norm_obs, dones=dones)
            # Async replay retains actions across iterations; graph outputs cannot be borrowed.
            if use_cuda_graphs:
                actions = actions.clone()
            noise_scales_by_id[current_ids] = actor_detach.noise_scales

        next_obs, rewards, dones, infos = envs.step(actions.float())
        previous = infos["transition"]
        if previous is None:
            obs = next_obs
            continue
        truncations = infos["time_outs"]
        received = infos["received_count"]
        valid = int(np.count_nonzero(infos["valid_cpu"]))
        agent_steps += received
        interval_received += received
        interval_valid += valid
        _collect_environment_stats(environment_stats, infos["environment_infos"])

        if args.reward_normalization and valid:
            if env_type == "mtbench":
                task_ids_one_hot = obs[..., -envs.num_tasks :]
                task_indices = torch.argmax(task_ids_one_hot, dim=1)
                update_stats(rewards, dones.float(), task_ids=task_indices)
            else:
                reward_ids = torch.as_tensor(previous[0], device=device, dtype=torch.long)
                if valid == args.num_envs:
                    update_stats(rewards, dones.float(), env_ids=reward_ids)
                else:
                    rows = torch.as_tensor(np.flatnonzero(infos["valid_cpu"]), device=device)
                    update_stats(rewards[rows], dones[rows].float(), env_ids=reward_ids[rows])

        if envs.asymmetric_obs:
            next_critic_obs = infos["observations"]["critic"]
        # Compute 'true' next_obs and next_critic_obs for saving
        # raw preserves pre-removal/pre-reset observations for completed transitions.
        true_next_obs = infos["observations"]["raw"]["obs"]
        if envs.asymmetric_obs:
            true_next_critic_obs = torch.where(
                dones[:, None] > 0,
                infos["observations"]["raw"]["critic_obs"],
                next_critic_obs,
            )

        transition = TensorDict(
            {
                "observations": previous[1],
                "actions": previous[2],
                "valid": infos["valid"],
                "next": {
                    "observations": true_next_obs,
                    "rewards": torch.as_tensor(
                        rewards, device=device, dtype=torch.float
                    ),
                    "truncations": truncations.long(),
                    "dones": dones.long(),
                },
            },
            batch_size=(envs.num_envs,),
            device=device,
        )
        if envs.asymmetric_obs:
            transition["critic_observations"] = critic_obs
            transition["next"]["critic_observations"] = true_next_critic_obs
        # Match SPiCED's handling of invalid/dead-agent transitions.
        transition["next", "rewards"] = transition["next", "rewards"] * infos["valid"]
        transition["next", "dones"] = transition["next", "dones"].masked_fill(
            ~infos["valid"], 1
        )
        grouped_ids = previous[0].reshape(-1, agents_per_worker)
        worker_ids = grouped_ids[:, 0] // agents_per_worker
        expected_ids = worker_ids[:, None] * agents_per_worker + np.arange(agents_per_worker)
        if not np.array_equal(grouped_ids, expected_ids):
            raise RuntimeError("Replay requires complete, ordered PufferLib worker agent IDs")
        for worker_id, worker_transition, worker_valid in zip(
            worker_ids, transition.split(agents_per_worker),
            infos["valid_cpu"].reshape(-1, agents_per_worker),
        ):
            replay_buffers[worker_id].extend(
                worker_transition, valid=torch.from_numpy(worker_valid)
            )

        obs = next_obs
        if envs.asymmetric_obs:
            critic_obs = next_critic_obs

        if global_step > args.learning_starts:
            ready_buffers = [
                replay_buffers[worker_id] for worker_id in worker_ids
                if replay_buffers[worker_id].ptr >= args.num_steps
            ]
            for i in range(args.num_updates):
                if not ready_buffers:
                    break
                if replay_batch is not None:
                    offset = 0
                    for rb in ready_buffers:
                        end = offset + rb.n_env * samples_per_agent
                        sampled = rb.sample(
                            samples_per_agent, out=replay_batch[offset:end]
                        )
                        offset += sampled.numel()
                    data = replay_batch[:offset]
                else:
                    data = torch.cat([
                        rb.sample(samples_per_agent) for rb in ready_buffers
                    ], dim=0)
                    sample_valid = data.pop("_valid")
                    sample_indices = sample_valid.nonzero(as_tuple=True)[0]
                    data = data[sample_indices.to(device, non_blocking=True)]
                if data.numel() == 0:
                    continue
                data["observations"] = normalize_obs(data["observations"])
                data["next"]["observations"] = normalize_obs(
                    data["next"]["observations"]
                )
                if envs.asymmetric_obs:
                    data["critic_observations"] = normalize_critic_obs(
                        data["critic_observations"]
                    )
                    data["next"]["critic_observations"] = normalize_critic_obs(
                        data["next"]["critic_observations"]
                    )
                raw_rewards = data["next"]["rewards"]
                if env_type in ["mtbench"] and args.reward_normalization:
                    # Multi-task reward normalization
                    task_ids_one_hot = data["observations"][..., -envs.num_tasks :]
                    task_indices = torch.argmax(task_ids_one_hot, dim=1)
                    data["next"]["rewards"] = normalize_reward(
                        raw_rewards, task_ids=task_indices
                    )
                else:
                    data["next"]["rewards"] = normalize_reward(raw_rewards)

                graph_update = use_cuda_graphs and data.numel() == full_batch_size
                if graph_update:
                    mark_step()
                if args.compile:
                    data = data.to_dict()

                # LeanRL-style updates return detached metrics instead of mutating a log input.
                metrics = (graph_update_main if graph_update else update_main)(data)
                logs_dict.update({
                    key: value.clone() if graph_update else value
                    for key, value in metrics.items()
                })
                update_actor = (
                    i % args.policy_frequency == 1
                    if args.num_updates > 1
                    else global_step % args.policy_frequency == 0
                )
                if update_actor:
                    qnet.requires_grad_(False)
                    metrics = (graph_update_pol if graph_update else update_pol)(data)
                    logs_dict.update({
                        key: value.clone() if graph_update else value
                        for key, value in metrics.items()
                    })
                    qnet.requires_grad_(True)

                soft_update(qnet, qnet_target, args.tau)

            if next_checkpoint_step is not None and agent_steps >= next_checkpoint_step:
                print(f"Saving model at agent step {agent_steps}")
                save_checkpoint()
                next_checkpoint_step = (
                    agent_steps // args.save_interval_agent_steps + 1
                ) * args.save_interval_agent_steps

            eval_due = next_eval_step is not None and agent_steps >= next_eval_step
            if "actor_loss" in logs_dict and (
                (global_step % 100 == 0 and start_time is not None) or eval_due
            ):
                now = time.time()
                sps = (agent_steps - last_log_agent_steps) / (now - last_log_time)
                pbar.set_description(f"{sps: 4.4f} sps, " + desc)
                with torch.no_grad():
                    logs = {
                        "actor_loss": logs_dict["actor_loss"].mean(),
                        "qf_loss": logs_dict["qf_loss"].mean(),
                        "qf_max": logs_dict["qf_max"].mean(),
                        "qf_min": logs_dict["qf_min"].mean(),
                        "actor_grad_norm": logs_dict["actor_grad_norm"].mean(),
                        "critic_grad_norm": logs_dict["critic_grad_norm"].mean(),
                        "env_rewards": rewards.mean(),
                        "buffer_rewards": raw_rewards.mean(),
                        **_mean_environment_stats(environment_stats),
                    }

                    if eval_due:
                        print(f"Evaluating at agent step {agent_steps}")
                        logs.update(evaluate())
                        last_eval_agent_steps = agent_steps
                        next_eval_step = (
                            agent_steps // args.eval_interval_agent_steps + 1
                        ) * args.eval_interval_agent_steps

                logs = {
                    "SPS": sps,
                    "agent_steps": agent_steps,
                    "uptime": time.time() - run_start_time,
                    "environment/perc_transitions_used": (
                        interval_valid / interval_received if interval_received else 0.0
                    ),
                    "critic_lr": q_scheduler.get_last_lr()[0],
                    "actor_lr": actor_scheduler.get_last_lr()[0],
                    **logs,
                }
                logger.log(logs, step=agent_steps)
                all_logs.append(logs)
                interval_received = 0
                interval_valid = 0
                last_log_time = now
                last_log_agent_steps = agent_steps

        global_step += 1
        actor_scheduler.step()
        q_scheduler.step()
        pbar.update(1)

    final_path = save_checkpoint()
    final_logs = _mean_environment_stats(environment_stats)
    with torch.no_grad():
        if last_eval_agent_steps != agent_steps:
            final_logs.update(evaluate())
    now = time.time()
    final_logs.update({
        "SPS": (agent_steps - last_log_agent_steps) / (now - last_log_time),
        "agent_steps": agent_steps,
        "uptime": now - run_start_time,
        "critic_lr": q_scheduler.get_last_lr()[0],
        "actor_lr": actor_scheduler.get_last_lr()[0],
    })
    if interval_received:
        final_logs["environment/perc_transitions_used"] = interval_valid / interval_received
    logger.log(final_logs, step=agent_steps)
    all_logs.append(final_logs)
    vecenv.close()
    logger.close(final_path)
    return all_logs
