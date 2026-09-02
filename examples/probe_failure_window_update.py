import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

import pufferlib.pytorch
from pufferlib import pufferl
from examples.probe_failure_window_rollout import load_map_lookup, location


class ProbeFailure(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--assignments",
        type=Path,
        default=Path("results/failure_window/failure_window_map_assignments.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/failure_window/failure_window_update_probe.json"
        ),
    )
    parser.add_argument("--first-seed", type=int, default=1228)
    parser.add_argument("--action-seed", type=int, default=42)
    parser.add_argument("--last-resample-tick", type=int, default=1080170)
    parser.add_argument("--target-start", type=int, default=215)
    parser.add_argument("--target-end", type=int, default=246)
    parser.add_argument("--failure-epoch", type=int, default=33763)
    return parser.parse_args()


def load_config_without_probe_args():
    argv = sys.argv
    try:
        sys.argv = [argv[0]]
        return pufferl.load_config("puffer_drive")
    finally:
        sys.argv = argv


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def max_abs(tensor):
    if not tensor.numel():
        return 0.0
    return float(tensor.detach().abs().max().cpu())


def tensor_failure(stage, tensor, minibatch=None, sampled_segments=None):
    bad = torch.nonzero(~torch.isfinite(tensor), as_tuple=False)
    if not bad.numel():
        return None
    index = bad[0].detach().cpu().tolist()
    result = {
        "stage": stage,
        "index": index,
        "value": repr(tensor[tuple(index)].detach().cpu().item()),
    }
    if minibatch is not None:
        result["minibatch"] = int(minibatch)
    if sampled_segments is not None and index:
        row = index[0]
        if row < sampled_segments.numel():
            result["sampled_segment"] = int(sampled_segments[row].cpu())
            if len(index) > 1:
                result["timestep"] = int(index[1])
    return result


def parameter_failure(stage, policy, minibatch):
    for name, parameter in policy.named_parameters():
        failure = tensor_failure(stage, parameter, minibatch)
        if failure:
            failure["parameter"] = name
            return failure
    return None


def gradient_failure(stage, policy, minibatch):
    for name, parameter in policy.named_parameters():
        if parameter.grad is None:
            continue
        failure = tensor_failure(stage, parameter.grad, minibatch)
        if failure:
            failure["parameter"] = name
            return failure
    return None


def optimizer_failure(optimizer, policy, minibatch):
    names = {parameter: name for name, parameter in policy.named_parameters()}
    for parameter, state in optimizer.state.items():
        for state_name, value in state.items():
            if not torch.is_tensor(value):
                continue
            failure = tensor_failure(
                "optimizer_state_after_step", value, minibatch
            )
            if failure:
                failure["parameter"] = names.get(parameter, "unknown")
                failure["optimizer_state"] = state_name
                return failure
    return None


def attach_segment_location(failure, lookup, agents_per_worker):
    segment = failure.get("sampled_segment")
    if segment is None:
        return
    failure.update(
        location(
            np.asarray([segment], dtype=np.int64),
            0,
            lookup,
            agents_per_worker,
        )
    )
    failure.pop("batch_agent", None)


def require_finite(
    stage,
    tensor,
    failure_box,
    lookup,
    agents_per_worker,
    minibatch=None,
    sampled_segments=None,
):
    failure = tensor_failure(stage, tensor, minibatch, sampled_segments)
    if failure:
        attach_segment_location(failure, lookup, agents_per_worker)
        failure_box[0] = failure
        raise ProbeFailure(stage)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.assignments.is_file():
        raise FileNotFoundError(args.assignments)
    if args.target_end - args.target_start + 1 != 32:
        raise RuntimeError("Target window must contain exactly 32 steps")

    config = load_config_without_probe_args()
    expected = {
        "num_workers": 16,
        "num_envs": 16,
        "vec_batch_size": 4,
        "num_agents": 1024,
        "batch_size": 524288,
        "minibatch_size": 32768,
        "max_minibatch_size": 32768,
        "rollout_horizon": 32,
        "update_epochs": 1,
        "optimizer": "adam",
        "precision": "float32",
        "dynamics_model": "delta_local",
        "reg_mode": None,
    }
    actual = {
        "num_workers": config["vec"]["num_workers"],
        "num_envs": config["vec"]["num_envs"],
        "vec_batch_size": config["vec"]["batch_size"],
        "num_agents": config["env"]["num_agents"],
        "batch_size": config["train"]["batch_size"],
        "minibatch_size": config["train"]["minibatch_size"],
        "max_minibatch_size": config["train"]["max_minibatch_size"],
        "rollout_horizon": config["train"]["rollout_horizon"],
        "update_epochs": config["train"]["update_epochs"],
        "optimizer": config["train"]["optimizer"],
        "precision": config["train"]["precision"],
        "dynamics_model": config["env"]["dynamics_model"],
        "reg_mode": config["env"]["reg_mode"],
    }
    if actual != expected:
        raise RuntimeError(f"PPO configuration mismatch: {actual}")

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    if sorted(checkpoint) != ["full_args", "model_state_dict"]:
        raise RuntimeError(f"Unexpected checkpoint keys: {sorted(checkpoint)}")

    checkpoint_args = checkpoint["full_args"]
    checkpoint_audit = {
        "rnn_name": checkpoint_args["rnn_name"],
        "num_workers": checkpoint_args["vec"]["num_workers"],
        "num_envs": checkpoint_args["vec"]["num_envs"],
        "vec_batch_size": checkpoint_args["vec"]["batch_size"],
        "num_agents": checkpoint_args["env"]["num_agents"],
        "num_maps": checkpoint_args["env"]["num_maps"],
        "episode_length": checkpoint_args["env"]["episode_length"],
        "goal_radius": checkpoint_args["env"]["goal_radius"],
        "dynamics_model": checkpoint_args["env"]["dynamics_model"],
        "reg_mode": checkpoint_args["env"]["reg_mode"],
        "lambda_value": checkpoint_args["env"]["lambda_value"],
        "batch_size": checkpoint_args["train"]["batch_size"],
        "minibatch_size": checkpoint_args["train"]["minibatch_size"],
        "max_minibatch_size": checkpoint_args["train"][
            "max_minibatch_size"
        ],
        "rollout_horizon": checkpoint_args["train"]["rollout_horizon"],
        "update_epochs": checkpoint_args["train"]["update_epochs"],
        "optimizer": checkpoint_args["train"]["optimizer"],
        "precision": checkpoint_args["train"]["precision"],
        "seed": checkpoint_args["train"]["seed"],
        "total_timesteps": checkpoint_args["train"]["total_timesteps"],
        "learning_rate": checkpoint_args["train"]["learning_rate"],
        "anneal_lr": checkpoint_args["train"]["anneal_lr"],
        "adam_beta1": checkpoint_args["train"]["adam_beta1"],
        "adam_beta2": checkpoint_args["train"]["adam_beta2"],
        "adam_eps": checkpoint_args["train"]["adam_eps"],
        "clip_coef": checkpoint_args["train"]["clip_coef"],
        "ent_coef": checkpoint_args["train"]["ent_coef"],
        "gae_lambda": checkpoint_args["train"]["gae_lambda"],
        "gamma": checkpoint_args["train"]["gamma"],
        "max_grad_norm": checkpoint_args["train"]["max_grad_norm"],
        "prio_alpha": checkpoint_args["train"]["prio_alpha"],
        "prio_beta0": checkpoint_args["train"]["prio_beta0"],
        "vf_clip_coef": checkpoint_args["train"]["vf_clip_coef"],
        "vf_coef": checkpoint_args["train"]["vf_coef"],
        "vtrace_c_clip": checkpoint_args["train"]["vtrace_c_clip"],
        "vtrace_rho_clip": checkpoint_args["train"]["vtrace_rho_clip"],
    }
    expected_checkpoint = {
        "rnn_name": "Recurrent",
        "num_workers": 16,
        "num_envs": 16,
        "vec_batch_size": 4,
        "num_agents": 1024,
        "num_maps": 50000,
        "episode_length": 150,
        "goal_radius": 2.0,
        "dynamics_model": "delta_local",
        "reg_mode": None,
        "lambda_value": 0.0,
        "batch_size": 524288,
        "minibatch_size": 32768,
        "max_minibatch_size": 32768,
        "rollout_horizon": 32,
        "update_epochs": 1,
        "optimizer": "adam",
        "precision": "float32",
        "seed": 42,
        "total_timesteps": 20000000000,
        "learning_rate": 0.004261385334377629,
        "anneal_lr": True,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_eps": 1e-8,
        "clip_coef": 0.2,
        "ent_coef": 0.001,
        "gae_lambda": 0.95,
        "gamma": 0.99,
        "max_grad_norm": 1,
        "prio_alpha": 0.8499999999999999,
        "prio_beta0": 0.8499999999999999,
        "vf_clip_coef": 0.1999999999999999,
        "vf_coef": 2,
        "vtrace_c_clip": 1,
        "vtrace_rho_clip": 1,
    }
    if checkpoint_audit != expected_checkpoint:
        raise RuntimeError(
            f"Checkpoint PPO configuration mismatch: {checkpoint_audit}"
        )

    environment_keys = (
        "num_agents",
        "action_type",
        "dynamics_model",
        "fix_rewards",
        "fix_lambdas",
        "lambda_value",
        "reward_vehicle_collision",
        "reward_offroad_collision",
        "reward_goal",
        "obs_partner_noise_speed",
        "obs_partner_noise_pos",
        "dt",
        "async_resets",
        "goal_radius",
        "goal_speed",
        "goal_behavior",
        "goal_target_distance",
        "collision_behavior",
        "offroad_behavior",
        "episode_length",
        "resample_frequency",
        "termination_mode",
        "map_dir",
        "num_maps",
        "init_steps",
        "control_mode",
        "sdc_controller",
        "non_sdc_controller",
        "non_vehicle_controller",
        "init_mode",
        "max_controlled_agents",
        "reg_mode",
        "anchor_cpt_path",
    )
    runtime_mismatch = {
        key: {
            "runtime": config["env"].get(key),
            "checkpoint": checkpoint_args["env"].get(key),
        }
        for key in environment_keys
        if config["env"].get(key) != checkpoint_args["env"].get(key)
    }
    if config["policy"] != checkpoint_args["policy"]:
        runtime_mismatch["policy"] = {
            "runtime": config["policy"],
            "checkpoint": checkpoint_args["policy"],
        }
    if config["rnn"] != checkpoint_args["rnn"]:
        runtime_mismatch["rnn"] = {
            "runtime": config["rnn"],
            "checkpoint": checkpoint_args["rnn"],
        }
    if runtime_mismatch:
        raise RuntimeError(
            f"Runtime/checkpoint configuration mismatch: {runtime_mismatch}"
        )

    workers = expected["num_workers"]
    agents_per_worker = expected["num_agents"]
    total_agents = workers * agents_per_worker
    horizon = expected["rollout_horizon"]
    lookup = load_map_lookup(args.assignments, workers, agents_per_worker)

    config["vec"]["seed"] = args.first_seed
    config["load_model_path"] = str(args.checkpoint)
    config["train"]["device"] = "cuda"

    random.seed(args.action_seed)
    np.random.seed(args.action_seed)
    torch.manual_seed(args.action_seed)
    torch.cuda.manual_seed_all(args.action_seed)

    device = torch.device("cuda")
    vecenv = None
    failure_box = [None]
    start_time = time.time()
    rollout_seconds = None
    update_seconds = None
    completed_minibatches = 0
    update_maxima = {}

    try:
        vecenv = pufferl.load_env("puffer_drive", config)
        if vecenv.num_agents != total_agents:
            raise RuntimeError(
                f"Expected {total_agents} total agents, got {vecenv.num_agents}"
            )
        policy = pufferl.load_policy(config, vecenv, "puffer_drive")
        policy.train()

        failure = parameter_failure("checkpoint_parameter", policy, -1)
        if failure:
            failure_box[0] = failure
            raise ProbeFailure("checkpoint parameter")

        # Construction consumes RNG although checkpoint loading overwrites weights.
        random.seed(args.action_seed)
        np.random.seed(args.action_seed)
        torch.manual_seed(args.action_seed)
        torch.cuda.manual_seed_all(args.action_seed)

        obs_shape = vecenv.single_observation_space.shape
        action_shape = vecenv.single_action_space.shape
        observation_dtype = pufferlib.pytorch.numpy_to_torch_dtype_dict[
            vecenv.single_observation_space.dtype
        ]
        action_dtype = pufferlib.pytorch.numpy_to_torch_dtype_dict[
            vecenv.single_action_space.dtype
        ]
        observations = torch.zeros(
            total_agents,
            horizon,
            *obs_shape,
            device=device,
            dtype=observation_dtype,
        )
        actions = torch.zeros(
            total_agents,
            horizon,
            *action_shape,
            device=device,
            dtype=action_dtype,
        )
        logprobs = torch.zeros(total_agents, horizon, device=device)
        rewards = torch.zeros(total_agents, horizon, device=device)
        terminals = torch.zeros(total_agents, horizon, device=device)
        values = torch.zeros(total_agents, horizon, device=device)
        masks = torch.ones(total_agents, horizon, device=device)
        ratio_buffer = torch.ones(total_agents, horizon, device=device)
        agent_dead = torch.zeros(total_agents, device=device, dtype=torch.bool)

        hidden_size = policy.hidden_size
        lstm_h = torch.zeros(total_agents, hidden_size, device=device)
        lstm_c = torch.zeros(total_agents, hidden_size, device=device)
        offsets = np.full(workers, -1, dtype=np.int64)

        first_boundary = 32 - ((args.last_resample_tick - 1) % 32)
        reset_offsets = set(range(first_boundary, args.target_end + 1, 32))
        if args.target_start not in reset_offsets:
            raise RuntimeError(
                f"Target start {args.target_start} is not a rollout boundary"
            )

        vecenv.async_reset(seed=args.first_seed)
        next_progress = 0
        rollout_start = time.time()

        while int(offsets.min()) < args.target_end:
            observation, reward, terminal, truncation, _, env_ids, mask = (
                vecenv.recv()
            )
            env_ids = np.asarray(env_ids, dtype=np.int64)
            batch_workers = np.unique(env_ids // agents_per_worker)
            for worker in batch_workers:
                offsets[worker] += 1
            batch_offsets = {int(offsets[worker]) for worker in batch_workers}
            if len(batch_offsets) != 1:
                raise RuntimeError(
                    f"Desynchronized workers: {sorted(batch_offsets)}"
                )
            offset = batch_offsets.pop()
            ids = torch.as_tensor(env_ids, device=device, dtype=torch.long)

            if offset == 0 or offset in reset_offsets:
                lstm_h[ids] = 0
                lstm_c[ids] = 0
            if offset == args.target_start:
                agent_dead[ids] = False

            observation_t = torch.as_tensor(observation, device=device)
            reward_t = torch.as_tensor(reward, device=device)
            terminal_t = torch.as_tensor(terminal, device=device)
            truncation_t = torch.as_tensor(truncation, device=device)
            done_t = (terminal_t + truncation_t).clamp(max=1)
            state = {
                "reward": reward_t,
                "done": done_t,
                "env_id": slice(int(env_ids[0]), int(env_ids[-1]) + 1),
                "mask": mask,
                "lstm_h": lstm_h[ids],
                "lstm_c": lstm_c[ids],
            }

            with torch.no_grad():
                logits, value = policy.forward_eval(observation_t, state)
                action, logprob, _ = pufferlib.pytorch.sample_logits(logits)
                reward_t = torch.clamp(reward_t, -1, 1)

            for stage, tensor in (
                ("rollout_observation", observation_t),
                ("rollout_reward", reward_t),
                ("rollout_value", value),
                ("rollout_logprob", logprob),
                ("rollout_lstm_h", state["lstm_h"]),
                ("rollout_lstm_c", state["lstm_c"]),
            ):
                require_finite(
                    stage,
                    tensor,
                    failure_box,
                    lookup,
                    agents_per_worker,
                )
            for head, head_logits in enumerate(logits):
                require_finite(
                    f"rollout_logits_head_{head}",
                    head_logits,
                    failure_box,
                    lookup,
                    agents_per_worker,
                )

            lstm_h[ids] = state["lstm_h"]
            lstm_c[ids] = state["lstm_c"]

            if args.target_start <= offset <= args.target_end:
                timestep = offset - args.target_start
                observations[ids, timestep] = observation_t
                actions[ids, timestep] = action
                logprobs[ids, timestep] = logprob

                terminal_flag = (terminal_t > 0) & (truncation_t == 0)
                truncation_flag = truncation_t > 0
                valid = (~agent_dead[ids]).float()
                masks[ids, timestep] = valid
                agent_dead[ids] |= terminal_flag.bool()
                agent_dead[ids] &= ~truncation_flag.bool()

                if timestep > 0:
                    truncation_mask = (truncation_t > 0) & (terminal_t == 0)
                    reward_t = reward_t + (
                        truncation_mask.to(reward_t.dtype)
                        * config["train"]["gamma"]
                        * values[ids, timestep - 1]
                    )
                rewards[ids, timestep] = reward_t
                terminals[ids, timestep] = done_t.float()
                values[ids, timestep] = value.flatten()

            vecenv.send(action.cpu().numpy())
            minimum = int(offsets.min())
            if minimum >= next_progress:
                print(
                    f"minimum_offset={minimum}/{args.target_end} "
                    f"elapsed={time.time() - rollout_start:.1f}s",
                    flush=True,
                )
                next_progress += 25

        rollout_seconds = time.time() - rollout_start
        rewards *= masks
        values *= masks
        terminals[masks == 0] = 1.0

        for stage, tensor in (
            ("buffer_observations", observations),
            ("buffer_actions", actions.float()),
            ("buffer_logprobs", logprobs),
            ("buffer_rewards", rewards),
            ("buffer_terminals", terminals),
            ("buffer_values", values),
            ("buffer_masks", masks),
        ):
            require_finite(
                stage,
                tensor,
                failure_box,
                lookup,
                agents_per_worker,
            )

        train_config = config["train"]
        total_epochs = (
            train_config["total_timesteps"] // train_config["batch_size"]
        )
        lr_factor = max(0.0, 1.0 - args.failure_epoch / max(1, total_epochs))
        probe_lr = train_config["learning_rate"] * lr_factor
        optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=probe_lr,
            betas=(train_config["adam_beta1"], train_config["adam_beta2"]),
            eps=train_config["adam_eps"],
        )

        total_minibatches = (
            train_config["update_epochs"]
            * train_config["batch_size"]
            // train_config["minibatch_size"]
        )
        minibatch_segments = train_config["minibatch_size"] // horizon
        anneal_beta = train_config["prio_beta0"] + (
            (1 - train_config["prio_beta0"])
            * train_config["prio_alpha"]
            * args.failure_epoch
            / total_epochs
        )
        ratio_buffer.fill_(1)
        update_start = time.time()

        for minibatch in range(total_minibatches):
            advantages = torch.zeros_like(values)
            advantages = pufferl.compute_puff_advantage(
                values,
                rewards,
                terminals,
                ratio_buffer,
                advantages,
                train_config["gamma"],
                train_config["gae_lambda"],
                train_config["vtrace_rho_clip"],
                train_config["vtrace_c_clip"],
            )
            require_finite(
                "advantages_full",
                advantages,
                failure_box,
                lookup,
                agents_per_worker,
                minibatch,
            )

            priority_advantage = advantages.abs().sum(axis=1)
            priority_weights = torch.nan_to_num(
                priority_advantage ** train_config["prio_alpha"], 0, 0, 0
            )
            priority_probs = (priority_weights + 1e-6) / (
                priority_weights.sum() + 1e-6
            )
            require_finite(
                "priority_probs",
                priority_probs,
                failure_box,
                lookup,
                agents_per_worker,
                minibatch,
            )
            sampled = torch.multinomial(priority_probs, minibatch_segments)
            minibatch_priority = (
                total_agents * priority_probs[sampled, None]
            ) ** -anneal_beta
            require_finite(
                "minibatch_priority",
                minibatch_priority,
                failure_box,
                lookup,
                agents_per_worker,
                minibatch,
                sampled,
            )

            mb_obs = observations[sampled]
            mb_actions = actions[sampled]
            mb_logprobs = logprobs[sampled]
            mb_rewards = rewards[sampled]
            mb_terminals = terminals[sampled]
            mb_values = values[sampled]
            mb_returns = advantages[sampled] + mb_values
            mb_advantages = advantages[sampled]
            mb_masks = masks[sampled]

            state = {"action": mb_actions, "lstm_h": None, "lstm_c": None}
            logits, newvalue = policy(mb_obs, state)
            _, newlogprob, entropy = pufferlib.pytorch.sample_logits(
                logits, action=mb_actions
            )
            newlogprob = newlogprob.reshape(mb_logprobs.shape)
            logratio = newlogprob - mb_logprobs
            ratio = logratio.exp()

            for stage, tensor in (
                ("train_newvalue", newvalue),
                ("train_newlogprob", newlogprob),
                ("train_entropy", entropy),
                ("train_logratio", logratio),
                ("train_ratio", ratio),
                ("train_returns", mb_returns),
            ):
                require_finite(
                    stage,
                    tensor,
                    failure_box,
                    lookup,
                    agents_per_worker,
                    minibatch,
                    sampled,
                )

            ratio_buffer[sampled] = ratio.detach()
            discarded_advantage = advantages[sampled]
            discarded_advantage = pufferl.compute_puff_advantage(
                mb_values,
                mb_rewards,
                mb_terminals,
                ratio,
                discarded_advantage,
                train_config["gamma"],
                train_config["gae_lambda"],
                train_config["vtrace_rho_clip"],
                train_config["vtrace_c_clip"],
            )
            require_finite(
                "discarded_recomputed_advantage",
                discarded_advantage,
                failure_box,
                lookup,
                agents_per_worker,
                minibatch,
                sampled,
            )

            advantage = mb_advantages
            valid_advantage = advantage[mb_masks == 1]
            if valid_advantage.numel() > 0:
                advantage_mean = valid_advantage.mean()
                advantage_std = valid_advantage.std() + 1e-8
            else:
                advantage_mean = advantage.mean()
                advantage_std = advantage.std() + 1e-8
            advantage = (advantage - advantage_mean) / advantage_std
            advantage = advantage * mb_masks
            valid_count = mb_masks.sum().clamp(min=1)

            pg_loss1 = -advantage * ratio
            pg_loss2 = -advantage * torch.clamp(
                ratio,
                1 - train_config["clip_coef"],
                1 + train_config["clip_coef"],
            )
            pg_loss = (
                torch.max(pg_loss1, pg_loss2) * mb_masks
            ).sum() / valid_count

            newvalue = newvalue.view(mb_returns.shape)
            clipped_value = mb_values + torch.clamp(
                newvalue - mb_values,
                -train_config["vf_clip_coef"],
                train_config["vf_clip_coef"],
            )
            value_loss_unclipped = (newvalue - mb_returns) ** 2
            value_loss_clipped = (clipped_value - mb_returns) ** 2
            value_loss = 0.5 * (
                torch.max(value_loss_unclipped, value_loss_clipped) * mb_masks
            ).sum() / valid_count
            entropy_loss = (
                entropy.reshape(mb_masks.shape) * mb_masks
            ).sum() / valid_count
            ratio_buffer[sampled] = (ratio * mb_masks).detach()
            old_approx_kl = ((-logratio) * mb_masks).sum() / valid_count
            approx_kl = (
                ((ratio - 1) - logratio) * mb_masks
            ).sum() / valid_count
            clipfrac = (
                ((ratio - 1.0).abs() > train_config["clip_coef"])
                * mb_masks
            ).sum() / valid_count
            loss = (
                pg_loss
                + train_config["vf_coef"] * value_loss
                - train_config["ent_coef"] * entropy_loss
            )

            for stage, tensor in (
                ("advantage_mean", advantage_mean),
                ("advantage_std", advantage_std),
                ("normalized_advantage", advantage),
                ("policy_loss", pg_loss),
                ("value_loss_unclipped", value_loss_unclipped),
                ("value_loss_clipped", value_loss_clipped),
                ("value_loss", value_loss),
                ("entropy_loss", entropy_loss),
                ("old_approx_kl", old_approx_kl),
                ("approx_kl", approx_kl),
                ("clipfrac", clipfrac),
                ("total_loss", loss),
            ):
                require_finite(
                    stage,
                    tensor,
                    failure_box,
                    lookup,
                    agents_per_worker,
                    minibatch,
                    sampled,
                )

            values[sampled] = newvalue.detach().float()
            loss.backward()
            failure = gradient_failure("gradient_before_clip", policy, minibatch)
            if failure:
                failure_box[0] = failure
                raise ProbeFailure("gradient before clip")

            gradient_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), train_config["max_grad_norm"]
            )
            require_finite(
                "gradient_norm",
                gradient_norm,
                failure_box,
                lookup,
                agents_per_worker,
                minibatch,
            )
            failure = gradient_failure("gradient_after_clip", policy, minibatch)
            if failure:
                failure_box[0] = failure
                raise ProbeFailure("gradient after clip")

            optimizer.step()
            failure = parameter_failure(
                "parameter_after_optimizer_step", policy, minibatch
            )
            if failure:
                failure_box[0] = failure
                raise ProbeFailure("parameter after optimizer step")
            failure = optimizer_failure(optimizer, policy, minibatch)
            if failure:
                failure_box[0] = failure
                raise ProbeFailure("optimizer state")
            optimizer.zero_grad()

            completed_minibatches += 1
            update_maxima = {
                "advantage": max(
                    update_maxima.get("advantage", 0.0), max_abs(advantage)
                ),
                "gradient_norm_before_clip": max(
                    update_maxima.get("gradient_norm_before_clip", 0.0),
                    float(gradient_norm.detach().cpu()),
                ),
                "logratio": max(
                    update_maxima.get("logratio", 0.0), max_abs(logratio)
                ),
                "ratio": max(
                    update_maxima.get("ratio", 0.0), max_abs(ratio)
                ),
                "total_loss": max(
                    update_maxima.get("total_loss", 0.0), max_abs(loss)
                ),
                "value_loss": max(
                    update_maxima.get("value_loss", 0.0),
                    max_abs(value_loss),
                ),
            }
            print(
                f"minibatch={completed_minibatches}/{total_minibatches} "
                f"loss={float(loss.detach().cpu()):.6g} "
                f"grad_norm={float(gradient_norm.detach().cpu()):.6g}",
                flush=True,
            )

        update_seconds = time.time() - update_start

    except ProbeFailure:
        pass
    finally:
        if vecenv is not None:
            vecenv.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "failed" if failure_box[0] else "passed",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_config_audit": checkpoint_audit,
        "failure_epoch": args.failure_epoch,
        "first_seed": args.first_seed,
        "last_seed": args.first_seed + workers - 1,
        "action_seed": args.action_seed,
        "target_start": args.target_start,
        "target_end": args.target_end,
        "completed_minibatches": completed_minibatches,
        "expected_minibatches": 16,
        "probe_optimizer": "fresh Adam; historical optimizer state unavailable",
        "probe_learning_rate": locals().get("probe_lr"),
        "update_max_abs": update_maxima,
        "first_failure": failure_box[0],
        "rollout_seconds": rollout_seconds,
        "update_seconds": update_seconds,
        "elapsed_seconds": time.time() - start_time,
    }
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"output = {args.output}")
    if failure_box[0]:
        print("FAILURE_WINDOW_UPDATE_PROBE_FAILED")
        raise SystemExit(2)
    print("FAILURE_WINDOW_UPDATE_PROBE_PASSED")


if __name__ == "__main__":
    main()
