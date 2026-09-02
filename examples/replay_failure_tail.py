import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

import pufferlib.pytorch
import pufferlib.vector
from pufferlib import pufferl
from pufferlib.ocean.drive.drive import Drive


INITIAL_RESAMPLE_COUNT = 1


def replay_drive_creator(*args, **kwargs):
    env = Drive(*args, **kwargs)
    # The constructor already selected the current resample's maps. The next
    # resample must advance to the following seed instead of repeating it.
    env._resample_count = INITIAL_RESAMPLE_COUNT
    return env


class PythonIntEnvIds:
    """Preserve vector behavior while avoiding NumPy-int torch.arange issues."""

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def recv(self):
        observation, reward, terminal, truncation, info, env_ids, mask = (
            self._env.recv()
        )
        return (
            observation,
            reward,
            terminal,
            truncation,
            info,
            [int(value) for value in env_ids],
            mask,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trainer-state", type=Path, required=True)
    parser.add_argument("--start-epoch", type=int, default=33500)
    parser.add_argument("--failure-epoch", type=int, default=33763)
    parser.add_argument("--resample-seed", type=int, default=1219)
    parser.add_argument("--burn-in-steps", type=int, default=20)
    parser.add_argument("--action-seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/failure_window/tail_replay"),
    )
    return parser.parse_args()


def load_config_without_replay_args():
    argv = sys.argv
    try:
        sys.argv = [argv[0]]
        return pufferl.load_config("puffer_drive")
    finally:
        sys.argv = argv


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_nonfinite_tensor(name, tensor):
    bad = torch.nonzero(~torch.isfinite(tensor), as_tuple=False)
    if not bad.numel():
        return None
    index = bad[0].detach().cpu().tolist()
    return {
        "stage": name,
        "index": index,
        "value": repr(tensor[tuple(index)].detach().cpu().item()),
    }


def buffer_failure(trainer):
    for name in (
        "observations",
        "logprobs",
        "rewards",
        "terminals",
        "values",
        "masks",
    ):
        failure = first_nonfinite_tensor(
            f"rollout_buffer.{name}", getattr(trainer, name)
        )
        if failure:
            return failure
    return None


def parameter_failure(trainer):
    for name, parameter in trainer.uncompiled_policy.named_parameters():
        failure = first_nonfinite_tensor(
            "parameter_after_train", parameter
        )
        if failure:
            failure["parameter"] = name
            return failure
    return None


def optimizer_failure(trainer):
    names = {
        parameter: name
        for name, parameter in trainer.uncompiled_policy.named_parameters()
    }
    for parameter, state in trainer.optimizer.state.items():
        for state_name, value in state.items():
            if not torch.is_tensor(value):
                continue
            failure = first_nonfinite_tensor(
                "optimizer_state_after_train", value
            )
            if failure:
                failure["parameter"] = names.get(parameter, "unknown")
                failure["optimizer_state"] = state_name
                return failure
    return None


def make_pre_update_state(trainer):
    return {
        "epoch": trainer.epoch,
        "global_step": trainer.global_step,
        "model_state_dict": cpu_copy(
            trainer.uncompiled_policy.state_dict()
        ),
        "optimizer_state_dict": cpu_copy(trainer.optimizer.state_dict()),
        "scheduler_state_dict": copy.deepcopy(trainer.scheduler.state_dict()),
        "values": trainer.values.detach().cpu().clone(),
        "ratio": trainer.ratio.detach().cpu().clone(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),
    }


def save_crash_bundle(path, trainer, pre_update, failure, full_args):
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "failure": failure,
        "pre_update": pre_update,
        "full_args": full_args,
        "buffers": {
            "observations": trainer.observations.detach().cpu(),
            "actions": trainer.actions.detach().cpu(),
            "logprobs": trainer.logprobs.detach().cpu(),
            "rewards": trainer.rewards.detach().cpu(),
            "terminals": trainer.terminals.detach().cpu(),
            "truncations": trainer.truncations.detach().cpu(),
            "masks": trainer.masks.detach().cpu(),
        },
    }
    temporary = Path(f"{path}.tmp")
    torch.save(bundle, temporary)
    os.replace(temporary, path)


def burn_in(vecenv, policy, total_agents, steps, action_seed):
    random.seed(action_seed)
    np.random.seed(action_seed)
    torch.manual_seed(action_seed)
    torch.cuda.manual_seed_all(action_seed)

    device = torch.device("cuda")
    hidden_size = policy.hidden_size
    lstm_h = torch.zeros(total_agents, hidden_size, device=device)
    lstm_c = torch.zeros(total_agents, hidden_size, device=device)
    completed = np.zeros(16, dtype=np.int64)

    while int(completed.min()) < steps:
        observation, reward, terminal, truncation, _, env_ids, mask = (
            vecenv.recv()
        )
        env_ids_array = np.asarray(env_ids, dtype=np.int64)
        workers = np.unique(env_ids_array // 1024)
        active_workers = [worker for worker in workers if completed[worker] < steps]
        if len(active_workers) != len(workers):
            raise RuntimeError("Workers became desynchronized during burn-in")

        ids = torch.as_tensor(env_ids_array, device=device, dtype=torch.long)
        state = {
            "reward": torch.as_tensor(reward, device=device),
            "done": torch.as_tensor(
                np.asarray(terminal) | np.asarray(truncation), device=device
            ),
            "env_id": slice(int(env_ids_array[0]), int(env_ids_array[-1]) + 1),
            "mask": mask,
            "lstm_h": lstm_h[ids],
            "lstm_c": lstm_c[ids],
        }
        with torch.no_grad():
            logits, value = policy.forward_eval(
                torch.as_tensor(observation, device=device), state
            )
            action, logprob, entropy = pufferlib.pytorch.sample_logits(logits)

        for name, tensor in (
            ("burn_in.value", value),
            ("burn_in.logprob", logprob),
            ("burn_in.entropy", entropy),
            ("burn_in.lstm_h", state["lstm_h"]),
            ("burn_in.lstm_c", state["lstm_c"]),
        ):
            failure = first_nonfinite_tensor(name, tensor)
            if failure:
                raise RuntimeError(f"Non-finite burn-in tensor: {failure}")

        lstm_h[ids] = state["lstm_h"]
        lstm_c[ids] = state["lstm_c"]
        vecenv.send(action.cpu().numpy())
        for worker in workers:
            completed[worker] += 1

    return completed.tolist()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.trainer_state.is_file():
        raise FileNotFoundError(args.trainer_state)
    checkpoint_sha256 = sha256(args.checkpoint)
    trainer_state_sha256 = sha256(args.trainer_state)
    if checkpoint_sha256 != (
        "83e476c4594d08659c580a347d1516bb0425c9447ceaa5fc5f3ec18732192255"
    ):
        raise RuntimeError(f"Unexpected checkpoint SHA256: {checkpoint_sha256}")
    if trainer_state_sha256 != (
        "6970dd1bc39c9a897718ec0f80a3c0d3676079afc07d5022fac238fb50c43ea7"
    ):
        raise RuntimeError(
            f"Unexpected trainer-state SHA256: {trainer_state_sha256}"
        )
    if args.start_epoch * 32 != 1072000:
        raise RuntimeError("Unexpected checkpoint tick alignment")
    if args.failure_epoch * 32 != 1080416:
        raise RuntimeError("Unexpected failure tick alignment")
    if 1178 * 910 + args.burn_in_steps != args.start_epoch * 32:
        raise RuntimeError("Unexpected resample/burn-in alignment")

    config = load_config_without_replay_args()
    required = {
        "num_workers": config["vec"]["num_workers"],
        "num_envs": config["vec"]["num_envs"],
        "vec_batch_size": config["vec"]["batch_size"],
        "num_agents": config["env"]["num_agents"],
        "num_maps": config["env"]["num_maps"],
        "episode_length": config["env"]["episode_length"],
        "resample_frequency": config["env"]["resample_frequency"],
        "batch_size": config["train"]["batch_size"],
        "minibatch_size": config["train"]["minibatch_size"],
        "rollout_horizon": config["train"]["rollout_horizon"],
        "update_epochs": config["train"]["update_epochs"],
        "reg_mode": config["env"]["reg_mode"],
    }
    expected = {
        "num_workers": 16,
        "num_envs": 16,
        "vec_batch_size": 4,
        "num_agents": 1024,
        "num_maps": 50000,
        "episode_length": 150,
        "resample_frequency": 910,
        "batch_size": 524288,
        "minibatch_size": 32768,
        "rollout_horizon": 32,
        "update_epochs": 1,
        "reg_mode": None,
    }
    if required != expected:
        raise RuntimeError(f"PPO configuration mismatch: {required}")

    config["load_model_path"] = str(args.checkpoint)
    config["train"]["device"] = "cuda"
    config["vec"]["seed"] = args.resample_seed
    config["env"]["uses_memory"] = True
    config["env"]["memory_size"] = config["train"]["rollout_horizon"]
    for key in (
        "self_play_eval",
        "human_replay_eval",
        "render_self_play_eval",
        "render_human_replay_eval",
        "wosac_realism_eval",
    ):
        config["eval"][key] = False

    seed = config["train"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    raw_vecenv = pufferlib.vector.make(
        replay_drive_creator,
        env_kwargs=config["env"],
        **config["vec"],
    )
    vecenv = PythonIntEnvIds(raw_vecenv)
    policy = pufferl.load_policy(config, vecenv, "puffer_drive")
    train_config = dict(
        **config["train"], env="puffer_drive", eval=config["eval"]
    )
    trainer = pufferl.PuffeRL(
        train_config, vecenv, policy, logger=None, full_args=config
    )

    saved_trainer_state = torch.load(
        args.trainer_state, map_location="cpu", weights_only=False
    )
    expected_state = {
        "update": args.start_epoch,
        "global_step": args.start_epoch * train_config["batch_size"],
        "agent_step": args.start_epoch * train_config["batch_size"],
        "model_name": f"model_puffer_drive_{args.start_epoch:06d}.pt",
        "run_id": "mtfk9t24",
    }
    actual_state = {
        key: saved_trainer_state[key] for key in expected_state
    }
    if actual_state != expected_state:
        raise RuntimeError(
            f"Trainer-state/checkpoint mismatch: {actual_state}"
        )
    trainer.optimizer.load_state_dict(
        saved_trainer_state["optimizer_state_dict"]
    )
    failure = optimizer_failure(trainer)
    if failure:
        raise RuntimeError(f"Non-finite restored optimizer state: {failure}")

    result = {
        "status": "running",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "trainer_state": str(args.trainer_state),
        "trainer_state_sha256": trainer_state_sha256,
        "start_epoch": args.start_epoch,
        "failure_epoch": args.failure_epoch,
        "start_tick": args.start_epoch * 32,
        "failure_tick": args.failure_epoch * 32,
        "resample_number": 1178,
        "resample_tick": 1178 * 910,
        "resample_seed_first_worker": args.resample_seed,
        "burn_in_steps": args.burn_in_steps,
        "optimizer_state": "exact epoch 33500 Adam state restored",
        "rng_limitation": "historical action RNG state unavailable",
        "initial_state_limitation": (
            "20-step pre-checkpoint environment burn-in uses checkpoint 33500"
        ),
        "first_failure": None,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "tail_replay_summary.json"
    crash_path = args.output_dir / "tail_replay_crash_bundle.pt"
    start_time = time.time()

    try:
        # Reset with the same seed that initialized resample 1178, then advance
        # from tick 1,071,980 to checkpoint tick 1,072,000.
        vecenv.async_reset(seed=args.resample_seed)
        burn_completed = burn_in(
            vecenv,
            trainer.uncompiled_policy,
            trainer.total_agents,
            args.burn_in_steps,
            args.action_seed,
        )
        result["burn_in_completed"] = burn_completed

        trainer.epoch = saved_trainer_state["update"]
        trainer.global_step = saved_trainer_state["global_step"]
        trainer.last_log_step = trainer.global_step
        trainer.last_log_time = time.time()

        total_epochs = trainer.total_epochs
        lr_factor = max(0.0, 1.0 - args.start_epoch / total_epochs)
        expected_lr = train_config["learning_rate"] * lr_factor
        current_lr = trainer.optimizer.param_groups[0]["lr"]
        if not np.isclose(current_lr, expected_lr, rtol=0, atol=1e-15):
            raise RuntimeError(
                f"Trainer-state LR mismatch: {current_lr} != {expected_lr}"
            )
        trainer.scheduler.last_epoch = args.start_epoch
        trainer.scheduler._last_lr = [
            group["lr"] for group in trainer.optimizer.param_groups
        ]
        result["initial_learning_rate"] = current_lr

        while trainer.epoch < args.failure_epoch:
            torch.compiler.cudagraph_mark_step_begin()
            trainer.evaluate()
            failure = buffer_failure(trainer)
            # PuffeRL.train() performs the same reset as its first operation.
            # Doing it here makes a crash bundle independently replayable.
            trainer.ratio.fill_(1)
            pre_update = make_pre_update_state(trainer)
            if failure:
                failure["epoch"] = trainer.epoch
                failure["global_step"] = trainer.global_step
                result["first_failure"] = failure
                save_crash_bundle(
                    crash_path, trainer, pre_update, failure, config
                )
                break

            try:
                torch.compiler.cudagraph_mark_step_begin()
                trainer.train()
            except Exception as error:
                failure = {
                    "stage": "train_exception",
                    "epoch": trainer.epoch,
                    "global_step": trainer.global_step,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                result["first_failure"] = failure
                save_crash_bundle(
                    crash_path, trainer, pre_update, failure, config
                )
                break

            failure = parameter_failure(trainer) or optimizer_failure(trainer)
            if failure:
                failure["epoch"] = trainer.epoch
                failure["global_step"] = trainer.global_step
                result["first_failure"] = failure
                save_crash_bundle(
                    crash_path, trainer, pre_update, failure, config
                )
                break

            if trainer.epoch % 10 == 0 or trainer.epoch == args.failure_epoch:
                print(
                    f"tail_epoch={trainer.epoch}/{args.failure_epoch} "
                    f"lr={trainer.optimizer.param_groups[0]['lr']:.9g} "
                    f"elapsed={time.time() - start_time:.1f}s",
                    flush=True,
                )

        result["end_epoch"] = trainer.epoch
        result["end_global_step"] = trainer.global_step
        result["elapsed_seconds"] = time.time() - start_time
        if result["first_failure"] is None:
            result["status"] = "passed"
        else:
            result["status"] = "failed"
            result["crash_bundle"] = str(crash_path)
    finally:
        trainer.utilization.stop()
        vecenv.close()

    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"output = {result_path}")
    if result["first_failure"] is not None:
        print("FAILURE_TAIL_REPLAY_REPRODUCED")
        raise SystemExit(2)
    print("FAILURE_TAIL_REPLAY_PASSED")


if __name__ == "__main__":
    main()
