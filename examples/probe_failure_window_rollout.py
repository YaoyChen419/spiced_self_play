import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

import pufferlib.pytorch
from pufferlib import pufferl


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
        "--output-dir", type=Path, default=Path("results/failure_window")
    )
    parser.add_argument("--first-seed", type=int, default=1228)
    parser.add_argument("--action-seed", type=int, default=42)
    parser.add_argument("--last-resample-tick", type=int, default=1080170)
    parser.add_argument("--target-start", type=int, default=215)
    parser.add_argument("--target-end", type=int, default=246)
    return parser.parse_args()


def load_config_without_probe_args():
    argv = sys.argv
    try:
        sys.argv = [argv[0]]
        return pufferl.load_config("puffer_drive")
    finally:
        sys.argv = argv


def feature_name(index, ego_features=12, partner_objects=31, object_features=7):
    if index < ego_features:
        return f"ego[{index}]"
    index -= ego_features
    partner_size = partner_objects * object_features
    if index < partner_size:
        return f"partner[{index // object_features}][{index % object_features}]"
    index -= partner_size
    return f"road[{index // object_features}][{index % object_features}]"


def load_map_lookup(path, workers=16, agents_per_worker=1024):
    lookup = {
        worker: [None] * agents_per_worker
        for worker in range(workers)
    }
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            worker = int(row["worker_idx"])
            start = int(row["agent_start"])
            stop = min(int(row["agent_stop"]), agents_per_worker)
            value = {
                "map_id": int(row["map_id"]),
                "map_slot": int(row["map_slot"]),
                "resample_seed": int(row["resample_seed"]),
            }
            for agent in range(start, stop):
                if lookup[worker][agent] is not None:
                    raise RuntimeError(
                        f"duplicate assignment: worker={worker}, agent={agent}"
                    )
                lookup[worker][agent] = value

    for worker, values in lookup.items():
        missing = sum(value is None for value in values)
        if missing:
            raise RuntimeError(f"worker {worker}: {missing} agents have no map")
    return lookup


def location(env_ids, batch_agent, lookup, agents_per_worker):
    global_agent = int(env_ids[batch_agent])
    worker = global_agent // agents_per_worker
    local_agent = global_agent % agents_per_worker
    return {
        "batch_agent": int(batch_agent),
        "global_agent": global_agent,
        "worker_idx": worker,
        "local_agent": local_agent,
        **lookup[worker][local_agent],
    }


def first_nonfinite_numpy(name, array, env_ids, lookup, agents_per_worker, offset):
    bad = np.argwhere(~np.isfinite(array))
    if not len(bad):
        return None
    index = bad[0].tolist()
    record = {
        "phase": name,
        "offset_from_resample": int(offset),
        "index": index,
        "value": repr(array[tuple(index)]),
    }
    if array.ndim and index[0] < len(env_ids):
        record.update(location(env_ids, index[0], lookup, agents_per_worker))
    if name == "observation" and len(index) > 1:
        record["feature"] = feature_name(index[1])
    return record


def first_nonfinite_tensor(name, tensor, env_ids, lookup, agents_per_worker, offset, **extra):
    bad = torch.nonzero(~torch.isfinite(tensor), as_tuple=False)
    if not bad.numel():
        return None
    index = bad[0].detach().cpu().tolist()
    record = {
        "phase": name,
        "offset_from_resample": int(offset),
        "index": index,
        "value": repr(tensor[tuple(index)].detach().cpu().item()),
        **extra,
    }
    if tensor.ndim and index[0] < len(env_ids):
        record.update(location(env_ids, index[0], lookup, agents_per_worker))
    return record


def update_maxima(maxima, name, value):
    if isinstance(value, np.ndarray):
        current = float(np.max(np.abs(value))) if value.size else 0.0
    else:
        current = float(value.detach().abs().max().cpu()) if value.numel() else 0.0
    maxima[name] = max(maxima.get(name, 0.0), current)


def main():
    args = parse_args()
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
        "batch_size": 4,
        "num_agents": 1024,
        "rollout_horizon": 32,
        "dynamics_model": "delta_local",
        "reg_mode": None,
    }
    actual = {
        "num_workers": config["vec"]["num_workers"],
        "num_envs": config["vec"]["num_envs"],
        "batch_size": config["vec"]["batch_size"],
        "num_agents": config["env"]["num_agents"],
        "rollout_horizon": config["train"]["rollout_horizon"],
        "dynamics_model": config["env"]["dynamics_model"],
        "reg_mode": config["env"]["reg_mode"],
    }
    if actual != expected:
        raise RuntimeError(f"PPO configuration mismatch: {actual}")

    workers = expected["num_workers"]
    agents_per_worker = expected["num_agents"]
    total_agents = workers * agents_per_worker
    lookup = load_map_lookup(args.assignments, workers, agents_per_worker)

    config["vec"]["seed"] = args.first_seed
    config["load_model_path"] = str(args.checkpoint)
    config["train"]["device"] = "cuda"

    random.seed(args.action_seed)
    np.random.seed(args.action_seed)
    torch.manual_seed(args.action_seed)
    torch.cuda.manual_seed_all(args.action_seed)

    vecenv = None
    failure = None
    maxima = {}
    start_time = time.time()
    offsets = np.full(workers, -1, dtype=np.int64)
    next_progress = 0

    try:
        vecenv = pufferl.load_env("puffer_drive", config)
        if vecenv.num_agents != total_agents:
            raise RuntimeError(
                f"Expected {total_agents} total agents, got {vecenv.num_agents}"
            )
        policy = pufferl.load_policy(config, vecenv, "puffer_drive")
        policy.eval()

        for name, parameter in policy.named_parameters():
            if not torch.isfinite(parameter).all():
                raise RuntimeError(f"Non-finite checkpoint parameter: {name}")

        # Policy construction consumes RNG even though checkpoint weights replace
        # the initialization. Reset here so action_seed controls rollout sampling.
        random.seed(args.action_seed)
        np.random.seed(args.action_seed)
        torch.manual_seed(args.action_seed)
        torch.cuda.manual_seed_all(args.action_seed)

        device = torch.device("cuda")
        hidden_size = policy.hidden_size
        lstm_h = torch.zeros((total_agents, hidden_size), device=device)
        lstm_c = torch.zeros((total_agents, hidden_size), device=device)

        # At the real resample, the first new-map rollout boundary is offset 23;
        # subsequent boundaries are 32 steps apart, including target offset 215.
        first_boundary = 32 - ((args.last_resample_tick - 1) % 32)
        reset_offsets = set(range(first_boundary, args.target_end + 1, 32))
        if args.target_start not in reset_offsets:
            raise RuntimeError(
                f"Target start {args.target_start} is not a rollout boundary: "
                f"{sorted(reset_offsets)}"
            )

        # Constructors now select the exact active map batches with seeds 1228..1243.
        vecenv.async_reset(seed=args.first_seed)

        while int(offsets.min()) < args.target_end:
            observation, reward, terminal, truncation, _, env_ids, mask = vecenv.recv()
            env_ids = np.asarray(env_ids, dtype=np.int64)
            batch_workers = np.unique(env_ids // agents_per_worker)
            for worker in batch_workers:
                offsets[worker] += 1
            batch_offsets = {int(offsets[worker]) for worker in batch_workers}
            if len(batch_offsets) != 1:
                raise RuntimeError(
                    f"Desynchronized workers {batch_workers.tolist()}: "
                    f"{sorted(batch_offsets)}"
                )
            offset = batch_offsets.pop()

            for name, array in (
                ("observation", observation),
                ("reward", reward),
            ):
                failure = first_nonfinite_numpy(
                    name, np.asarray(array), env_ids, lookup, agents_per_worker, offset
                )
                if failure:
                    raise ProbeFailure(name)

            ids = torch.as_tensor(env_ids, device=device, dtype=torch.long)
            if offset == 0 or offset in reset_offsets:
                lstm_h[ids] = 0
                lstm_c[ids] = 0

            observation_t = torch.as_tensor(observation, device=device)
            state = {
                "reward": torch.as_tensor(reward, device=device),
                "done": torch.as_tensor(
                    np.asarray(terminal) | np.asarray(truncation), device=device
                ),
                "env_id": slice(int(env_ids[0]), int(env_ids[-1]) + 1),
                "mask": mask,
                "lstm_h": lstm_h[ids],
                "lstm_c": lstm_c[ids],
            }

            try:
                with torch.no_grad():
                    logits, value = policy.forward_eval(observation_t, state)
            except Exception as error:
                failure = {
                    "phase": "policy_forward_exception",
                    "offset_from_resample": int(offset),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "workers": batch_workers.tolist(),
                }
                raise ProbeFailure("policy forward") from error

            failure = first_nonfinite_tensor(
                "value", value, env_ids, lookup, agents_per_worker, offset
            )
            if failure:
                raise ProbeFailure("value")
            for head, head_logits in enumerate(logits):
                failure = first_nonfinite_tensor(
                    "logits",
                    head_logits,
                    env_ids,
                    lookup,
                    agents_per_worker,
                    offset,
                    action_head=head,
                )
                if failure:
                    raise ProbeFailure("logits")

            for state_name in ("lstm_h", "lstm_c"):
                failure = first_nonfinite_tensor(
                    state_name,
                    state[state_name],
                    env_ids,
                    lookup,
                    agents_per_worker,
                    offset,
                )
                if failure:
                    raise ProbeFailure(state_name)

            with torch.no_grad():
                action, logprob, entropy = pufferlib.pytorch.sample_logits(logits)

            for name, tensor in (
                ("logprob", logprob),
                ("entropy", entropy),
            ):
                failure = first_nonfinite_tensor(
                    name, tensor, env_ids, lookup, agents_per_worker, offset
                )
                if failure:
                    raise ProbeFailure(name)

            action_bins = vecenv.single_action_space.nvec.tolist()
            for head, bins in enumerate(action_bins):
                bad = torch.nonzero(
                    (action[:, head] < 0) | (action[:, head] >= bins),
                    as_tuple=False,
                )
                if bad.numel():
                    batch_agent = int(bad[0, 0].cpu())
                    failure = {
                        "phase": "action_out_of_range",
                        "offset_from_resample": int(offset),
                        "action_head": head,
                        "value": int(action[batch_agent, head].cpu()),
                        **location(
                            env_ids, batch_agent, lookup, agents_per_worker
                        ),
                    }
                    raise ProbeFailure("action")

            lstm_h[ids] = state["lstm_h"]
            lstm_c[ids] = state["lstm_c"]

            if args.target_start <= offset <= args.target_end:
                update_maxima(maxima, "observation", np.asarray(observation))
                update_maxima(maxima, "reward", np.asarray(reward))
                update_maxima(maxima, "value", value)
                update_maxima(maxima, "lstm_h", state["lstm_h"])
                update_maxima(maxima, "lstm_c", state["lstm_c"])
                update_maxima(maxima, "logprob", logprob)
                update_maxima(maxima, "entropy", entropy)
                for head, head_logits in enumerate(logits):
                    update_maxima(maxima, f"logits_head_{head}", head_logits)

            vecenv.send(action.cpu().numpy())

            minimum = int(offsets.min())
            if minimum >= next_progress:
                print(
                    f"minimum_offset={minimum}/{args.target_end} "
                    f"elapsed={time.time() - start_time:.1f}s",
                    flush=True,
                )
                next_progress += 25

    except ProbeFailure:
        pass
    finally:
        if vecenv is not None:
            vecenv.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "failed" if failure else "passed",
        "checkpoint": str(args.checkpoint),
        "first_seed": args.first_seed,
        "last_seed": args.first_seed + workers - 1,
        "action_seed": args.action_seed,
        "target_start": args.target_start,
        "target_end": args.target_end,
        "offsets_completed": offsets.tolist(),
        "target_window_max_abs": maxima,
        "first_failure": failure,
        "elapsed_seconds": time.time() - start_time,
    }
    output_path = args.output_dir / "failure_window_rollout_probe.json"
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"output = {output_path}")
    if failure:
        print("FAILURE_WINDOW_ROLLOUT_PROBE_FAILED")
        raise SystemExit(2)
    print("FAILURE_WINDOW_ROLLOUT_PROBE_PASSED")


if __name__ == "__main__":
    main()
