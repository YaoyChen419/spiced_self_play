import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from pufferlib.ocean.drive import binding


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--failure-epoch", type=int, default=33763)
    parser.add_argument("--rollout-horizon", type=int, default=32)
    parser.add_argument("--resample-frequency", type=int, default=910)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--num-agents", type=int, default=1024)
    parser.add_argument("--num-maps", type=int, default=50000)
    parser.add_argument(
        "--map-dir",
        type=Path,
        default=Path("resources/drive/binaries/training"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/failure_window"),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.map_dir.is_dir():
        raise FileNotFoundError(args.map_dir)

    map_files = {}
    for path in args.map_dir.glob("map_*.bin"):
        map_id = int(path.stem.rsplit("_", 1)[1])
        map_files[map_id] = path

    if len(map_files) < args.num_maps:
        raise RuntimeError(
            f"Expected at least {args.num_maps} maps, found {len(map_files)}"
        )

    tick_start = (args.failure_epoch - 1) * args.rollout_horizon + 1
    tick_end = args.failure_epoch * args.rollout_horizon

    resample_number = tick_end // args.resample_frequency
    if resample_number < 1:
        raise RuntimeError("Failure precedes the first map resampling")

    last_resample_tick = resample_number * args.resample_frequency

    # _resample_count starts at zero. The first resampling therefore
    # reuses the worker's base seed.
    resample_seed_offset = resample_number - 1
    first_active_seed = args.base_seed + resample_seed_offset
    last_active_seed = first_active_seed + args.num_workers - 1

    if (
        args.failure_epoch == 33763
        and args.rollout_horizon == 32
        and args.resample_frequency == 910
        and args.base_seed == 42
        and args.num_workers == 16
    ):
        assert resample_number == 1187
        assert first_active_seed == 1228
        assert last_active_seed == 1243

    rows = []

    for worker_idx in range(args.num_workers):
        worker_seed = args.base_seed + worker_idx
        resample_seed = worker_seed + resample_seed_offset

        agent_offsets, map_ids, num_envs = binding.shared(
            seed=resample_seed,
            map_dir=str(args.map_dir),
            num_agents=args.num_agents,
            num_maps=args.num_maps,
            init_mode=0,            # create_all_valid
            control_mode=1,         # control_agents
            sdc_controller=1,       # policy
            non_sdc_controller=1,   # policy
            non_vehicle_controller=1,
            init_steps=0,
            goal_behavior=0,
            goal_target_distance=30.0,
            goal_speed=100.0,
            max_controlled_agents=32,
        )

        agent_offsets = np.asarray(agent_offsets, dtype=np.int64)
        map_ids = np.asarray(map_ids, dtype=np.int64)
        num_envs = int(num_envs)

        if len(map_ids) != num_envs:
            raise RuntimeError(
                f"worker {worker_idx}: map_ids={len(map_ids)}, num_envs={num_envs}"
            )

        if len(agent_offsets) != num_envs + 1:
            raise RuntimeError(
                f"worker {worker_idx}: offsets={len(agent_offsets)}, "
                f"expected={num_envs + 1}"
            )

        for map_slot, map_id_value in enumerate(map_ids):
            map_id = int(map_id_value)
            map_path = map_files.get(map_id)

            if map_path is None:
                raise FileNotFoundError(f"Missing binary for map_id={map_id}")

            agent_start = int(agent_offsets[map_slot])
            agent_stop = int(agent_offsets[map_slot + 1])

            rows.append(
                {
                    "worker_idx": worker_idx,
                    "worker_seed": worker_seed,
                    "resample_number": resample_number,
                    "resample_seed": resample_seed,
                    "map_slot": map_slot,
                    "map_id": map_id,
                    "agent_start": agent_start,
                    "agent_stop": agent_stop,
                    "num_agents": agent_stop - agent_start,
                    "map_path": str(map_path),
                    "file_size": map_path.stat().st_size,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_path = args.output_dir / "failure_window_map_assignments.csv"
    fields = list(rows[0].keys())

    with all_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    by_map = defaultdict(list)
    for row in rows:
        by_map[row["map_id"]].append(row)

    unique_rows = []
    for map_id in sorted(by_map):
        occurrences = by_map[map_id]
        unique_rows.append(
            {
                "map_id": map_id,
                "occurrences": len(occurrences),
                "total_agents": sum(row["num_agents"] for row in occurrences),
                "workers": ";".join(
                    str(value)
                    for value in sorted(
                        {row["worker_idx"] for row in occurrences}
                    )
                ),
                "resample_seeds": ";".join(
                    str(value)
                    for value in sorted(
                        {row["resample_seed"] for row in occurrences}
                    )
                ),
                "map_path": occurrences[0]["map_path"],
                "file_size": occurrences[0]["file_size"],
            }
        )

    unique_path = args.output_dir / "failure_window_unique_maps.csv"
    unique_fields = list(unique_rows[0].keys())

    with unique_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=unique_fields)
        writer.writeheader()
        writer.writerows(unique_rows)

    summary = {
        "failure_epoch": args.failure_epoch,
        "rollout_horizon": args.rollout_horizon,
        "rollout_tick_start": tick_start,
        "rollout_tick_end": tick_end,
        "resample_frequency": args.resample_frequency,
        "resample_number": resample_number,
        "last_resample_tick": last_resample_tick,
        "steps_from_resample_to_rollout_start": tick_start - last_resample_tick,
        "steps_from_resample_to_rollout_end": tick_end - last_resample_tick,
        "first_active_seed": first_active_seed,
        "last_active_seed": last_active_seed,
        "num_workers": args.num_workers,
        "assignment_rows": len(rows),
        "unique_maps": len(unique_rows),
    }

    summary_path = args.output_dir / "failure_window_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"assignments = {all_path}")
    print(f"unique maps = {unique_path}")
    print(f"summary = {summary_path}")
    print("FAILURE_WINDOW_RECONSTRUCTION_PASSED")


if __name__ == "__main__":
    main()
