import argparse
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    return parser.parse_args()


def latest_model(directory):
    models = sorted(directory.glob("model_puffer_drive_*.pt"))
    if not models:
        raise FileNotFoundError(f"No model checkpoints in {directory}")
    return models[-1]


def compare_exact(name, left, right, path="root"):
    if torch.is_tensor(left):
        if not torch.is_tensor(right):
            raise AssertionError(f"{name}:{path} type mismatch")
        if left.dtype != right.dtype or left.shape != right.shape:
            raise AssertionError(f"{name}:{path} tensor metadata mismatch")
        if not torch.equal(left.cpu(), right.cpu()):
            difference = (left.cpu() - right.cpu()).abs().max().item()
            raise AssertionError(
                f"{name}:{path} tensor mismatch; max_abs_diff={difference}"
            )
        return

    if isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise AssertionError(f"{name}:{path} dictionary mismatch")
        for key in left:
            compare_exact(name, left[key], right[key], f"{path}.{key}")
        return

    if isinstance(left, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise AssertionError(f"{name}:{path} sequence mismatch")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            compare_exact(
                name, left_item, right_item, f"{path}.{index}"
            )
        return

    if left != right:
        raise AssertionError(f"{name}:{path} {left!r} != {right!r}")


def main():
    args = parse_args()
    control_model_path = latest_model(args.control_dir)
    diagnostic_model_path = latest_model(args.diagnostic_dir)

    control_model = torch.load(
        control_model_path, map_location="cpu", weights_only=False
    )["model_state_dict"]
    diagnostic_model = torch.load(
        diagnostic_model_path, map_location="cpu", weights_only=False
    )["model_state_dict"]
    compare_exact("model", control_model, diagnostic_model)

    control_state = torch.load(
        args.control_dir / "trainer_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    diagnostic_state = torch.load(
        args.diagnostic_dir / "trainer_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    for key in ("update", "global_step", "agent_step", "model_name"):
        compare_exact(
            "trainer_state", control_state[key], diagnostic_state[key], key
        )
    compare_exact(
        "optimizer",
        control_state["optimizer_state_dict"],
        diagnostic_state["optimizer_state_dict"],
    )

    print(f"control_model = {control_model_path}")
    print(f"diagnostic_model = {diagnostic_model_path}")
    print("MODEL_EXACT_MATCH = True")
    print("OPTIMIZER_EXACT_MATCH = True")
    print("PPO_DIAGNOSTIC_EQUIVALENCE_PASSED")


if __name__ == "__main__":
    main()
