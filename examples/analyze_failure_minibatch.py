import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--last-good", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--mode", choices=("eager", "compiled"), default="eager")
    parser.add_argument(
        "--components",
        nargs="+",
        choices=("policy", "value", "entropy", "total"),
        default=("policy", "value", "entropy", "total"),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def first_nonfinite(name, tensor):
    if not tensor.is_floating_point() and not tensor.is_complex():
        return None
    bad = ~torch.isfinite(tensor)
    if not bool(bad.any()):
        return None
    index = torch.nonzero(bad, as_tuple=False)[0].detach().cpu().tolist()
    value = tensor[tuple(index)].detach().cpu().item()
    return {"tensor": name, "index": index, "value": repr(value)}


def tensor_report(name, tensor):
    detached = tensor.detach()
    floating = detached.is_floating_point() or detached.is_complex()
    report = {
        "name": name,
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
    }
    if not floating:
        return report

    finite = torch.isfinite(detached)
    report.update(
        finite=bool(finite.all()),
        nan_count=int(torch.isnan(detached).sum().item()),
        posinf_count=int(torch.isposinf(detached).sum().item()),
        neginf_count=int(torch.isneginf(detached).sum().item()),
    )
    if bool(finite.any()):
        finite_values = detached[finite]
        report["finite_max_abs"] = float(finite_values.abs().max().item())
    return report


def gradient_report(policy):
    first = None
    nan_count = 0
    posinf_count = 0
    neginf_count = 0
    norms = []
    parameters_with_grad = 0

    for name, parameter in policy.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        parameters_with_grad += 1
        detached = gradient.detach()
        nan_count += int(torch.isnan(detached).sum().item())
        posinf_count += int(torch.isposinf(detached).sum().item())
        neginf_count += int(torch.isneginf(detached).sum().item())
        if first is None:
            first = first_nonfinite(f"gradient.{name}", detached)
        norms.append(detached.float().norm(2))

    total_norm = None
    if norms:
        total_norm = float(torch.stack(norms).norm(2).detach().cpu().item())

    return {
        "parameters_with_grad": parameters_with_grad,
        "total_norm": total_norm,
        "nan_count": nan_count,
        "posinf_count": posinf_count,
        "neginf_count": neginf_count,
        "first_nonfinite": first,
    }


def restore_rng(state, device):
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_cpu_rng_state"])
    if device == "cuda":
        torch.cuda.set_rng_state_all(state["torch_cuda_rng_state_all"])


def build_policy(full_args, model_state, device, mode):
    from pufferlib import pufferl

    config = copy.deepcopy(full_args)
    config["wandb"] = False
    config["neptune"] = False
    config["load_id"] = None
    config["load_model_path"] = None
    config["train"]["device"] = device
    config["vec"] = {"backend": "PufferEnv", "num_envs": 1}
    config["env"]["num_agents"] = 32
    config["env"]["num_maps"] = 1

    environment = pufferl.load_env("puffer_drive", config)
    policy = pufferl.load_policy(config, environment, "puffer_drive")
    environment.close()
    policy.load_state_dict(model_state)
    policy.train()

    runner = policy
    if mode == "compiled":
        runner = torch.compile(policy, mode=config["train"]["compile_mode"])
    return policy, runner, config


def compute_advantages(bundle, config, device):
    from pufferlib import pufferl

    values = bundle["initial_values"].detach().clone().to(device)
    rewards = bundle["buffers"]["rewards"].detach().clone().to(device)
    terminals = bundle["buffers"]["terminals"].detach().clone().to(device)
    # PuffeRL resets the rollout importance ratios at the start of every
    # train() call, before computing the first minibatch advantages.
    ratio = torch.ones_like(values)
    advantages = torch.zeros_like(values)

    previous = pufferl.ADVANTAGE_CUDA
    pufferl.ADVANTAGE_CUDA = device == "cuda"
    try:
        advantages = pufferl.compute_puff_advantage(
            values,
            rewards,
            terminals,
            ratio,
            advantages,
            config["train"]["gamma"],
            config["train"]["gae_lambda"],
            config["train"]["vtrace_rho_clip"],
            config["train"]["vtrace_c_clip"],
        )
    finally:
        pufferl.ADVANTAGE_CUDA = previous
    return advantages


def prepare_minibatch(bundle, advantages, device):
    indices = bundle["minibatch_indices"][0]["indices"].long()
    buffers = bundle["buffers"]
    values = bundle["initial_values"]

    def selected(name, dtype=None):
        tensor = buffers[name][indices]
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor.to(device)

    result = {
        "observations": selected("observations"),
        "actions": selected("actions"),
        "logprobs": selected("logprobs"),
        "rewards": selected("rewards"),
        "terminals": selected("terminals"),
        "truncations": selected("truncations"),
        "masks": selected("masks"),
        "values": values[indices].to(device),
        "advantages": advantages[indices.to(advantages.device)].to(device),
    }
    result["returns"] = result["advantages"] + result["values"]
    return result


def compute_losses(runner, minibatch, train_config):
    import pufferlib.pytorch

    observations = minibatch["observations"]
    actions = minibatch["actions"]
    old_logprobs = minibatch["logprobs"]
    old_values = minibatch["values"]
    returns = minibatch["returns"]
    masks = minibatch["masks"]

    state = {"action": actions, "lstm_h": None, "lstm_c": None}
    logits, newvalue = runner(observations, state)
    _, newlogprob, entropy = pufferlib.pytorch.sample_logits(
        logits, action=actions
    )

    newlogprob = newlogprob.reshape(old_logprobs.shape)
    logratio = newlogprob - old_logprobs
    ratio = logratio.exp()

    advantage = minibatch["advantages"]
    valid_advantage = advantage[masks == 1]
    if valid_advantage.numel() > 0:
        advantage_mean = valid_advantage.mean()
        advantage_std = valid_advantage.std() + 1e-8
    else:
        advantage_mean = advantage.mean()
        advantage_std = advantage.std() + 1e-8
    normalized_advantage = (advantage - advantage_mean) / advantage_std
    normalized_advantage = normalized_advantage * masks

    valid_count = masks.sum().clamp(min=1)
    clip_coef = train_config["clip_coef"]
    pg_loss1 = -normalized_advantage * ratio
    pg_loss2 = -normalized_advantage * torch.clamp(
        ratio, 1 - clip_coef, 1 + clip_coef
    )
    policy_loss = (torch.max(pg_loss1, pg_loss2) * masks).sum() / valid_count

    newvalue = newvalue.view(returns.shape)
    vf_clip = train_config["vf_clip_coef"]
    clipped_value = old_values + torch.clamp(
        newvalue - old_values, -vf_clip, vf_clip
    )
    value_loss_unclipped = (newvalue - returns) ** 2
    value_loss_clipped = (clipped_value - returns) ** 2
    value_loss = 0.5 * (
        torch.max(value_loss_unclipped, value_loss_clipped) * masks
    ).sum() / valid_count

    entropy_loss = (entropy.reshape(masks.shape) * masks).sum() / valid_count
    components = {
        "policy": policy_loss,
        "value": train_config["vf_coef"] * value_loss,
        "entropy": -train_config["ent_coef"] * entropy_loss,
    }
    components["total"] = (
        components["policy"]
        + components["value"]
        + components["entropy"]
    )

    intermediates = {
        "newvalue": newvalue,
        "newlogprob": newlogprob,
        "entropy": entropy,
        "logratio": logratio,
        "ratio": ratio,
        "normalized_advantage": normalized_advantage,
    }
    for index, head in enumerate(logits):
        intermediates[f"logits_head_{index}"] = head
    return components, intermediates


def run_component(
    name,
    policy,
    runner,
    minibatch,
    train_config,
    rng_state,
    device,
):
    restore_rng(rng_state, device)
    policy.zero_grad(set_to_none=True)
    components, intermediates = compute_losses(
        runner, minibatch, train_config
    )
    loss = components[name]
    loss.backward()

    before_clip = gradient_report(policy)
    result = {
        "loss": float(loss.detach().cpu().item()),
        "loss_finite": bool(torch.isfinite(loss).item()),
        "gradients_before_clip": before_clip,
    }

    if name == "total":
        clip_return = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), train_config["max_grad_norm"]
        )
        result["clip_return"] = tensor_report(
            "clip_return", clip_return
        )
        result["gradients_after_clip"] = gradient_report(policy)
        result["intermediates"] = {
            key: tensor_report(key, value)
            for key, value in intermediates.items()
        }
    return result


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is available")

    bundle = torch.load(
        args.bundle,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    last_good = torch.load(
        args.last_good,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )

    model_equal = all(
        torch.equal(
            bundle["failed_model_state_dict"][key],
            last_good["model_state_dict"][key],
        )
        for key in last_good["model_state_dict"]
    )
    if not model_equal:
        raise RuntimeError("Failure model differs from last-good model")

    full_args = bundle["full_args"]
    train_config = full_args["train"]
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.deterministic = train_config["torch_deterministic"]
    torch.backends.cudnn.benchmark = True

    policy, runner, runtime_config = build_policy(
        full_args,
        bundle["failed_model_state_dict"],
        args.device,
        args.mode,
    )
    advantages = compute_advantages(bundle, runtime_config, args.device)
    advantage_report = tensor_report("advantages", advantages)
    if not advantage_report["finite"]:
        raise RuntimeError(f"Non-finite saved-rollout advantage: {advantage_report}")

    minibatch = prepare_minibatch(bundle, advantages, args.device)
    input_reports = {
        name: tensor_report(name, value)
        for name, value in minibatch.items()
    }
    results = {}
    for component in args.components:
        print(f"RUNNING_COMPONENT={component}", flush=True)
        results[component] = run_component(
            component,
            policy,
            runner,
            minibatch,
            train_config,
            bundle["pre_update_rng"],
            args.device,
        )
        print(
            f"{component}: loss={results[component]['loss']:.9g} "
            f"preclip={results[component]['gradients_before_clip']}",
            flush=True,
        )

    output = {
        "bundle": str(args.bundle),
        "device": args.device,
        "mode": args.mode,
        "failure": bundle["failure"],
        "checkpoint_model_exact": model_equal,
        "advantage": advantage_report,
        "inputs": input_reports,
        "components": results,
    }
    output_path = args.output or Path(
        f"results/failure_minibatch_{args.device}_{args.mode}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"output = {output_path}")
    print("FAILURE_MINIBATCH_ANALYSIS_PASSED")


if __name__ == "__main__":
    main()
