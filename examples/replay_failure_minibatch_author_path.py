import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch

from examples.analyze_failure_minibatch import (
    gradient_report,
    restore_rng,
    tensor_report,
)


class StopBeforeClip(Exception):
    pass


class ComponentBackwardComplete(Exception):
    pass


class FirstMinibatchProbe:
    def __init__(self, expected_indices, trainer=None, component=None):
        self.expected_indices = expected_indices.cpu().long()
        self.trainer = trainer
        self.component = component
        self.indices_match = None
        self.pre_backward = None
        self.component_result = None

    def record_minibatch(self, minibatch, indices):
        if minibatch != 0:
            raise RuntimeError("Replay reached an unexpected second minibatch")
        actual = indices.detach().cpu().long()
        self.indices_match = torch.equal(actual, self.expected_indices)
        if not self.indices_match:
            raise RuntimeError("Replayed minibatch indices do not match the crash bundle")

    def check_pre_backward(self, minibatch, named_values):
        if minibatch != 0:
            raise RuntimeError("Unexpected minibatch in pre-backward probe")
        self.pre_backward = {
            name: tensor_report(name, value) for name, value in named_values
        }
        if self.component is None:
            return

        values = dict(named_values)
        if self.component == "policy":
            selected_loss = values["policy_loss"]
        elif self.component == "value":
            selected_loss = (
                self.trainer.config["vf_coef"] * values["value_loss"]
            )
        elif self.component == "entropy":
            selected_loss = (
                -self.trainer.config["ent_coef"] * values["entropy_loss"]
            )
        elif self.component == "total":
            selected_loss = values["loss"]
        else:
            raise RuntimeError(f"Unknown component: {self.component}")

        self.trainer.optimizer.zero_grad(set_to_none=True)
        selected_loss.backward()
        self.component_result = {
            "loss": tensor_report(self.component, selected_loss),
            "raw_gradients_before_clip": gradient_report(
                self.trainer.uncompiled_policy
            ),
            "parameter_gradients": parameter_gradient_reports(
                self.trainer.uncompiled_policy
            ),
        }
        raise ComponentBackwardComplete

    def check_gradients(self, minibatch, grad_norm):
        raise RuntimeError("Gradient probe did not intercept clip_grad_norm_")

    def check_post_step(self, minibatch):
        raise RuntimeError("optimizer.step() must not run during replay")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--last-good", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--components",
        nargs="+",
        choices=("policy", "value", "entropy", "total"),
        default=None,
    )
    parser.add_argument("--trace-policy-boundary", action="store_true")
    parser.add_argument("--disable-compile", action="store_true")
    parser.add_argument("--disable-cudnn", action="store_true")
    parser.add_argument("--disable-cudnn-benchmark", action="store_true")
    parser.add_argument("--disable-cudnn-tf32", action="store_true")
    return parser.parse_args()


def copy_tensor(destination, source):
    destination.copy_(source.to(device=destination.device, dtype=destination.dtype))


def parameter_gradient_reports(policy):
    reports = {}
    for name, parameter in policy.named_parameters():
        if parameter.grad is None:
            reports[name] = {"has_gradient": False}
            continue
        report = tensor_report(f"gradient.{name}", parameter.grad)
        report["has_gradient"] = True
        reports[name] = report
    return reports


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    from pufferlib import pufferl

    bundle = torch.load(
        args.bundle, map_location="cpu", weights_only=False, mmap=True
    )
    last_good = torch.load(
        args.last_good, map_location="cpu", weights_only=False, mmap=True
    )
    config = copy.deepcopy(bundle["full_args"])
    config["wandb"] = False
    config["neptune"] = False
    config["load_id"] = None
    config["load_model_path"] = None
    config["train"]["device"] = "cuda"
    if args.disable_compile:
        config["train"]["compile"] = False
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False

    seed = config["train"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if config["rnn_name"] is None:
        config["env"]["uses_memory"] = False
        config["env"]["memory_size"] = 0
    else:
        config["env"]["uses_memory"] = True
        config["env"]["memory_size"] = config["train"]["rollout_horizon"]

    vecenv = pufferl.load_env("puffer_drive", config)
    policy = pufferl.load_policy(config, vecenv, "puffer_drive")
    train_config = dict(
        **config["train"], env="puffer_drive", eval=config.get("eval", {})
    )
    trainer = pufferl.PuffeRL(
        train_config, vecenv, policy, logger=None, full_args=config
    )
    trainer.diagnostics = None
    if args.disable_cudnn_benchmark:
        torch.backends.cudnn.benchmark = False
    if args.disable_cudnn_tf32:
        torch.backends.cudnn.allow_tf32 = False

    result = {
        "status": "started",
        "compile_mode": train_config["compile_mode"],
        "compile_enabled": train_config["compile"],
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "warmup": "one official evaluate/train update",
        "optimizer_step_on_failure_data": False,
    }
    print(
        f"BACKEND_CONFIG compile={train_config['compile']} "
        f"cudnn={torch.backends.cudnn.enabled} "
        f"benchmark={torch.backends.cudnn.benchmark} "
        f"allow_tf32={torch.backends.cudnn.allow_tf32}",
        flush=True,
    )

    try:
        # Match the official call order that created and reused the compiled
        # eval, policy, and action-distribution graphs from epoch one onward.
        torch.compiler.cudagraph_mark_step_begin()
        trainer.evaluate()
        torch.compiler.cudagraph_mark_step_begin()
        trainer.train()
        print("AUTHOR_PATH_WARMUP_PASSED", flush=True)

        original_sample_logits = None
        if args.trace_policy_boundary:
            import pufferlib.pytorch

            original_sample_logits = pufferlib.pytorch.sample_logits
            boundary = {}
            result["policy_boundary"] = boundary

            def capture_gradient(name):
                def hook(gradient):
                    boundary[name] = tensor_report(name, gradient)
                    return gradient

                return hook

            def traced_sample_logits(logits, action=None):
                outputs = original_sample_logits(logits, action=action)
                if action is not None:
                    heads = [logits] if torch.is_tensor(logits) else logits
                    for index, head in enumerate(heads):
                        boundary[f"logits_head_{index}_forward"] = (
                            tensor_report(f"logits_head_{index}", head)
                        )
                        head.register_hook(
                            capture_gradient(f"logits_head_{index}_gradient")
                        )
                    outputs[1].register_hook(
                        capture_gradient("newlogprob_gradient")
                    )
                return outputs

            pufferlib.pytorch.sample_logits = traced_sample_logits

        expected_indices = bundle["minibatch_indices"][0]["indices"]

        def restore_failure_state():
            trainer.uncompiled_policy.load_state_dict(
                last_good["model_state_dict"]
            )
            trainer.optimizer.load_state_dict(
                last_good["optimizer_state_dict"]
            )
            trainer.scheduler.load_state_dict(
                last_good["scheduler_state_dict"]
            )
            trainer.optimizer.zero_grad(set_to_none=True)
            trainer.epoch = int(bundle["failure"]["internal_epoch"])
            trainer.global_step = int(last_good["global_step"])

            for name in (
                "observations",
                "actions",
                "logprobs",
                "rewards",
                "terminals",
                "truncations",
                "masks",
            ):
                copy_tensor(getattr(trainer, name), bundle["buffers"][name])
            copy_tensor(trainer.values, bundle["initial_values"])

            model_equal = all(
                torch.equal(
                    value.detach().cpu(),
                    last_good["model_state_dict"][name],
                )
                for name, value in trainer.uncompiled_policy.state_dict().items()
            )
            if not model_equal:
                raise RuntimeError(
                    "Restored model does not exactly match last-good"
                )

        if args.components is not None:
            result["components"] = {}
            for component in args.components:
                restore_failure_state()
                probe = FirstMinibatchProbe(
                    expected_indices, trainer=trainer, component=component
                )
                trainer.diagnostics = probe
                restore_rng(bundle["pre_update_rng"], "cuda")
                try:
                    torch.compiler.cudagraph_mark_step_begin()
                    trainer.train()
                    raise RuntimeError(
                        "Component replay unexpectedly reached optimizer.step()"
                    )
                except ComponentBackwardComplete:
                    pass

                component_result = probe.component_result
                component_result["indices_match"] = probe.indices_match
                result["components"][component] = component_result
                print(
                    f"COMPONENT={component} "
                    f"INDICES_MATCH={probe.indices_match} "
                    f"LOSS_FINITE={component_result['loss']['finite']} "
                    f"GRADIENTS={component_result['raw_gradients_before_clip']}",
                    flush=True,
                )
                for name, report in component_result[
                    "parameter_gradients"
                ].items():
                    print(
                        f"PARAM_GRAD component={component} name={name} "
                        f"finite={report.get('finite')} "
                        f"nan_count={report.get('nan_count', 0)} "
                        f"max_abs={report.get('finite_max_abs')}",
                        flush=True,
                    )

            result["status"] = "components_complete"
            print("AUTHOR_PATH_STATUS = components_complete", flush=True)
        else:
            restore_failure_state()
            probe = FirstMinibatchProbe(expected_indices)
            trainer.diagnostics = probe
            restore_rng(bundle["pre_update_rng"], "cuda")

            original_clip = torch.nn.utils.clip_grad_norm_

            def intercept_clip(parameters, max_norm, *clip_args, **clip_kwargs):
                raw_gradients = gradient_report(trainer.uncompiled_policy)
                result["raw_gradients_before_clip"] = raw_gradients
                result["first_raw_nonfinite"] = raw_gradients[
                    "first_nonfinite"
                ]
                raise StopBeforeClip

            torch.nn.utils.clip_grad_norm_ = intercept_clip
            try:
                torch.compiler.cudagraph_mark_step_begin()
                trainer.train()
                raise RuntimeError(
                    "Replay unexpectedly completed an optimizer update"
                )
            except StopBeforeClip:
                result["indices_match"] = probe.indices_match
                result["pre_backward"] = probe.pre_backward
                result["status"] = (
                    "nonfinite" if result["first_raw_nonfinite"] else "finite"
                )
            finally:
                torch.nn.utils.clip_grad_norm_ = original_clip

            print("INDICES_MATCH =", result["indices_match"], flush=True)
            print(
                "RAW_GRADIENTS =",
                result["raw_gradients_before_clip"],
                flush=True,
            )
            print("AUTHOR_PATH_STATUS =", result["status"], flush=True)
        if args.trace_policy_boundary:
            print("POLICY_BOUNDARY =", result["policy_boundary"], flush=True)
            pufferlib.pytorch.sample_logits = original_sample_logits
    finally:
        try:
            vecenv.close()
        finally:
            trainer.utilization.stop()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("output =", args.output, flush=True)
    print("AUTHOR_PATH_REPLAY_PASSED", flush=True)


if __name__ == "__main__":
    main()
