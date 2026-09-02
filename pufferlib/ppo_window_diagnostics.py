import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


FORMAL_RNG_START = 33700
FORMAL_RNG_END = 33770
FORMAL_CHECK_START = 33740
FORMAL_CHECK_END = 33770


def _cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def _iter_tensors(name, value):
    if torch.is_tensor(value):
        yield name, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_tensors(f"{name}.{key}", item)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _iter_tensors(f"{name}.{index}", item)


def _first_nonfinite(named_values):
    for name, value in named_values:
        for tensor_name, tensor in _iter_tensors(name, value):
            if not tensor.is_floating_point() and not tensor.is_complex():
                continue
            if bool(torch.isfinite(tensor).all()):
                continue
            bad = torch.nonzero(~torch.isfinite(tensor), as_tuple=False)[0]
            index = bad.detach().cpu().tolist()
            return {
                "tensor": tensor_name,
                "index": index,
                "value": repr(tensor[tuple(index)].detach().cpu().item()),
            }
    return None


def _atomic_torch_save(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_json_save(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


class PPOWindowDiagnostics:
    """Read-only PPO diagnostics armed only around the known failure window."""

    def __init__(self, trainer):
        self.trainer = trainer
        self.enabled = os.environ.get("PUFFER_DIAGNOSTICS_DISABLE") != "1"
        self.test_mode = os.environ.get("PUFFER_DIAGNOSTICS_TEST_MODE") == "1"

        if self.test_mode:
            self.rng_start, self.rng_end = 0, 3
            self.check_start, self.check_end = 1, 3
        else:
            self.rng_start, self.rng_end = FORMAL_RNG_START, FORMAL_RNG_END
            self.check_start, self.check_end = (
                FORMAL_CHECK_START,
                FORMAL_CHECK_END,
            )

        run_dir = Path(trainer.config["data_dir"]) / (
            f"{trainer.config['env']}_{trainer.logger.run_id}"
        )
        self.output_dir = run_dir / "numerical_diagnostics"
        self.active = False
        self.display_epoch = None
        self.pre_update_rng = None
        self.initial_values = None
        self.initial_ratio = None
        self.minibatch_indices = []
        self.last_good_state_path = self.output_dir / "last_good_epoch_state.pt"

        mode = "disabled" if not self.enabled else (
            "test" if self.test_mode else "formal"
        )
        print(
            "PPO_DIAGNOSTICS "
            f"mode={mode} rng={self.rng_start}-{self.rng_end} "
            f"checks={self.check_start}-{self.check_end}",
            flush=True,
        )

    def _rng_state(self, epoch, stage):
        state = {
            "epoch": int(epoch),
            "global_step": int(self.trainer.global_step),
            "stage": stage,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_cpu_rng_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        return state

    def _display_epoch_is_active(self, display_epoch):
        return self.enabled and self.check_start <= display_epoch <= self.check_end

    def capture_epoch_boundary(self):
        if not self.enabled:
            return

        epoch = int(self.trainer.epoch)
        if self.rng_start <= epoch <= self.rng_end:
            path = self.output_dir / f"rng_boundary_epoch_{epoch:06d}.pt"
            _atomic_torch_save(
                self._rng_state(epoch, "before_evaluate"), path
            )

        display_epoch = epoch + 1
        if self._display_epoch_is_active(display_epoch):
            state = {
                "epoch": epoch,
                "display_epoch": display_epoch,
                "global_step": int(self.trainer.global_step),
                "model_state_dict": _cpu_copy(
                    self.trainer.uncompiled_policy.state_dict()
                ),
                "optimizer_state_dict": _cpu_copy(
                    self.trainer.optimizer.state_dict()
                ),
                "scheduler_state_dict": copy.deepcopy(
                    self.trainer.scheduler.state_dict()
                ),
                "rng_state": self._rng_state(epoch, "before_evaluate"),
            }
            _atomic_torch_save(state, self.last_good_state_path)

    def begin_update(self):
        self.display_epoch = int(self.trainer.epoch) + 1
        self.active = self._display_epoch_is_active(self.display_epoch)
        if not self.active:
            return

        self.pre_update_rng = self._rng_state(
            self.trainer.epoch, "after_evaluate_before_train"
        )
        self.initial_values = self.trainer.values.detach().cpu().clone()
        self.initial_ratio = self.trainer.ratio.detach().cpu().clone()
        self.minibatch_indices = []

    def end_update(self):
        self.active = False
        self.display_epoch = None
        self.pre_update_rng = None
        self.initial_values = None
        self.initial_ratio = None
        self.minibatch_indices = []

    def check_rollout(self):
        if not self.active:
            return
        failure = _first_nonfinite(
            [
                ("rollout.observations", self.trainer.observations),
                ("rollout.logprobs", self.trainer.logprobs),
                ("rollout.rewards", self.trainer.rewards),
                ("rollout.terminals", self.trainer.terminals),
                ("rollout.truncations", self.trainer.truncations),
                ("rollout.values", self.trainer.values),
                ("rollout.masks", self.trainer.masks),
            ]
        )
        if failure is not None:
            self._fail("rollout_after_evaluate", None, failure)

    def record_minibatch(self, minibatch, indices):
        if not self.active:
            return
        self.minibatch_indices.append(
            {
                "minibatch": int(minibatch),
                "indices": indices.detach().cpu().clone(),
                "rng_after_sampling": self._rng_state(
                    self.trainer.epoch,
                    f"minibatch_{minibatch}_after_sampling",
                ),
            }
        )

    def check_pre_backward(self, minibatch, named_values):
        if not self.active:
            return
        failure = _first_nonfinite(named_values)
        if failure is not None:
            self._fail("before_backward", minibatch, failure)

    def check_gradients(self, minibatch, grad_norm):
        if not self.active:
            return
        failure = _first_nonfinite([("gradient_norm_before_clip", grad_norm)])
        if failure is None:
            return

        gradients = [
            (f"gradient.{name}", parameter.grad)
            for name, parameter in self.trainer.uncompiled_policy.named_parameters()
            if parameter.grad is not None
        ]
        gradient_failure = _first_nonfinite(gradients)
        self._fail(
            "after_backward_before_optimizer_step",
            minibatch,
            gradient_failure or failure,
        )

    def check_post_step(self, minibatch):
        if not self.active:
            return

        parameters = [
            (f"parameter.{name}", parameter)
            for name, parameter in self.trainer.uncompiled_policy.named_parameters()
        ]
        failure = _first_nonfinite(parameters)
        if failure is None:
            optimizer_values = []
            names = {
                parameter: name
                for name, parameter in self.trainer.uncompiled_policy.named_parameters()
            }
            for parameter, state in self.trainer.optimizer.state.items():
                parameter_name = names.get(parameter, "unknown")
                for state_name, value in state.items():
                    optimizer_values.append(
                        (f"optimizer.{parameter_name}.{state_name}", value)
                    )
            failure = _first_nonfinite(optimizer_values)

        if failure is not None:
            self._fail("after_optimizer_step", minibatch, failure)

    def _fail(self, stage, minibatch, tensor_failure):
        metadata = {
            "stage": stage,
            "internal_epoch": int(self.trainer.epoch),
            "display_epoch": int(self.display_epoch),
            "global_step": int(self.trainer.global_step),
            "minibatch": None if minibatch is None else int(minibatch),
            "tensor_failure": tensor_failure,
            "last_good_state": str(self.last_good_state_path),
        }
        _atomic_json_save(metadata, self.output_dir / "failure_marker.json")

        bundle = {
            "failure": metadata,
            "full_args": self.trainer.full_args,
            "pre_update_rng": self.pre_update_rng,
            "minibatch_indices": self.minibatch_indices,
            "initial_values": self.initial_values,
            "initial_ratio": self.initial_ratio,
            "failed_model_state_dict": _cpu_copy(
                self.trainer.uncompiled_policy.state_dict()
            ),
            "failed_optimizer_state_dict": _cpu_copy(
                self.trainer.optimizer.state_dict()
            ),
            "buffers": {
                "observations": _cpu_copy(self.trainer.observations),
                "actions": _cpu_copy(self.trainer.actions),
                "logprobs": _cpu_copy(self.trainer.logprobs),
                "rewards": _cpu_copy(self.trainer.rewards),
                "terminals": _cpu_copy(self.trainer.terminals),
                "truncations": _cpu_copy(self.trainer.truncations),
                "masks": _cpu_copy(self.trainer.masks),
            },
        }
        bundle_path = self.output_dir / (
            f"failure_epoch_{self.display_epoch:06d}_crash_bundle.pt"
        )
        _atomic_torch_save(bundle, bundle_path)
        metadata["crash_bundle"] = str(bundle_path)
        _atomic_json_save(metadata, self.output_dir / "failure_marker.json")
        raise FloatingPointError(
            "PPO numerical diagnostics detected a non-finite tensor: "
            f"{metadata}"
        )
