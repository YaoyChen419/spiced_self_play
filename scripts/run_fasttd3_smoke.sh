#!/usr/bin/env bash
# Run the normal trainer; bound wall time and collect reproducible diagnostics.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
duration=${1:-10m}
if (( $# )); then shift; fi
mkdir -p experiments
run_dir=$(mktemp -d "$PWD/experiments/fasttd3_smoke_XXXXXXXX")
gpu_pid=
train_pid=
cleanup() {
    status=$?
    trap - EXIT INT TERM
    # This session belongs exclusively to this test, including its workers.
    if [[ -n "$train_pid" ]]; then
        kill -TERM -- "-$train_pid" 2>/dev/null || true
    fi
    if [[ -n "$gpu_pid" ]]; then
        kill "$gpu_pid" 2>/dev/null || true
        wait "$gpu_pid" 2>/dev/null || true
    fi
    printf '%s\n' "$status" > "$run_dir/exit_code.txt"
    printf '\nExit code: %s\nLog directory: %s\n' "$status" "$run_dir"
}
trap cleanup EXIT
trap '[[ -z "$train_pid" ]] || kill -INT "$train_pid" 2>/dev/null || true' INT TERM
git rev-parse HEAD > "$run_dir/commit.txt"
sha256sum pufferlib/fasttd3.py pufferlib/fasttd3_train.py pufferlib/fasttd3_replay.py \
    pufferlib/pufferl.py pufferlib/config/ocean/drive.ini > "$run_dir/source_sha256.txt"
python -c 'import json, torch; print(json.dumps(dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), vram_bytes=torch.cuda.get_device_properties(0).total_memory)))' \
    | tee "$run_dir/runtime.json"
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv -l 5 > "$run_dir/gpu.csv" &
gpu_pid=$!
export WANDB_MODE=${WANDB_MODE:-online}
# foreground delivers SIGINT to the learner, allowing its own env.close() to
# stop workers. A private session makes cleanup safe if the grace period expires.
setsid timeout --foreground --signal=INT --kill-after=120s "$duration" \
    python -u -m pufferlib.pufferl train puffer_drive \
    --algorithm fasttd3 --env.action-type continuous --train.device cuda \
    --train.data-dir "$run_dir" --wandb --wandb-project spiced-fasttd3 \
    --wandb-group fasttd3-smoke --tag smoke "$@" \
    > "$run_dir/train.log" 2>&1 &
train_pid=$!
printf 'Training PID: %s\nFollow progress: tail -f "%s/train.log"\n' "$train_pid" "$run_dir"
set +e
wait "$train_pid"
status=$?
# A shell signal can interrupt wait before the training process finishes.
while kill -0 "$train_pid" 2>/dev/null; do
    wait "$train_pid"
    status=$?
done
exit "$status"
