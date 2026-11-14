#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-1}"
case "$gpu" in 0|1|2) ;; *) echo 'GPU must be 0, 1 or 2'; exit 1 ;; esac
root="$HOME/Show-o-Jittor"
python="$HOME/envs/showo-jittor/bin/python"
cache="$HOME/tmp/jittor-cache/native-magvit-20261114-140512-206070038"
cd "$root"
test -x "$python"
test -d "$cache"
test -f tools/native_pipeline.py
printf '%s  %s\n' \
  '36c0aee447bd2c8e14e6bdc31cbeb344b12398708459c0cbf5744b9b2765acee' \
  'tools/validate_full_gradients.py' | sha256sum -c -
if pgrep -u "$(id -u)" -af '[p]ython.*tools/(validate_full_gradients[.]py|run_native_matrix[.]py|native_pipeline[.]py|infer_full_jittor[.]py --native-pipeline)'; then
    echo 'An existing validation process is still running; no duplicate started.'
    exit 1
fi
used=$(timeout 10 nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
case "$used" in ''|*[!0-9]*) echo 'Could not read GPU memory'; exit 1 ;; esac
if [ "$used" -gt 1024 ]; then
    echo "GPU $gpu occupied: $used MiB; no task started."
    exit 1
fi
export CUDA_VISIBLE_DEVICES="$gpu" NVIDIA_TF32_OVERRIDE=0
export CUDA_HOME="$root/deps/cuda-jittor"
export nvcc_path="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export JITTOR_HOME="$cache" disable_lock=1 DISABLE_MULTIPROCESSING=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TMPDIR="$HOME/tmp" TMP="$HOME/tmp" TEMP="$HOME/tmp"
mkdir -p "$HOME/tmp/showo-logs"
log="$HOME/tmp/showo-logs/full-gradient-512-$(date +%Y%m%d-%H%M%S-%N).log"
nohup "$python" -u "$root/tools/validate_full_gradients.py" --gpu "$gpu" > "$log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$log.pid"
printf '%s\n' "$log" > "$HOME/tmp/showo-logs/full-gradient-512.latest"
printf 'STARTED PID %s\nLOG %s\n' "$pid" "$log"
printf 'Full FP32 MMU/T2I backward, no weight updates. About 12 GB saved gradients; requires 18 GiB free disk.\n'
printf 'Initial estimate 5-20 minutes; backward compilation or disk I/O may take longer.\n'
