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
printf '%s  %s\n' \
  'a883e3783b79a2c6ea436206cf5d5c93dc7998b2ebda91cf395c4a991a5b780a' \
  'tools/run_native_matrix.py' | sha256sum -c -

if pgrep -u "$(id -u)" -af '[p]ython.*tools/(run_native_matrix[.]py|native_pipeline[.]py|infer_full_jittor[.]py --native-pipeline)'; then
    echo 'An existing native pipeline is still running; no duplicate queue started.'
    exit 1
fi
used=$(timeout 10 nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
case "$used" in ''|*[!0-9]*) echo 'Could not read GPU memory'; exit 1 ;; esac
if [ "$used" -gt 1024 ]; then
    echo "GPU $gpu occupied: $used MiB; no queue started."
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
log="$HOME/tmp/showo-logs/native-multi-512-$(date +%Y%m%d-%H%M%S-%N).log"
nohup "$python" -u "$root/tools/run_native_matrix.py" --gpu "$gpu" > "$log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$log.pid"
printf '%s\n' "$log" > "$HOME/tmp/showo-logs/native-multi-512.latest"
printf 'STARTED PID %s\nLOG %s\n' "$pid" "$log"
printf '12 T2I cases; the same dog MMU input is repeated. Estimated 90-150 minutes, possibly longer.\n'
