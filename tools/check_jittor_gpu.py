"""Prepare a private CUDA/cuDNN 8 view and test Jittor on the server."""

import argparse
import importlib.metadata
import os
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime


GPU_CHECK = """
import jittor as jt
import numpy as np

assert jt.has_cuda, "Jittor did not detect CUDA"
jt.flags.use_cuda = 1
x = jt.randn((16, 16))
loss = (x @ x.transpose()).sqr().mean()
grad = jt.grad(loss, x)
assert np.isfinite(grad.numpy()).all(), "Non-finite matrix gradient"
print("JITTOR_GPU_BACKWARD_OK", "use_cuda=", jt.flags.use_cuda, flush=True)

image = jt.randn((1, 1, 8, 8))
conv = jt.nn.Conv(1, 2, 3)
output = conv(image)
grad = jt.grad(output.sqr().mean(), image)
assert np.isfinite(output.numpy()).all(), "Non-finite convolution output"
assert np.isfinite(grad.numpy()).all(), "Non-finite convolution gradient"
print("JITTOR_GPU_CONV_BACKWARD_OK", flush=True)
"""


def link(source, destination):
    if destination.is_symlink():
        if destination.readlink() == source:
            return
        raise RuntimeError(f"Conflicting existing link: {destination}")
    if destination.exists():
        raise RuntimeError(f"Will not overwrite existing file: {destination}")
    destination.symlink_to(source, target_is_directory=source.is_dir())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()
    expected_env = Path.home() / "envs/showo-jittor"
    if Path(sys.prefix).resolve() != expected_env.resolve():
        raise RuntimeError(f"Activate {expected_env} before running this script")

    project = Path.home() / "Show-o-Jittor"
    toolkit = Path("/usr/local/cuda-12.4")
    cudnn = project / "deps/cudnn8/nvidia/cudnn"
    view = project / "deps/cuda-jittor"
    for required in (
        toolkit / "bin/nvcc",
        toolkit / "lib64/libcudart.so",
        cudnn / "include/cudnn.h",
        cudnn / "lib/libcudnn.so.8",
        cudnn / "lib/libcudnn_ops_infer.so.8",
        cudnn / "lib/libcudnn_ops_train.so.8",
        cudnn / "lib/libcudnn_cnn_infer.so.8",
        cudnn / "lib/libcudnn_cnn_train.so.8",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    view.mkdir(parents=True, exist_ok=True)
    if view.is_symlink():
        raise RuntimeError(f"Private CUDA directory must not be a symlink: {view}")
    for name in ("bin", "include", "lib64"):
        directory = view / name
        if directory.is_symlink():
            raise RuntimeError(f"Private directory must not be a symlink: {directory}")
        directory.mkdir(exist_ok=True)

    for item in toolkit.iterdir():
        if item.name not in ("bin", "include", "lib64"):
            link(item, view / item.name)
    for name in ("bin", "include", "lib64"):
        for item in (toolkit / name).iterdir():
            if item.name == "nvcc" or "cudnn" in item.name:
                continue
            link(item, view / name / item.name)
    for item in (cudnn / "include").glob("cudnn*.h"):
        link(item, view / "include" / item.name)
    for item in (cudnn / "lib").glob("libcudnn*.so.8"):
        link(item, view / "lib64" / item.name)
        link(item, view / "lib64" / item.name.removesuffix(".8"))

    # Jittor resolves nvcc symlinks; a wrapper preserves the private root.
    wrapper = view / "bin/nvcc"
    wrapper_text = f'#!/bin/sh\nexec {shlex.quote(str(toolkit / "bin/nvcc"))} "$@"\n'
    if wrapper.is_symlink():
        raise RuntimeError(f"Will not replace nvcc symlink: {wrapper}")
    if wrapper.exists():
        if wrapper.read_text() != wrapper_text:
            raise RuntimeError(f"Will not overwrite existing wrapper: {wrapper}")
    else:
        wrapper.write_text(wrapper_text)
    wrapper.chmod(0o755)

    env = os.environ.copy()
    env.update(
        CUDA_HOME=str(view),
        nvcc_path=str(wrapper),
        JITTOR_HOME="/tmp/xxt-showo-jittor",
        DISABLE_MULTIPROCESSING="1",
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        PYTHONUNBUFFERED="1",
    )
    env["PATH"] = str(view / "bin") + os.pathsep + env.get("PATH", "")
    old_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = str(view / "lib64") + (":" + old_ld if old_ld else "")

    logs = project / "logs"
    logs.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    logfile = logs / f"jittor-gpu-{stamp}.log"
    with logfile.open("x", encoding="utf-8") as log:
        header = (
            f"Python: {sys.executable}\n"
            f"Jittor: {importlib.metadata.version('jittor')}\n"
            f"PyTorch cuDNN package: {importlib.metadata.version('nvidia-cudnn-cu12')}\n"
            f"Private cuDNN directory: {cudnn}\n"
            f"Physical GPU: {args.gpu}\nPrivate CUDA: {view}\nLog: {logfile}\n"
        )
        print(header, flush=True)
        log.write(header)
        log.flush()
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", GPU_CHECK],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
        log.write(f"\nExit code: {code}\n")
    print(f"Exit code: {code}; log: {logfile}", flush=True)
    return code if code >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
