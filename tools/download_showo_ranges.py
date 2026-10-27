from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import threading
import time
import uuid
import requests

root = Path.home() / "Show-o-Jittor"
parser = argparse.ArgumentParser(description="Server-only four-range Show-o 1 download")
parser.add_argument("--metadata", type=Path)
arguments = parser.parse_args()
metadata = dict(repo="showlab/show-o", revision="c2c6f548c00444432a97bc8fbbce7b2fd73fc9b5",
                filename="pytorch_model.safetensors", bytes=5792758016,
                sha256="67ca6506dc5a0d47bed8f4aa957c9b23153df0d33540b188ac65115cad7eaea4")
if arguments.metadata:
    metadata = json.loads(arguments.metadata.read_text(encoding="utf-8"))
repo = metadata["repo"]
if repo not in ("showlab/show-o", "showlab/show-o-512x512"):
    raise ValueError("This downloader is restricted to original Show-o 1 checkpoints")
if metadata["filename"] != "pytorch_model.safetensors":
    raise ValueError("Unexpected checkpoint filename")
revision, expected, total = metadata["revision"], metadata["sha256"], metadata["bytes"]
if (len(revision) != 40 or len(expected) != 64
        or any(c not in "0123456789abcdef" for c in revision + expected)
        or not isinstance(total, int) or total <= 0):
    raise ValueError("Invalid immutable checkpoint metadata")
url = "https://huggingface.co/" + repo + "/resolve/" + revision + "/pytorch_model.safetensors"
stage = Path("/tmp/xxt-showo-preflight-nuagft23")
if not stage.is_dir() or not (root / "official/models/modeling_showo.py").is_file():
    raise RuntimeError("Expected server project or staging directory missing; report this output.")
parts = stage / ("range4-" + expected)
parts.mkdir(exist_ok=True)
lock = (parts / "download.lock").open("a")
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise RuntimeError("Another four-part downloader is active; do not start a duplicate.")
target = root / "weights" / repo.split("/")[1] / "pytorch_model.safetensors"
target.parent.mkdir(parents=True, exist_ok=True)

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_config():
    from huggingface_hub import hf_hub_download

    source = Path(hf_hub_download(repo_id=repo, filename="config.json", revision=revision,
                                cache_dir=str(stage / "hub-cache")))
    body = source.read_bytes()
    config = json.loads(body)
    tokens = 1024 if repo.endswith("512x512") else 256
    if (config.get("num_vq_tokens") != tokens or config.get("vocab_size") != 58498
            or config.get("mask_token_id") != 58497 or config.get("w_clip_vit") is not False):
        raise ValueError("Checkpoint config differs from expected Show-o 1 layout")
    destination = target.parent / "config.json"
    if destination.exists():
        if destination.read_bytes() != body:
            raise RuntimeError("Existing config differs; preserved: " + str(destination))
    else:
        with destination.open("xb") as stream:
            stream.write(body)
    print("SHOWO_CONFIG_READY", destination, "image_tokens:", tokens, flush=True)
    return hashlib.sha256(body).hexdigest()


def record_verification():
    verified = dict(repo=repo, revision=revision, filename=metadata["filename"], bytes=total,
                    sha256=expected, config_sha256=config_sha, verified=True)
    path = target.parent / "verification.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != verified:
            raise RuntimeError("Existing verification record differs; preserved: " + str(path))
    else:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(verified, stream, indent=2)
    print("SHOWO_VERIFICATION_SAVED", path, flush=True)


print("SHOWO_CHECKPOINT:", repo, revision, flush=True)
config_sha = prepare_config()
if target.exists():
    print("CHECKING_EXISTING_WEIGHT", flush=True)
    if target.stat().st_size != total or sha256(target) != expected:
        raise RuntimeError("Existing target differs; preserved: " + str(target))
    record_verification()
    print("SHOWO_WEIGHT_SHA256_OK", target, flush=True)
    raise SystemExit(0)

ranges = [(total * i // 4, total * (i + 1) // 4) for i in range(4)]
paths = [parts / ("part-" + str(i) + ".bin") for i in range(4)]
sizes = [p.stat().st_size if p.exists() else 0 for p in paths]
if any(size > end - start for size, (start, end) in zip(sizes, ranges)):
    raise RuntimeError("Unexpected oversized part; all files preserved.")
if shutil.disk_usage(parts).free < total - sum(sizes) + 128 * 1024 * 1024:
    raise RuntimeError("Insufficient /tmp space.")
if shutil.disk_usage(target.parent).free < total + 128 * 1024 * 1024:
    raise RuntimeError("Insufficient project space.")
stop = threading.Event()
interrupted = threading.Event()

def on_interrupt(signum, frame):
    interrupted.set()
    stop.set()
    print("STOP_REQUESTED: preserving parts; waiting for active reads.", flush=True)

signal.signal(signal.SIGINT, on_interrupt)

def download(index):
    start, end = ranges[index]
    path = paths[index]
    with requests.Session() as session:
        while not stop.is_set():
            offset = path.stat().st_size if path.exists() else 0
            if start + offset == end:
                return
            last = min(end, start + offset + 16 * 1024 * 1024) - 1
            for attempt in range(4):
                if stop.is_set():
                    return
                offset = path.stat().st_size if path.exists() else 0
                first = start + offset
                if first > last:
                    break
                headers = {"Range": f"bytes={first}-{last}", "Accept-Encoding": "identity"}
                try:
                    with session.get(url, headers=headers, stream=True, timeout=(10, 30)) as response:
                        wanted = f"bytes {first}-{last}/{total}"
                        if response.status_code != 206 or response.headers.get("Content-Range") != wanted:
                            raise RuntimeError("Range rejected: HTTP=" + str(response.status_code)
                                               + " Content-Range=" + str(response.headers.get("Content-Range")))
                        if response.headers.get("Content-Encoding", "identity") != "identity":
                            raise RuntimeError("Unexpected encoded response.")
                        remaining = last - first + 1
                        with path.open("ab" if path.exists() else "xb") as stream:
                            for chunk in response.iter_content(chunk_size=256 * 1024):
                                if stop.is_set():
                                    return
                                if len(chunk) > remaining:
                                    raise RuntimeError("Oversized range response; stopped before extra write.")
                                stream.write(chunk)
                                remaining -= len(chunk)
                        if remaining:
                            raise RuntimeError("Short range response.")
                    break
                except (requests.RequestException, RuntimeError) as error:
                    print("RETRY part", index, "attempt", attempt + 1, type(error).__name__, flush=True)
                    if isinstance(error, RuntimeError):
                        print(str(error), flush=True)
                    if attempt == 3:
                        stop.set()
                        raise
                    if stop.wait(2 * (attempt + 1)):
                        return

baseline = sum(sizes)
started = time.monotonic()
print("SERVER_RANGE_DOWNLOAD: 4 connections; resumed bytes:", baseline, "of", total, flush=True)
pool = ThreadPoolExecutor(max_workers=4)
futures = [pool.submit(download, i) for i in range(4)]
try:
    while True:
        for future in futures:
            if future.done():
                future.result()
        current = sum(p.stat().st_size if p.exists() else 0 for p in paths)
        elapsed = max(time.monotonic() - started, 0.001)
        speed = (current - baseline) / elapsed
        eta = round((total - current) / speed) if speed > 0 else None
        print(json.dumps(dict(downloaded=current, total=total, percent=round(current / total * 100, 2),
                              MB_per_s=round(speed / 1e6, 3), eta_seconds=eta)), flush=True)
        if stop.is_set() or all(f.done() for f in futures):
            break
        stop.wait(10)
finally:
    stop.set()
    pool.shutdown(wait=True, cancel_futures=True)
if interrupted.is_set():
    print("DOWNLOAD_PAUSED: rerun this same block to resume.", flush=True)
    raise SystemExit(130)
for future in futures:
    future.result()
if [p.stat().st_size for p in paths] != [end - start for start, end in ranges]:
    raise RuntimeError("Incomplete parts preserved; rerun to resume.")

pending = target.parent / (".showo-assembled-" + uuid.uuid4().hex + ".partial")
digest = hashlib.sha256()
print("ASSEMBLING_AND_VERIFYING_SHA256", flush=True)
with pending.open("xb") as output:
    for path in paths:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                output.write(chunk)
                digest.update(chunk)
if interrupted.is_set():
    raise RuntimeError("Interrupted during assembly; parts and assembled file preserved.")
if pending.stat().st_size != total or digest.hexdigest() != expected:
    raise RuntimeError("SHA256 mismatch; all files preserved, do not load this weight.")
os.link(pending, target)
record_verification()
print("SHOWO_WEIGHT_SHA256_OK", target, flush=True)
print("SHA256:", expected, flush=True)
print("PARTS_PRESERVED:", parts, flush=True)
