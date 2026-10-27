from collections import Counter
from datetime import datetime
from pathlib import Path
import hashlib
import json
import math
import shutil
import struct
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

root = Path.home() / "Show-o-Jittor"
preflight = root / "results/official-preflight-20261026-214218-476683/official-preflight.json"
report = json.loads(preflight.read_text(encoding="utf-8"))
stage = Path(report["stage"])
if not stage.is_dir() or stage.parent != Path("/tmp") or not stage.name.startswith("xxt-showo-preflight-"):
    raise RuntimeError("Original server staging directory unavailable; report this output.")
show_rev = "c2c6f548c00444432a97bc8fbbce7b2fd73fc9b5"
phi_rev = "77aa61eeac94fbf33d492b9f2744c98b42d5b5eb"
weight_sha = "67ca6506dc5a0d47bed8f4aa957c9b23153df0d33540b188ac65115cad7eaea4"
weight_size = 5792758016
for repo, revision in (("showlab/show-o", show_rev), ("microsoft/phi-1_5", phi_rev)):
    if report["models"][repo]["revision"] != revision:
        raise RuntimeError("Preflight revision differs: " + repo)

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def copy_checked(source, target, expected=None):
    expected = expected or sha256(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(target) != expected:
            raise RuntimeError("Existing file differs; preserved: " + str(target))
        print("REUSED:", target, flush=True)
    else:
        print("COPYING:", target, flush=True)
        with source.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        if sha256(target) != expected:
            raise RuntimeError("Copy verification failed; preserved: " + str(target))

show_dir = root / "weights/show-o"
weight = show_dir / "pytorch_model.safetensors"
if weight.exists():
    print("CHECKING_EXISTING_WEIGHT:", weight, flush=True)
    if weight.stat().st_size != weight_size or sha256(weight) != weight_sha:
        raise RuntimeError("Existing weight differs; preserved. Report this output.")
    patterns = ["config.json"]
else:
    for folder in (stage, root):
        if shutil.disk_usage(folder).free < weight_size + 256 * 1024 * 1024:
            raise RuntimeError("Insufficient free space: " + str(folder))
    patterns = ["config.json", "pytorch_model.safetensors"]
print("DOWNLOADING_SHOWO:", show_rev, "SERVER_STAGE:", stage, flush=True)
downloaded = Path(snapshot_download(
    repo_id="showlab/show-o", revision=show_rev, allow_patterns=patterns,
    cache_dir=str(stage / "hub-cache"), local_dir=str(stage / "showo-files"), max_workers=1,
))
config_sha = report["models"]["showlab/show-o"]["config_sha256"]
if sha256(downloaded / "config.json") != config_sha:
    raise RuntimeError("Show-o config differs from preflight.")
if "pytorch_model.safetensors" in patterns:
    source = downloaded / "pytorch_model.safetensors"
    print("VERIFYING_DOWNLOAD_SHA256", flush=True)
    if source.stat().st_size != weight_size or sha256(source) != weight_sha:
        raise RuntimeError("Downloaded weight size or SHA256 mismatch.")
    copy_checked(source, weight, weight_sha)
copy_checked(downloaded / "config.json", show_dir / "config.json", config_sha)

phi_names = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
             "added_tokens.json", "vocab.json", "merges.txt"]
phi_files = [f for f in report["models"]["microsoft/phi-1_5"]["files"] if f["file"] in phi_names]
if not phi_files or any(f["bytes"] is None or f["bytes"] > 100 * 1024 * 1024 for f in phi_files):
    raise RuntimeError("Unexpected Phi tokenizer file metadata.")
phi_source = Path(snapshot_download(
    repo_id="microsoft/phi-1_5", revision=phi_rev, allow_patterns=[f["file"] for f in phi_files],
    cache_dir=str(stage / "hub-cache"), local_dir=str(stage / "phi-files"), max_workers=1,
))
phi_dir = root / "weights/phi-1_5-original"
phi_config_sha = report["models"]["microsoft/phi-1_5"]["config_sha256"]
if sha256(phi_source / "config.json") != phi_config_sha:
    raise RuntimeError("Phi config differs from preflight.")
for entry in phi_files:
    copy_checked(phi_source / entry["file"], phi_dir / entry["file"])

tokenizer = AutoTokenizer.from_pretrained(str(phi_dir), local_files_only=True, trust_remote_code=False)
base_vocab = len(tokenizer)
special_tokens = ["<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"]
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
tokenizer.add_tokens(special_tokens)
token_ids = {name: tokenizer.convert_tokens_to_ids(name) for name in ["[PAD]"] + special_tokens}
token_ids.update(bos=tokenizer.bos_token_id, eos=tokenizer.eos_token_id, image_offset=len(tokenizer), mask=58497)
if base_vocab != 50295 or len(tokenizer) != 50305 or token_ids["[PAD]"] != 50295:
    raise RuntimeError("Unexpected official tokenizer layout: " + json.dumps(token_ids))
if [token_ids[name] for name in special_tokens] != list(range(50296, 50305)):
    raise RuntimeError("Official special-token order mismatch.")
if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
    raise RuntimeError("Tokenizer must define BOS/EOS even though raw Phi config has null values.")
phi_config = AutoConfig.from_pretrained(str(phi_dir), local_files_only=True, trust_remote_code=False)
print("OFFICIAL_TOKEN_IDS:", json.dumps(token_ids), flush=True)
print("PHI_RESOLVED_KV_HEADS:", phi_config.num_key_value_heads, flush=True)

with weight.open("rb") as stream:
    header_size = struct.unpack("<Q", stream.read(8))[0]
    if not 0 < header_size < 20 * 1024 * 1024:
        raise RuntimeError("Unexpected safetensors header length.")
    header = json.loads(stream.read(header_size))
tensors = {name: value for name, value in header.items() if name != "__metadata__"}
for name, shape in {
    "showo.model.embed_tokens.weight": [58498, 2048],
    "showo.lm_head.weight": [58498, 2048],
    "showo.model.layers.0.self_attn.q_layernorm.weight": [64],
}.items():
    if tensors.get(name, {}).get("shape") != shape:
        raise RuntimeError("Unexpected checkpoint tensor: " + name)
layers = sorted({int(name.split(".")[3]) for name in tensors if name.startswith("showo.model.layers.")})
if layers != list(range(24)):
    raise RuntimeError("Unexpected layer indices: " + str(layers))
summary = dict(
    showo_revision=show_rev, phi_revision=phi_rev, weight_sha256=weight_sha,
    showo_path=str(show_dir), phi_path=str(phi_dir), stage=str(stage),
    tensor_count=len(tensors), parameters=sum(math.prod(t["shape"]) for t in tensors.values()),
    tensor_dtypes=dict(Counter(t["dtype"] for t in tensors.values())),
    layers=len(layers), token_ids=token_ids,
    scope="Download, SHA256, tokenizer layout and selected header shapes; no model forward or strict state load",
)
out = root / "results" / ("official-download-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
out.mkdir(parents=True, exist_ok=False)
with (out / "verification.json").open("x", encoding="utf-8") as stream:
    json.dump(summary, stream, indent=2)
print(json.dumps(summary, indent=2), flush=True)
print("OFFICIAL_WEIGHTS_DOWNLOAD_VERIFIED", out, flush=True)
