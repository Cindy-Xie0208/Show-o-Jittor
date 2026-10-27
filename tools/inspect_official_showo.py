from datetime import datetime
from pathlib import Path
import hashlib
import json
import shutil
import tempfile
from huggingface_hub import HfApi, hf_hub_download

root = Path.home() / "Show-o-Jittor"
if not (root / "official/models/modeling_showo.py").is_file():
    raise RuntimeError("Expected existing Show-o project; stop.")
stage = Path(tempfile.mkdtemp(prefix="xxt-showo-preflight-", dir="/tmp"))
api = HfApi()
report = {"endpoint": api.endpoint, "stage": str(stage), "models": {}}
print("SERVER_CONFIG_CACHE:", stage, flush=True)

for repo in ("showlab/show-o", "microsoft/phi-1_5"):
    print("READING_METADATA:", repo, flush=True)
    info = api.model_info(repo, files_metadata=True)
    print("REVISION:", repo, info.sha, flush=True)
    entries = []
    for item in info.siblings:
        lfs = item.lfs
        digest = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        entry = {"file": item.rfilename, "bytes": item.size, "lfs_sha256": digest}
        entries.append(entry)
        if repo == "showlab/show-o":
            print("FILE:", json.dumps(entry), flush=True)
    path = Path(hf_hub_download(
        repo_id=repo, filename="config.json", revision=info.sha,
        cache_dir=str(stage / "hub-cache"), etag_timeout=30,
    ))
    config_bytes = path.read_bytes()
    config = json.loads(config_bytes)
    print("CONFIG:", repo, json.dumps(config, indent=2), flush=True)
    report["models"][repo] = {
        "revision": info.sha, "files": entries, "config": config,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
    }

report["free_bytes"] = {
    "project_filesystem": shutil.disk_usage(root).free,
    "tmp_filesystem": shutil.disk_usage(stage).free,
}
out = root / "results" / ("official-preflight-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
out.mkdir(parents=True, exist_ok=False)
with (out / "official-preflight.json").open("x", encoding="utf-8") as stream:
    json.dump(report, stream, indent=2)
print("FREE_BYTES:", json.dumps(report["free_bytes"]), flush=True)
print("OFFICIAL_CONFIG_INSPECTION_DONE", out, flush=True)
