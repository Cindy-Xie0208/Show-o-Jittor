"""Server-only Flickr8k pilot subset; fixed shard, seed and original split."""
from pathlib import Path
from datetime import datetime
import hashlib, io, json, tempfile
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from huggingface_hub import snapshot_download

repo = "jxie/flickr8k"
revision = "56f58c967835f7c508d684f36bd7897cca9d7634"
files = {
    "train": "data/train-00000-of-00002-2f8f6bfa852eac4b.parquet",
    "validation": "data/validation-00000-of-00001-7025a2b596f14b7b.parquet",
}
stage = Path(tempfile.mkdtemp(prefix="xxt-showo-flickr-", dir="/tmp"))
print("SERVER_DOWNLOAD_DIR:", stage, flush=True)
folder = Path(snapshot_download(
    repo_id=repo, repo_type="dataset", revision=revision,
    allow_patterns=list(files.values()), max_workers=1,
    cache_dir=str(stage / "hub-cache"), local_dir=str(stage / "files"),
))
root = Path.home() / "Show-o-Jittor/data"
out = root / ("flickr8k-128-32-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
out.mkdir(parents=True, exist_ok=False)
rng, samples, seen = np.random.default_rng(42), [], set()
for split, count in (("train", 128), ("validation", 32)):
    table = pq.read_table(folder / files[split])
    indices = np.sort(rng.choice(table.num_rows, size=count, replace=False))
    rows = table.take(indices.tolist()).to_pylist()
    (out / split).mkdir()
    for number, (index, row) in enumerate(zip(indices, rows)):
        with Image.open(io.BytesIO(row["image"]["bytes"])) as image:
            rgb = image.convert("RGB")
        digest = hashlib.sha256(str(rgb.size).encode("ascii") + rgb.tobytes()).hexdigest()
        if digest in seen:
            raise RuntimeError("Duplicate image content; stop and report output.")
        seen.add(digest)
        relative = split + "/" + str(number).zfill(5) + ".png"
        rgb.save(out / relative)
        captions = [row["caption_" + str(i)] for i in range(5)]
        if not all(isinstance(c, str) and c.strip() for c in captions):
            raise RuntimeError("Missing caption; stop and report output.")
        samples.append(dict(split=split, image=relative, source_row=int(index),
                            source_shard=files[split], rgb_sha256=digest, captions=captions))
    print(split, count, "images; example:", captions[0], flush=True)
manifest = dict(repo=repo, revision=revision, seed=42, train_images=128,
                validation_images=32, samples=samples,
                scope="Pilot subset; training sampled from first original train shard only")
(out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("FLICKR_SUBSET_READY", out, flush=True)
