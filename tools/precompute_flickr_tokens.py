"""Server-only frozen MagViT image codes for the verified Flickr8k subset."""
from pathlib import Path
from datetime import datetime
import hashlib, importlib, json, sys, types
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms as T

root = Path.home() / "Show-o-Jittor"
data = root / "data/flickr8k-128-32-20261026-203910-164291"
raw = (data / "manifest.json").read_bytes()
manifest = json.loads(raw)
rows = manifest["samples"]
splits = np.array([row["split"] for row in rows])
train_ids, val_ids = np.flatnonzero(splits == "train"), np.flatnonzero(splits == "validation")
assert len(rows) == 160 and len(train_ids) == 128 and len(val_ids) == 32, "Wrong split counts"
assert len({row["rgb_sha256"] for row in rows}) == 160, "Duplicate image content"
package = types.ModuleType("showo_vq")
package.__path__ = [str(root / "official/models")]
sys.modules[package.__name__] = package
model = importlib.import_module("showo_vq.modeling_magvitv2").MAGVITv2()
model.load_state_dict(load_file(str(root / "weights/magvitv2/pytorch_model.safetensors")), strict=True)
model = model.eval().requires_grad_(False).cuda()
transform = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
                       T.CenterCrop(256), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
out = root / "data" / ("flickr8k-tokens-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
out.mkdir(parents=True, exist_ok=False)
arrays = []
print("GPU:", torch.cuda.get_device_name(0), "Output:", out, flush=True)
with torch.inference_mode():
    for start in range(0, len(rows), 4):
        part, images = rows[start:start + 4], []
        for row in part:
            with Image.open(data / row["image"]) as image:
                rgb = image.convert("RGB")
            digest = hashlib.sha256(str(rgb.size).encode("ascii") + rgb.tobytes()).hexdigest()
            assert digest == row["rgb_sha256"], "Image changed: " + row["image"]
            images.append(transform(rgb))
        codes = model.get_code(torch.stack(images).cuda()).cpu().numpy().astype(np.int32)
        assert codes.shape == (len(part), 256), "Wrong token shape"
        assert codes.min() >= 0 and codes.max() < 8192, "Token outside codebook"
        arrays.append(codes)
        done = start + len(part)
        if done % 16 == 0 or done == len(rows):
            print("ENCODED", done, "/", len(rows), flush=True)
tokens = np.concatenate(arrays)
np.savez_compressed(out / "tokens.npz", image_tokens=tokens, train_indices=train_ids, validation_indices=val_ids)
with np.load(out / "tokens.npz", allow_pickle=False) as saved:
    assert np.array_equal(saved["image_tokens"], tokens), "Token serialization mismatch"
(out / "source_manifest.json").write_bytes(raw)
report = dict(source_data=str(data), source_manifest_sha256=hashlib.sha256(raw).hexdigest(),
              tokenizer="showlab/magvitv2", tokenizer_revision="5c3fa78f8b3523347c5cd1a4c97f3c4e96f33d5d",
              backend="frozen PyTorch", torch_version=torch.__version__, codebook_size=8192,
              preprocessing="RGB, resize short edge 256 bicubic, center crop 256, normalize [-1,1]",
              shape=list(tokens.shape), train_images=128, validation_images=32,
              code_min=int(tokens.min()), code_max=int(tokens.max()), unique_codes=int(np.unique(tokens).size))
(out / "metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2), flush=True)
print("FLICKR_TOKENS_READY", out, flush=True)
