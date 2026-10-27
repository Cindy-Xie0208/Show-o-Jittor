"""Run on the server with CUDA_VISIBLE_DEVICES set to an available GPU."""
from pathlib import Path
from datetime import datetime
import importlib, json, math, sys, types
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms as T

root = Path.home() / "Show-o-Jittor"
package = types.ModuleType("showo_vq")
package.__path__ = [str(root / "official/models")]
sys.modules[package.__name__] = package
cls = importlib.import_module("showo_vq.modeling_magvitv2").MAGVITv2
model = cls()
model.load_state_dict(load_file(str(root / "weights/magvitv2/pytorch_model.safetensors")), strict=True)
model = model.eval().requires_grad_(False).cuda()
crop = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(256)])
out = root / "results" / ("magvit-reconstruction-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
out.mkdir(parents=True, exist_ok=False)
rows, all_codes = [], {}
print("GPU:", torch.cuda.get_device_name(0), "Results:", out, flush=True)
with torch.inference_mode():
    for name in ("dog.png", "sofa_under_water.jpg"):
        source = root / "official/mmu_validation" / name
        with Image.open(source) as image:
            original = crop(image.convert("RGB"))
        pixels = T.ToTensor()(original).unsqueeze(0).cuda()
        codes = model.get_code(pixels * 2 - 1)
        assert tuple(codes.shape) == (1, 256), "Unexpected token shape"
        low, high = int(codes.min().item()), int(codes.max().item())
        assert 0 <= low <= high < 8192, "Token outside codebook"
        decoded = model.decode_code(codes)
        assert decoded.shape == pixels.shape and torch.isfinite(decoded).all().item(), "Invalid reconstruction"
        reconstructed = ((decoded + 1) / 2).clamp(0, 1)
        mse = float((reconstructed - pixels).square().mean().item())
        restored = T.ToPILImage()(reconstructed[0].cpu())
        stem = source.stem
        original.save(out / (stem + "-input.png"))
        restored.save(out / (stem + "-reconstruction.png"))
        pair = Image.new("RGB", (512, 256))
        pair.paste(original, (0, 0))
        pair.paste(restored, (256, 0))
        pair.save(out / (stem + "-comparison.png"))
        all_codes[stem] = codes.cpu().numpy()
        row = dict(image=name, tokens=256, code_min=low, code_max=high,
                   mse=mse, psnr_db=-10 * math.log10(max(mse, 1e-12)))
        rows.append(row)
        print(json.dumps(row), flush=True)
np.savez_compressed(out / "image_tokens.npz", **all_codes)
report = dict(scope="Frozen PyTorch tokenizer reconstruction; not text-to-image generation",
              torch_version=torch.__version__, device=torch.cuda.get_device_name(0),
              preprocessing="RGB, resize short edge 256 bicubic, center crop 256, normalize [-1,1]",
              psnr_scope="Clipped float reconstruction vs preprocessed input, data range 1; two examples only",
              samples=rows)
(out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print("MAGVIT_REAL_IMAGE_RECON_OK", flush=True)
print("Results:", out, flush=True)
