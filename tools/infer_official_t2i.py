from pathlib import Path
from datetime import datetime
import argparse, hashlib, importlib, importlib.util, json, subprocess, sys, types
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from safetensors.torch import load_file
from transformers import AutoTokenizer
root = Path.home() / "Show-o-Jittor"
parser = argparse.ArgumentParser(description="Server-only official Show-o 1 T2I baseline")
parser.add_argument("--resolution", type=int, choices=(256, 512), default=256)
parser.add_argument("--prompt", default="A photo of a golden retriever sitting in a field of yellow flowers.")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
resolution = args.resolution
num_tokens = (resolution // 16) ** 2
checkpoint_name = "show-o-512x512" if resolution == 512 else "show-o"
checkpoint_dir = root / "weights" / checkpoint_name
config_name = "showo_demo_512x512.yaml" if resolution == 512 else "showo_demo.yaml"
config_path = root / "official/configs" / config_name
config = OmegaConf.load(config_path)
assert config.model.showo.pretrained_model_path == "showlab/" + checkpoint_name
assert config.model.showo.num_vq_tokens == num_tokens
assert config.dataset.preprocessing.resolution == resolution
checkpoint_config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
assert checkpoint_config["num_vq_tokens"] == num_tokens
assert checkpoint_config["vocab_size"] == 58498 and checkpoint_config["mask_token_id"] == 58497
weight_path = checkpoint_dir / "pytorch_model.safetensors"
weight_sha = None
if resolution == 512:
    expected_sha = "638739e52da4f5cba220b7cd59c04819f852dda180124c5a32c9c48b35a7299f"
    verification = json.loads((checkpoint_dir / "verification.json").read_text(encoding="utf-8"))
    assert verification["repo"] == "showlab/show-o-512x512" and verification["verified"] is True
    assert verification["revision"] == "dbe757d951099968fb102d249d35a764528df2dd"
    assert verification["sha256"] == expected_sha and weight_path.stat().st_size == 5792758016
    print("VERIFYING_512_WEIGHT_SHA256", flush=True)
    digest = hashlib.sha256()
    with weight_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    weight_sha = digest.hexdigest()
    assert weight_sha == expected_sha, "Checkpoint SHA256 mismatch"
commit = subprocess.check_output(["git", "-C", str(root / "official"), "rev-parse", "HEAD"], text=True).strip()
assert commit == "45a5a2de01d1ebd10cd5864d29310a76476cdf23"
used = int(subprocess.check_output(["nvidia-smi", "-i", "1", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip())
assert used <= 1024, "GPU 1 occupied; stop and report nvidia-smi"
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
torch.manual_seed(args.seed)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
package = types.ModuleType("official_showo")
package.__path__ = [str(root / "official/models")]
sys.modules[package.__name__] = package
Showo = importlib.import_module("official_showo.modeling_showo").Showo
MAGVITv2 = importlib.import_module("official_showo.modeling_magvitv2").MAGVITv2
spec = importlib.util.spec_from_file_location("official_prompt", root / "official/training/prompting_utils.py")
prompting = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = prompting
spec.loader.exec_module(prompting)
tokenizer = AutoTokenizer.from_pretrained(str(root / "weights/phi-1_5-original"), local_files_only=True)
prompt = prompting.UniversalPrompting(tokenizer, max_text_len=128, cond_dropout_prob=0.0)
assert len(tokenizer) == 50305
print("LOADING_OFFICIAL_T2I_MODELS", "resolution:", resolution, "image_tokens:", num_tokens, flush=True)
with torch.device("cuda"):
    model = Showo(w_clip_vit=False, vocab_size=58498, llm_vocab_size=50295,
                  num_vq_tokens=num_tokens, llm_model_path=str(root / "weights/phi-1_5-original"))
print(model.load_state_dict(load_file(str(weight_path), device="cuda"), strict=True), flush=True)
model.eval().requires_grad_(False)
vq = MAGVITv2()
print(vq.load_state_dict(load_file(str(root / "weights/magvitv2/pytorch_model.safetensors")), strict=True), flush=True)
vq = vq.eval().requires_grad_(False).cuda()
assert config.model.showo.llm_vocab_size + config.model.showo.num_new_special_tokens == len(tokenizer)
text = args.prompt
masked = torch.full((1, num_tokens), 58497, dtype=torch.long, device="cuda")
ids, _ = prompt(([text], masked), "t2i_gen")
uncond, _ = prompt(([""], masked), "t2i_gen")
mask = prompting.create_attention_mask_predict_next(torch.cat([ids, uncond]), pad_id=tokenizer.pad_token_id,
    soi_id=int(prompt.sptids_dict["<|soi|>"].item()), eoi_id=int(prompt.sptids_dict["<|eoi|>"].item()), rm_pad_in_image=True).float()
steps = [0]
def progress(module, args, output):
    steps[0] += 1
    if steps[0] == 1 or steps[0] % 10 == 0:
        print("T2I_STEP", steps[0], "/ 50", flush=True)
hook = model.register_forward_hook(progress)
with torch.inference_mode():
    codes = model.t2i_generate(ids, uncond_input_ids=uncond, attention_mask=mask, timesteps=50,
        guidance_scale=5.0, temperature=1.0, generator=torch.Generator(device="cuda").manual_seed(args.seed), config=config)
    assert codes.shape == (1, num_tokens) and codes.min().item() >= 0 and codes.max().item() < 8192
    decoded = vq.decode_code(codes)
hook.remove()
assert decoded.shape == (1, 3, resolution, resolution) and torch.isfinite(decoded).all().item()
pixels = ((decoded[0] + 1) / 2).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
prefix = "official-t2i-512-" if resolution == 512 else "official-t2i-"
out = root / "results" / (prefix + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
out.mkdir(parents=True, exist_ok=False)
Image.fromarray((pixels * 255).astype(np.uint8)).save(out / "generated.png")
np.savez_compressed(out / "image_tokens.npz", codes=codes.cpu().numpy())
report = dict(prompt=text, seed=args.seed, steps=50, guidance_scale=5.0, temperature=1.0, code_commit=commit,
              resolution=resolution, image_tokens=num_tokens, checkpoint=str(weight_path),
              weight_sha256=weight_sha, official_config=str(config_path), torch_version=torch.__version__,
              peak_allocated_bytes=torch.cuda.max_memory_allocated(),
              dtype=str(next(model.parameters()).dtype), attention=model.showo.config._attn_implementation,
              scope="Official PyTorch full-weight T2I single image; visual quality not yet inspected; not Jittor generation")
(out / "settings.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print("GENERATED_IMAGE:", out / "generated.png", flush=True)
print("OFFICIAL_T2I_BASELINE_DONE", out, flush=True)
