from pathlib import Path
from datetime import datetime
import importlib, importlib.util, json, subprocess, sys, types
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
root = Path.home() / "Show-o-Jittor"
commit = subprocess.check_output(["git", "-C", str(root / "official"), "rev-parse", "HEAD"], text=True).strip()
assert commit == "45a5a2de01d1ebd10cd5864d29310a76476cdf23", "Official source revision differs"
used = int(subprocess.check_output(["nvidia-smi", "-i", "1", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip())
assert used <= 1024, "GPU 1 occupied; stop and report nvidia-smi, do not terminate other jobs"
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
package = types.ModuleType("official_showo")
package.__path__ = [str(root / "official/models")]
sys.modules[package.__name__] = package
Showo = importlib.import_module("official_showo.modeling_showo").Showo
spec = importlib.util.spec_from_file_location("official_prompt", root / "official/training/prompting_utils.py")
prompting = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = prompting
spec.loader.exec_module(prompting)
tokenizer = AutoTokenizer.from_pretrained(str(root / "weights/phi-1_5-original"), local_files_only=True)
prompt = prompting.UniversalPrompting(tokenizer, max_text_len=128, cond_dropout_prob=0.0)
assert len(tokenizer) == 50305
print("BUILDING_OFFICIAL_MODEL_FP32", torch.cuda.get_device_name(0), flush=True)
with torch.device("cuda"):
    model = Showo(w_clip_vit=False, vocab_size=58498, llm_vocab_size=50295, llm_model_path=str(root / "weights/phi-1_5-original"))
weights = load_file(str(root / "weights/show-o/pytorch_model.safetensors"), device="cuda")
print(model.load_state_dict(weights, strict=True), flush=True)
del weights
model.eval().requires_grad_(False)
print("OFFICIAL_STRICT_LOAD_OK", sum(p.numel() for p in model.parameters()), flush=True)
tokens_path = root / "results/magvit-reconstruction-20261026-203006-738585/image_tokens.npz"
with np.load(tokens_path, allow_pickle=False) as archive:
    image = archive["dog"].copy()
assert image.shape == (1, 256) and image.min() >= 0 and image.max() < 8192
special = lambda name: int(prompt.sptids_dict[name].item())
question = "Describe this image in detail."
text_ids = tokenizer("USER: \n" + question + " ASSISTANT:")["input_ids"]
prefix = [special("<|mmu|>"), special("<|soi|>")]
prefix += (image[0] + len(tokenizer)).tolist()
prefix += [special("<|eoi|>"), tokenizer.bos_token_id]
ids = torch.tensor([prefix + text_ids], dtype=torch.long, device="cuda")
mask = prompting.create_attention_mask_for_mmu(ids, eoi_id=special("<|eoi|>")).float()
print("GENERATING_DOG_DESCRIPTION", flush=True)
with torch.inference_mode():
    generated = model.mmu_generate(ids, attention_mask=mask, max_new_tokens=64, top_k=1, eot_token=tokenizer.eos_token_id)
generated_ids = torch.stack(generated).cpu().tolist()
answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
out = root / "results" / ("official-mmu-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
out.mkdir(parents=True, exist_ok=False)
np.savez_compressed(out / "input.npz", input_ids=ids.cpu().numpy(), attention_mask=mask.cpu().numpy())
result = dict(question=question, answer=answer, generated_ids=generated_ids, source_image="official/mmu_validation/dog.png",
              source_image_tokens=str(tokens_path), code_commit=commit, dtype=str(next(model.parameters()).dtype),
              attention=model.showo.config._attn_implementation, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
              scope="Official PyTorch full-weight MMU single-sample baseline; no Jittor full-model or quality benchmark claim")
(out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
(out / "phi_config.json").write_text(model.showo.config.to_json_string(), encoding="utf-8")
print("ANSWER:", answer, flush=True)
print("OFFICIAL_MMU_BASELINE_DONE", out, flush=True)
