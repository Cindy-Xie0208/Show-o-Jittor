"""Replay one saved T2I forward to inspect argmax margins and mask state.

Does not rerun generation, change acceptance criteria, or modify old results.
The saved native logits are compared with a new independent PyTorch replay.
"""

import argparse
from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from native_pipeline import COMMIT, SHOWO_SHA, file_hash, load_module, read_json, score_comparison, write_json


def top_two(values):
    first = int(np.argmax(values))
    remaining = values.copy()
    remaining[first] = -np.inf
    second = int(np.argmax(remaining))
    return dict(top1=first, top2=second, top1_logit=float(values[first]),
                top2_logit=float(values[second]),
                margin=float(values[first]) - float(values[second]))


def inspect(native, reference, ids, history, masks, final_codes, step):
    if native.shape != reference.shape or native.shape != (1, 1024, 8192):
        raise RuntimeError("Unexpected guided logits shape")
    if not np.isfinite(native).all() or not np.isfinite(reference).all():
        raise RuntimeError("Nonfinite guided logits")
    if history.shape != (50, 1, 1024) or masks.shape != history.shape or final_codes.shape != (1, 1024):
        raise RuntimeError("Unexpected saved sampling history shape")
    current = ids[:1, -1025:-1]
    if current.shape != (1, 1024):
        raise RuntimeError("Unexpected image input shape")
    if step > 1 and not np.array_equal(current == 58497, masks[step - 2].astype(bool)):
        raise RuntimeError("Replay input mask disagrees with the preceding saved step")
    rows = []
    different = np.argwhere(native.argmax(-1) != reference.argmax(-1))
    for batch, position in different:
        batch, position = int(batch), int(position)
        n, r = native[batch, position], reference[batch, position]
        nt, rt = top_two(n), top_two(r)
        a, b = nt["top1"], rt["top1"]
        max_error = float(np.max(np.abs(n.astype(np.float64) - r.astype(np.float64))))
        active = bool(current[batch, position] == 58497)
        rows.append(dict(
            batch=batch, image_position_zero_based=position,
            grid_row=position // 32, grid_column=position % 32,
            native=nt, pytorch=rt,
            native_gap_between_winners=float(n[a]) - float(n[b]),
            pytorch_gap_between_winners=float(r[a]) - float(r[b]),
            max_abs_error_at_position=max_error,
            native_margin_le_twice_max_error=nt["margin"] <= 2 * max_error,
            pytorch_margin_le_twice_max_error=rt["margin"] <= 2 * max_error,
            input_is_mask=active, input_token_id=int(current[batch, position]),
            native_sampled_or_preserved_code=int(history[step - 1, batch, position]),
            masked_after_this_step=bool(masks[step - 1, batch, position]),
            final_native_code=int(final_codes[batch, position]),
            note=("Position eligible for resampling; logits argmax alone does not identify the categorical sample."
                  if active else
                  "Position already committed; native sampler preserves its current code at this step. "
                  "This argmax mismatch does not change the preserved code at this position in this step.")))
    return dict(metrics=score_comparison(native, reference), different_positions=rows,
                difference_reproduced=bool(rows))


def main(argv=None):
    root = Path.home() / "Show-o-Jittor"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--run", type=Path, default=root / "results/full-native-512-20261114-165623-359725")
    parser.add_argument("--step", type=int, choices=(1, 25, 50), default=25)
    args = parser.parse_args(argv)
    used = int(subprocess.check_output(["nvidia-smi", "-i", str(args.gpu),
        "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True, timeout=10).strip())
    if used > 1024:
        raise RuntimeError("GPU occupied: %d MiB; no process terminated" % used)
    config = read_json(args.run / "run_config.json")
    if config["official_commit"] != COMMIT or config["showo_sha256"] != SHOWO_SHA:
        raise RuntimeError("Saved run provenance differs")
    commit = subprocess.check_output(["git", "-C", str(root / "official"), "rev-parse", "HEAD"], text=True).strip()
    if commit != COMMIT:
        raise RuntimeError("Official commit differs")
    subprocess.run(["git", "-C", str(root / "official"), "diff", "--quiet", "HEAD", "--", "models"], check=True)
    print("VERIFYING_REPLAY_WEIGHT_SHA256", flush=True)
    weight_path = root / "weights/show-o-512x512/pytorch_model.safetensors"
    if file_hash(weight_path) != SHOWO_SHA:
        raise RuntimeError("Weight checksum differs")
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), NVIDIA_TF32_OVERRIDE="0",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import torch
    from safetensors.torch import load_file
    from transformers import PhiConfig

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    reference_run = Path(config["reference_run"])
    phi = load_module("diagnostic_official_phi", root / "official/models/phi.py")
    phi_config = PhiConfig.from_dict(read_json(reference_run / "phi_config.json"))
    phi_config._attn_implementation = "eager"
    with torch.device("cuda"):
        model = phi.PhiForCausalLM(phi_config)
    weights = load_file(str(weight_path), device="cpu")
    print(model.load_state_dict({k.removeprefix("showo."): v for k, v in weights.items()}, strict=True), flush=True)
    del weights
    model.eval().requires_grad_(False)
    fixture_path = args.run / ("t2i-replay-%02d.npz" % args.step)
    with np.load(fixture_path, allow_pickle=False) as fixture:
        ids = fixture["input_ids"].copy()
        native = fixture["guided_logits"].copy()
    with np.load(args.run / "official_inputs.npz", allow_pickle=False) as fixture:
        attention = fixture["attention_mask"].copy()
    print("SINGLE_FORWARD_REPLAY_START", args.step, flush=True)
    with torch.inference_mode():
        logits = model(input_ids=torch.from_numpy(ids.astype(np.int64)).cuda(),
                       attention_mask=torch.from_numpy(attention).cuda(), use_cache=False).logits
        cond, uncond = logits[:, -1025:-1, 50305:58497].chunk(2, dim=0)
        reference = (6.0 * cond - 5.0 * uncond).cpu().numpy()
    with np.load(args.run / "image_tokens.npz", allow_pickle=False) as saved:
        result = inspect(native, reference, ids, saved["history"], saved["masks"], saved["codes"], args.step)
    previous = read_json(args.run / "native_pipeline_report.json")
    result.update(source_run=str(args.run), step=args.step,
        original_replay=[x for x in previous["replay"] if x["input"] == fixture_path.name],
        checkpoint_sha256=SHOWO_SHA, fixture_sha256=file_hash(fixture_path),
        diagnostic_sha256=file_hash(__file__),
        limits="Only this saved input was replayed. No new Jittor generation, no independent PyTorch sampling; existing acceptance results unchanged.")
    output = root / "results" / ("native-argmax-diagnostic-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, output / Path(__file__).name)
    write_json(output / "argmax_diagnostic.json", result)
    print("ARGMAX_DIAGNOSTIC_COMPLETE", output, flush=True)
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
