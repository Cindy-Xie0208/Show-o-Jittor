"""Server-only full-weight Show-o 1 FP32 forward alignment, without training."""

import argparse
import ast
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


COMMIT = "45a5a2de01d1ebd10cd5864d29310a76476cdf23"
REVISION = "dbe757d951099968fb102d249d35a764528df2dd"
SHA256 = "638739e52da4f5cba220b7cd59c04819f852dda180124c5a32c9c48b35a7299f"
PARAMETERS = 1448176770
ATOL, RTOL = 1e-4, 1e-4
DEFAULT_MMU = "official-mmu-20261026-223748-504059"
DEFAULT_T2I = "official-t2i-512-20261027-001553-017691"


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def trace_names():
    return (["model.embed_tokens"] + [f"model.layers.{i}" for i in range(24)]
            + ["model.final_layernorm", "logits"])


def check_gpu(gpu):
    used = int(subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"], text=True).strip())
    if used > 1024:
        raise RuntimeError(f"GPU {gpu} uses {used} MiB; stop and report nvidia-smi; do not stop other jobs")


def torch_cuda_state():
    module = sys.modules.get("torch")
    cuda = getattr(module, "cuda", None)
    initialized = getattr(cuda, "is_initialized", None)
    return dict(module_loaded=module is not None,
                cuda_initialized=bool(initialized()) if callable(initialized) else False)


def check_torch_cuda(stage):
    state = torch_cuda_state()
    if state["cuda_initialized"]:
        raise RuntimeError("Unexpected PyTorch CUDA initialization in native Jittor worker: " + stage)
    return state


def make_cases(args, torch, np, prompting):
    from transformers import AutoTokenizer

    root = Path.home() / "Show-o-Jittor"
    tokenizer = AutoTokenizer.from_pretrained(str(root / "weights/phi-1_5-original"), local_files_only=True)
    prompt = prompting.UniversalPrompting(tokenizer, max_text_len=128, cond_dropout_prob=0.0)
    assert len(tokenizer) == 50305 and tokenizer.pad_token_id == 50295
    special = lambda key: int(prompt.sptids_dict[key].item())
    assert special("<|soi|>") == 50296 and special("<|eoi|>") == 50297

    with np.load(args.mmu_run / "input.npz", allow_pickle=False) as saved:
        mmu_ids = torch.tensor(saved["input_ids"], dtype=torch.long)
        mmu_mask = torch.tensor(saved["attention_mask"], dtype=torch.float32)
    assert mmu_ids.shape[0] == 1
    official_mmu = prompting.create_attention_mask_for_mmu(mmu_ids, eoi_id=special("<|eoi|>")).float()
    assert torch.equal(official_mmu, mmu_mask), "Saved MMU mask differs from official mask"
    length = mmu_ids.shape[1]
    end = int(torch.where(mmu_ids[0] == special("<|eoi|>"))[0].item())
    pos = torch.arange(length)
    allowed = (pos[None, :] <= pos[:, None]) | (pos[None, :] <= end)
    assert torch.equal(official_mmu == 0, allowed[None, None])

    with np.load(args.t2i_run / "image_tokens.npz", allow_pickle=False) as saved:
        codes = saved["codes"].copy()
    assert codes.shape == (1, 1024) and codes.min() >= 0 and codes.max() < 8192
    settings = json.loads((args.t2i_run / "settings.json").read_text(encoding="utf-8"))
    text = settings["prompt"]
    cases = [("mmu", mmu_ids, mmu_mask)]
    for name, mixed in (("t2i_all_masks", False), ("t2i_half_masks", True)):
        image = torch.full((1, 1024), 58497, dtype=torch.long)
        if mixed:
            image[:, ::2] = torch.tensor(codes[:, ::2].astype(np.int64)) + len(tokenizer)
        cond, _ = prompt(([text], image), "t2i_gen")
        uncond, _ = prompt(([""], image), "t2i_gen")
        ids = torch.cat([cond, uncond])
        mask = prompting.create_attention_mask_predict_next(
            ids, pad_id=tokenizer.pad_token_id, soi_id=special("<|soi|>"),
            eoi_id=special("<|eoi|>"), rm_pad_in_image=True).float()
        length = ids.shape[1]
        expected = torch.ones((2, 1, length, length), dtype=torch.bool).tril()
        for batch in range(2):
            start = int(torch.where(ids[batch] == special("<|soi|>"))[0].item())
            pads = torch.where(ids[batch] == tokenizer.pad_token_id)[0]
            expected[batch, 0, start:, :] = True
            if len(pads):
                expected[batch, 0, int(pads[-1]) + 1:, :int(pads[-1]) + 1] = False
                expected[batch, 0, start:, pads] = False
        assert torch.equal(mask == 0, expected), "Independent T2I mask oracle differs"
        assert ids.shape == (2, 1155)
        cases.append((name, ids, mask))
    manifest = []
    for name, ids, mask in cases:
        directory = args.output / "reference" / name
        directory.mkdir(parents=True)
        np.savez(directory / "input.npz", input_ids=ids.numpy(), attention_mask=mask.numpy())
        manifest.append(dict(name=name, input_shape=list(ids.shape), mask_shape=list(mask.shape),
                             conditional_first=name != "mmu"))
    write_json(args.output / "cases.json", manifest)
    print("FULL_OFFICIAL_MASK_ORACLES_OK", flush=True)
    return cases


def reference_worker(args):
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig

    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    root = Path.home() / "Show-o-Jittor"
    phi = load_module("full_official_phi", root / "official/models/phi.py")
    prompting = load_module("full_official_prompting", root / "official/training/prompting_utils.py")
    cases = make_cases(args, torch, np, prompting)
    config = AutoConfig.from_pretrained(str(root / "weights/phi-1_5-original"), local_files_only=True)
    config.vocab_size = 58498
    config._attn_implementation = "eager"
    with torch.device("cuda"):
        model = phi.PhiForCausalLM(config)
    weights = load_file(str(root / "weights/show-o-512x512/pytorch_model.safetensors"), device="cpu")
    if not all(name.startswith("showo.") for name in weights):
        raise ValueError("Unexpected state outside official Show-o Transformer")
    state = {name.removeprefix("showo."): value for name, value in weights.items()}
    print(model.load_state_dict(state, strict=True), flush=True)
    assert sum(p.numel() for p in model.parameters()) == PARAMETERS
    assert all(p.dtype == torch.float32 for p in model.parameters())
    model.eval().requires_grad_(False)
    del weights, state
    config_dict = config.to_dict()
    expected = dict(hidden_size=2048, intermediate_size=8192, num_hidden_layers=24,
                    num_attention_heads=32, num_key_value_heads=32, partial_rotary_factor=0.5,
                    qk_layernorm=True, vocab_size=58498, use_cache=False)
    for key, value in expected.items():
        assert config_dict[key] == value, f"Unexpected effective config: {key}"
    write_json(args.output / "phi_config.json", config_dict)
    print("FULL_TORCH_STRICT_LOAD_OK", PARAMETERS, flush=True)

    for name, ids, mask in cases:
        directory = args.output / "reference" / name
        captured = set()
        handles = []

        def capture(key):
            def hook(module, inputs, result):
                value = result[0] if isinstance(result, (tuple, list)) else result
                array = value.detach().cpu().numpy()
                assert np.isfinite(array).all(), f"Non-finite reference: {name}/{key}"
                np.save(directory / (key + ".npy"), array, allow_pickle=False)
                captured.add(key)
            return hook

        wanted = set(trace_names()) - {"logits"}
        for key, module in model.named_modules():
            if key in wanted:
                handles.append(module.register_forward_hook(capture(key)))
        print("FULL_TORCH_FORWARD", name, list(ids.shape), flush=True)
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                logits = model(input_ids=ids.cuda(), attention_mask=mask.cuda(), use_cache=False).logits
                actual = logits.cpu().numpy()
                assert np.isfinite(actual).all()
                np.save(directory / "logits.npy", actual, allow_pickle=False)
                if name != "mmu":
                    guided = 6.0 * actual[:1, -1025:-1, 50305:58497] - 5.0 * actual[1:, -1025:-1, 50305:58497]
                    np.save(directory / "guided_logits.npy", guided, allow_pickle=False)
                del actual, logits
        finally:
            for handle in handles:
                handle.remove()
        assert captured == wanted, f"Missing reference hooks: {wanted - captured}"
        print("FULL_TORCH_CASE_READY", name, "seconds:", round(time.perf_counter() - started, 2), flush=True)
    write_json(args.output / "reference_status.json", dict(status="passed", parameters=PARAMETERS,
               torch_version=torch.__version__, attention="eager", dtype="float32", commit=COMMIT,
               checkpoint_revision=REVISION, checkpoint_sha256=SHA256))
    print("FULL_TORCH_REFERENCE_READY", flush=True)


def compare(np, actual, expected, case, name):
    if actual.shape != expected.shape:
        raise ValueError(f"Shape mismatch: {case}/{name}: {actual.shape} vs {expected.shape}")
    a, b = actual.reshape(-1), expected.reshape(-1)
    finite, failed, squared_error, squared_reference, absolute_sum, maximum = True, 0, 0.0, 0.0, 0.0, 0.0
    for start in range(0, a.size, 1024 * 1024):
        x = a[start:start + 1024 * 1024].astype(np.float64)
        y = b[start:start + 1024 * 1024].astype(np.float64)
        error = np.abs(x - y)
        valid = np.isfinite(x) & np.isfinite(y)
        finite = finite and bool(valid.all())
        failed += int(np.count_nonzero(~valid | (error > ATOL + RTOL * np.abs(y))))
        if valid.all():
            maximum = max(maximum, float(error.max()))
            absolute_sum += float(error.sum())
            squared_error += float(np.square(error).sum())
            squared_reference += float(np.square(y).sum())
    return dict(case=case, name=name, shape=list(actual.shape), values=int(a.size),
                finite=finite, max_abs=maximum if finite else None,
                mean_abs=absolute_sum / a.size if finite else None,
                relative_l2=(squared_error / max(squared_reference, 1e-24)) ** 0.5 if finite else None,
                outside_tolerance=failed, atol=ATOL, rtol=RTOL, passed=finite and failed == 0)


def diagnostic_comparison(np, actual, expected, case, name):
    row = compare(np, actual, expected, case, name)
    worst_ratio, worst = -1.0, None
    a, b = actual.reshape(-1), expected.reshape(-1)
    for start in range(0, a.size, 1024 * 1024):
        x = a[start:start + 1024 * 1024].astype(np.float64)
        y = b[start:start + 1024 * 1024].astype(np.float64)
        ratios = np.abs(x - y) / (ATOL + RTOL * np.abs(y))
        index = int(np.argmax(ratios))
        if float(ratios[index]) > worst_ratio:
            worst_ratio = float(ratios[index])
            worst = dict(index=[int(i) for i in np.unravel_index(start + index, actual.shape)],
                         actual=float(x[index]), expected=float(y[index]),
                         absolute_error=float(abs(x[index] - y[index])),
                         allowed_error=float(ATOL + RTOL * abs(y[index])))
    row.update(worst_tolerance_ratio=worst_ratio, worst=worst)
    return row


def diagnostic_reference_worker(args):
    import numpy as np
    import torch
    from safetensors import safe_open
    from transformers import PhiConfig

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    root = Path.home() / "Show-o-Jittor"
    phi = load_module("diagnostic_official_phi", root / "official/models/phi.py")
    config = PhiConfig.from_dict(json.loads((args.reference_run / "phi_config.json").read_text()))
    config._attn_implementation = "eager"
    cases = json.loads((args.reference_run / "cases.json").read_text())
    rows = []
    with safe_open(str(root / "weights/show-o-512x512/pytorch_model.safetensors"), framework="pt", device="cpu") as weights:
        for index in (0, 22, 23):
            prefix = f"model.layers.{index}"
            with torch.device("cuda"):
                layer = phi.PhiDecoderLayer(config, layer_idx=index)
            state = {key.removeprefix("showo." + prefix + "."): weights.get_tensor(key)
                     for key in weights.keys() if key.startswith("showo." + prefix + ".")}
            layer.load_state_dict(state, strict=True)
            layer.eval().requires_grad_(False)
            del state
            for case in cases:
                name = case["name"]
                reference = args.reference_run / "reference" / name
                output = args.output / "diagnostic_reference" / name
                output.mkdir(parents=True, exist_ok=True)
                previous = "model.embed_tokens" if index == 0 else f"model.layers.{index - 1}"
                hidden = torch.from_numpy(np.load(reference / (previous + ".npy"))).cuda()
                with np.load(reference / "input.npz", allow_pickle=False) as saved:
                    mask = torch.from_numpy(saved["attention_mask"]).cuda()
                handles = []
                captured = set()

                def capture(key):
                    def hook(module, inputs, result):
                        value = result[0] if isinstance(result, (tuple, list)) else result
                        np.save(output / (prefix + "." + key + ".npy"), value.detach().cpu().numpy(), allow_pickle=False)
                        captured.add(key)
                    return hook

                wanted = {"input_layernorm", "self_attn", "self_attn.q_proj", "self_attn.k_proj",
                          "self_attn.v_proj", "self_attn.q_layernorm", "self_attn.k_layernorm", "mlp"}
                for key, module in layer.named_modules():
                    if key in wanted:
                        handles.append(module.register_forward_hook(capture(key)))
                try:
                    with torch.inference_mode():
                        positions = torch.arange(hidden.shape[1], device="cuda")[None]
                        result = layer(hidden, attention_mask=mask, position_ids=positions, use_cache=False)[0]
                        expected = np.load(reference / (prefix + ".npy"), mmap_mode="r")
                        row = diagnostic_comparison(np, result.cpu().numpy(), expected, name, prefix + ".torch_replay")
                        rows.append(row)
                        print("TORCH_REPLAY_CONTROL", json.dumps(row), flush=True)
                        if not row["passed"]:
                            raise RuntimeError("Same-input PyTorch replay differs from saved oracle; inspect before Jittor comparison")
                finally:
                    for handle in handles:
                        handle.remove()
                assert captured == wanted, "Diagnostic hook keys differ"
                del hidden, mask, positions, result, expected
            del layer
    write_json(args.output / "diagnostic_reference_status.json", dict(status="passed", controls=rows,
               layers=[0, 22, 23], reference_run=str(args.reference_run)))
    print("SAME_INPUT_TORCH_DIAGNOSTIC_READY", flush=True)


def local_replay_worker(args, jt, np, model):
    oracle = json.loads((args.output / "diagnostic_reference_status.json").read_text())
    assert oracle["status"] == "passed"
    baseline = json.loads((args.diagnose_run / "alignment.json").read_text())
    cases = json.loads((args.reference_run / "cases.json").read_text())
    rows = []
    with (args.output / "local_replay.jsonl").open("x", encoding="utf-8") as stream:
        def record(actual, expected, name, key):
            row = diagnostic_comparison(np, actual, expected, name, key)
            if row["worst"] is not None:
                coordinate = row["worst"]["index"]
                sequence_axis = 2 if actual.ndim == 4 else 1
                token_id = int(raw_ids[coordinate[0], coordinate[sequence_axis]])
                row["worst"]["input_token_id"] = token_id
                row["worst"]["input_region"] = ("padding" if token_id == 50295 else
                                                 "image_or_mask" if token_id >= 50305 else "text_or_special")
            rows.append(row)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            print("SAME_INPUT_COMPARE", json.dumps(row), flush=True)

        for case in cases:
            name = case["name"]
            reference = args.reference_run / "reference" / name
            with np.load(reference / "input.npz", allow_pickle=False) as saved:
                raw_ids = saved["input_ids"].copy()
                mask = jt.array(saved["attention_mask"].astype(np.float32)).stop_grad()
            with jt.no_grad():
                for index, layer in enumerate(model.model.layers):
                    key = f"model.layers.{index}"
                    previous = "model.embed_tokens" if index == 0 else f"model.layers.{index - 1}"
                    hidden = jt.array(np.load(reference / (previous + ".npy"))).stop_grad()
                    trace = {}
                    result = layer(hidden, mask, trace, key)
                    expected = np.load(reference / (key + ".npy"), mmap_mode="r")
                    record(result.numpy(), expected, name, key + ".same_input")
                    if index in (0, 22, 23):
                        for component in ("input_layernorm", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                                          "self_attn.q_layernorm", "self_attn.k_layernorm", "self_attn", "mlp"):
                            component_key = key + "." + component
                            expected_component = np.load(args.output / "diagnostic_reference" / name / (component_key + ".npy"), mmap_mode="r")
                            record(trace[component_key].numpy(), expected_component, name, component_key + ".same_input")
                            del expected_component
                    del hidden, result, trace, expected
                    jt.sync_all(True)
                    jt.gc()
                hidden = jt.array(np.load(reference / "model.layers.23.npy")).stop_grad()
                norm = model.model.final_layernorm(hidden)
                expected = np.load(reference / "model.final_layernorm.npy", mmap_mode="r")
                record(norm.numpy(), expected, name, "model.final_layernorm.same_input")
                del hidden, norm, expected
                hidden = jt.array(np.load(reference / "model.final_layernorm.npy")).stop_grad()
                result = model.lm_head(hidden)
                expected = np.load(reference / "logits.npy", mmap_mode="r")
                record(result.numpy(), expected, name, "lm_head.same_input")
                del hidden, result, expected
            del mask
            jt.sync_all(True)
            jt.gc()
            check_torch_cuda("same-input replay " + name)
    failed = [row for row in rows if not row["passed"]]
    summary = dict(status="diagnostic_completed", alignment_status="not_reclassified",
                   baseline_run=str(args.diagnose_run), baseline_failed=baseline["failed"],
                   checked=len(rows), outside_tolerance=len(failed), tolerances=dict(atol=ATOL, rtol=RTOL),
                   isolated_failures=[dict(case=row["case"], name=row["name"], max_abs=row["max_abs"],
                                           outside_tolerance=row["outside_tolerance"], worst=row["worst"]) for row in failed],
                   scope="Each native block/final norm/head receives exact saved PyTorch input; selected components "
                         "use independent same-input official PyTorch block oracles; diagnostic only, original failure preserved")
    write_json(args.output / "local_replay.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    print("FULL_NUMERICAL_DIAGNOSTIC_DONE", args.output, flush=True)


def precision_reference_worker(args):
    import numpy as np
    import torch

    root = Path.home() / "Show-o-Jittor"
    phi = load_module("precision_official_phi", root / "official/models/phi.py")
    config = json.loads((args.reference_run / "phi_config.json").read_text())
    dim = int(config["partial_rotary_factor"] * config["hidden_size"] // config["num_attention_heads"])
    with torch.device("cuda"):
        rope = phi.PhiRotaryEmbedding(dim, max_position_embeddings=config["max_position_embeddings"],
                                     base=config["rope_theta"])
    assert rope.cos_cached.dtype == torch.float32 and rope.sin_cached.dtype == torch.float32
    np.savez(args.output / "official_rope.npz", cos=rope.cos_cached.cpu().numpy(), sin=rope.sin_cached.cpu().numpy())
    write_json(args.output / "rope_provenance.json", dict(commit=COMMIT, reference_run=str(args.reference_run),
               generator="Pinned official PhiRotaryEmbedding, CUDA FP32, identical constructor to reference",
               sha256=file_hash(args.output / "official_rope.npz")))
    print("OFFICIAL_ROPE_CONSTANTS_READY", flush=True)


def precision_worker(args, jt, np, model, imports):
    reference = args.reference_run / "reference/t2i_half_masks"
    key = "model.layers.23"
    hidden_np = np.load(reference / "model.layers.22.npy")
    with np.load(reference / "input.npz", allow_pickle=False) as saved:
        mask = jt.array(saved["attention_mask"].astype(np.float32)).stop_grad()
    with jt.no_grad():
        hidden = jt.array(hidden_np).stop_grad()
        trace = {}
        result = model.model.layers[23](hidden, mask, trace, key).numpy()
        attention = trace[key + ".self_attn"].numpy()
        mlp = trace[key + ".mlp"].numpy()
        ordered = np.add(np.add(attention, mlp, dtype=np.float32), hidden_np, dtype=np.float32)
        stored = args.precision_run / "diagnostic_reference/t2i_half_masks"
        expected_attention = np.load(stored / (key + ".self_attn.npy"))
        expected_mlp = np.load(stored / (key + ".mlp.npy"))
        expected = np.load(reference / (key + ".npy"))
        torch_ordered = np.add(np.add(expected_attention, expected_mlp, dtype=np.float32), hidden_np, dtype=np.float32)
        cancellation = dict(
            native_vs_ordered_numpy=diagnostic_comparison(np, result, ordered, "t2i_half_masks", "native_residual_vs_explicit_fp32_sum"),
            torch_vs_ordered_numpy=diagnostic_comparison(np, torch_ordered, expected, "t2i_half_masks", "torch_residual_control"),
            native_branches_ordered_vs_reference=diagnostic_comparison(np, ordered, expected, "t2i_half_masks", "ordered_native_branches"))
        failure = json.loads((args.precision_run / "local_replay.json").read_text())["isolated_failures"]
        cancellation["examples"] = []
        for row in failure:
            if row["case"] == "t2i_half_masks" and row["name"] == key + ".same_input":
                coordinate = tuple(row["worst"]["index"])
                cancellation["examples"].append(dict(index=list(coordinate),
                    native_attention=float(attention[coordinate]), reference_attention=float(expected_attention[coordinate]),
                    native_mlp=float(mlp[coordinate]), reference_mlp=float(expected_mlp[coordinate]),
                    residual=float(hidden_np[coordinate]), native_output=float(result[coordinate]),
                    reference_output=float(expected[coordinate]), ordered_native_output=float(ordered[coordinate])))
        write_json(args.output / "residual_cancellation.json", cancellation)
        print("RESIDUAL_CANCELLATION", json.dumps(cancellation), flush=True)
    del hidden_np, hidden, mask, trace, result, attention, mlp, ordered, expected_attention, expected_mlp, expected, torch_ordered
    jt.sync_all(True)
    jt.gc()

    layers = list(model.model.layers)
    original_rope = [(layer.self_attn._cos, layer.self_attn._sin) for layer in layers]
    original_execute = type(layers[0]).execute

    def ordered_execute(block, hidden, mask, trace, prefix):
        normalized = block.input_layernorm(hidden)
        trace[prefix + ".input_layernorm"] = normalized
        attention = block.self_attn(normalized, mask, trace, prefix + ".self_attn")
        feed_forward = block.mlp(normalized)
        trace[prefix + ".mlp"] = feed_forward
        jt.sync_all(True)
        combined = attention + feed_forward
        jt.sync_all(True)
        output = combined + hidden
        trace[prefix] = output
        return output

    with np.load(args.output / "official_rope.npz", allow_pickle=False) as archive:
        cosine, sine = archive["cos"].copy(), archive["sin"].copy()
    assert cosine.shape == tuple(original_rope[0][0].shape[2:]) and cosine.dtype == np.float32
    rope_delta = dict(cos_max_abs=float(np.abs(original_rope[0][0].numpy()[0, 0] - cosine).max()),
                      sin_max_abs=float(np.abs(original_rope[0][1].numpy()[0, 0] - sine).max()))
    print("ROPE_CONSTANT_DIFFERENCES", json.dumps(rope_delta), flush=True)
    matched_cosine = jt.array(cosine)[None, None].stop_grad()
    matched_sine = jt.array(sine)[None, None].stop_grad()
    variants = []
    try:
        for variant in ("baseline_repeat", "matched_rope", "matched_rope_ordered_residual"):
            for index, layer in enumerate(layers):
                layer.self_attn._cos, layer.self_attn._sin = (original_rope[index] if variant == "baseline_repeat"
                                                           else (matched_cosine, matched_sine))
            type(layers[0]).execute = ordered_execute if variant == "matched_rope_ordered_residual" else original_execute
            directory = args.output / variant
            directory.mkdir()
            print("PRECISION_VARIANT_START", variant, flush=True)
            report = forward_alignment(args, jt, np, model, directory, imports)
            variants.append(dict(variant=variant, status=report["status"], checked=report["checked"],
                                 failed=report["failed"], decisions=report["decisions"]))
            print("PRECISION_VARIANT_RESULT", json.dumps(variants[-1]), flush=True)
            if report["status"] == "passed":
                break
    finally:
        type(layers[0]).execute = original_execute
        for layer, tables in zip(layers, original_rope):
            layer.self_attn._cos, layer.self_attn._sin = tables
    accepted = next((row["variant"] for row in variants if row["status"] == "passed"), None)
    summary = dict(status="candidate_passed" if accepted else "no_candidate_passed", accepted_variant=accepted,
                   variants=variants, rope_differences=rope_delta, parameters=PARAMETERS,
                   tolerances=dict(atol=ATOL, rtol=RTOL), precision_run=str(args.precision_run),
                   scope="Original weights and FP32 arithmetic retained; runtime precision candidates checked on all 83 original comparisons; "
                         "no native source overwritten, training/decoding not yet verified")
    write_json(args.output / "precision_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    print("FULL_PRECISION_SWEEP_DONE", args.output, flush=True)
    if not accepted:
        raise RuntimeError("No precision candidate passed original tolerances; inspect precision_summary.json and residual_cancellation.json")


def full_operator_candidate_worker(args, jt, np, model, imports):
    summary = json.loads((args.operator_candidate_run / "candidate_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "local_candidates_completed"
    assert summary["tolerances"] == dict(atol=ATOL, rtol=RTOL)
    assert "centered_layernorm_flat_linear" in summary["local_passed_variants"]
    script = args.operator_candidate_run / "operator_script.py"
    source = script.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {node.name: node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef) and node.name in ("centered_layernorm", "flat_linear")}
    assert set(functions) == {"centered_layernorm", "flat_linear"}
    namespace = {"jt": jt, "np": np}
    exec(compile(ast.Module(body=list(functions.values()), type_ignores=[]), "verified_full_operator_candidate", "exec"), namespace)
    for layer in model.model.layers:
        type(layer.input_layernorm).execute = namespace["centered_layernorm"]
        type(layer.mlp.fc1).execute = namespace["flat_linear"]
    type(model.lm_head).execute = namespace["flat_linear"]
    write_json(args.output / "runtime_operator_candidate.json", dict(
        variant="centered_layernorm_flat_linear", source_run=str(args.operator_candidate_run),
        source_sha256=file_hash(script),
        functions={name: ast.get_source_segment(source, node) for name, node in functions.items()},
        scope="Runtime-only FP32 overrides for all 24 input LayerNorms and Linear modules; checkpoint and saved native source unchanged"))
    report = forward_alignment(args, jt, np, model, args.output, imports)
    if report["status"] != "passed":
        print("FULL_OPERATOR_CANDIDATE_REMAINS_OUTSIDE_TOLERANCE", report["failed"], "/ 83", flush=True)
    else:
        print("FULL_OPERATOR_CANDIDATE_PASSED_ORIGINAL_TOLERANCES", flush=True)


def jittor_worker(args):
    imports = {"before_dependencies": torch_cuda_state()}
    import numpy as np
    imports["after_numpy"] = torch_cuda_state()
    import jittor as jt
    imports["after_jittor"] = torch_cuda_state()
    from safetensors import safe_open

    imports["after_safetensors"] = torch_cuda_state()
    print("NATIVE_DEPENDENCY_IMPORTS", json.dumps(imports), flush=True)
    check_torch_cuda("dependency imports")
    jt.flags.use_cuda = 1
    jt.flags.amp_level = 0
    jt.set_global_seed(42)
    port = load_module("full_native_phi", args.output / "native_phi.py")
    reference_run = args.reference_run or args.output
    reference = json.loads((reference_run / "reference_status.json").read_text(encoding="utf-8"))
    assert reference["status"] == "passed" and reference["parameters"] == PARAMETERS
    config = json.loads((reference_run / "phi_config.json").read_text(encoding="utf-8"))
    model = port.model_class(jt, np)(config)
    model.eval()
    state = model.state_dict()
    assert sum(int(np.prod(value.shape)) for value in state.values()) == PARAMETERS
    weight = Path.home() / "Show-o-Jittor/weights/show-o-512x512/pytorch_model.safetensors"
    with safe_open(str(weight), framework="np") as archive:
        mapping = {name.removeprefix("showo."): name for name in archive.keys()}
        assert set(mapping) == set(state), "Full native state keys differ"
        for index, (key, parameter) in enumerate(state.items(), 1):
            array = archive.get_tensor(mapping[key])
            assert array.dtype == np.float32 and tuple(parameter.shape) == tuple(array.shape), key
            parameter.assign(jt.array(array)).stop_grad()
            if not np.array_equal(parameter.numpy(), array):
                raise ValueError("Full parameter did not load exactly: " + key)
            if index == 1 or index % 48 == 0 or index == len(state):
                print("FULL_JITTOR_LOAD", index, "/", len(state), key, flush=True)
    print("FULL_JITTOR_WEIGHTS_LOADED_EXACTLY", PARAMETERS, flush=True)
    if args.operator_candidate_run:
        full_operator_candidate_worker(args, jt, np, model, imports)
        return
    if args.precision_run:
        precision_worker(args, jt, np, model, imports)
        return
    if args.diagnose_run:
        local_replay_worker(args, jt, np, model)
        return
    report = forward_alignment(args, jt, np, model, args.output, imports)
    if report["status"] != "passed":
        raise RuntimeError("Full forward alignment exceeded preset tolerances; inspect alignment.jsonl; do not widen automatically")
    print("FULL_SHOWO_JITTOR_FORWARD_ALIGN_OK", flush=True)


def forward_alignment(args, jt, np, model, output, imports):
    reference_run = args.reference_run or args.output
    cases = json.loads((reference_run / "cases.json").read_text(encoding="utf-8"))
    rows, decisions = [], []
    with (output / "alignment.jsonl").open("x", encoding="utf-8") as stream:
        for case in cases:
            name = case["name"]
            directory = reference_run / "reference" / name
            with np.load(directory / "input.npz", allow_pickle=False) as saved:
                ids = jt.array(saved["input_ids"].astype(np.int32)).stop_grad()
                mask = jt.array(saved["attention_mask"].astype(np.float32)).stop_grad()
            print("FULL_JITTOR_FORWARD", name, case["input_shape"], flush=True)
            started = time.perf_counter()
            with jt.no_grad():
                logits, trace = model(ids, mask)
                for key in trace_names():
                    expected = np.load(directory / (key + ".npy"), mmap_mode="r", allow_pickle=False)
                    actual = trace[key].numpy()
                    row = compare(np, actual, expected, name, key)
                    rows.append(row)
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    print("FULL_ALIGN", json.dumps(row), flush=True)
                    if key == "logits":
                        top1 = (actual.argmax(-1) == expected.argmax(-1))
                        decisions.append(dict(case=name, all_positions_top1_agreement=float(top1.mean())))
                        if name != "mmu":
                            guided = 6.0 * actual[:1, -1025:-1, 50305:58497] - 5.0 * actual[1:, -1025:-1, 50305:58497]
                            guided_expected = np.load(directory / "guided_logits.npy", mmap_mode="r", allow_pickle=False)
                            guided_row = compare(np, guided, guided_expected, name, "guided_logits")
                            rows.append(guided_row)
                            stream.write(json.dumps(guided_row) + "\n")
                            stream.flush()
                            print("FULL_ALIGN", json.dumps(guided_row), flush=True)
                            decisions[-1]["image_guided_top1_agreement"] = float(
                                (guided.argmax(-1) == guided_expected.argmax(-1)).mean())
                            del guided, guided_expected
                    del actual, expected
            del ids, mask, logits, trace
            jt.sync_all(True)
            jt.gc()
            check_torch_cuda("completed " + name)
            print("FULL_JITTOR_CASE_DONE", name, "seconds:", round(time.perf_counter() - started, 2), flush=True)
    failed = [row for row in rows if not row["passed"]]
    report = dict(status="failed" if failed else "passed", parameters=PARAMETERS, commit=COMMIT,
                  checkpoint_revision=REVISION, checkpoint_sha256=SHA256,
                  jittor_version=jt.__version__, dtype="float32", checked=len(rows), failed=len(failed),
                  reference_run=str(reference_run), dependency_imports=imports,
                  torch_cuda_state=check_torch_cuda("final report"),
                  tolerances=dict(atol=ATOL, rtol=RTOL), decisions=decisions,
                  scope="Full pretrained Transformer, every value of embedding/24 block outputs/final norm/logits; "
                        "fixed MMU and 1024-image-token all/half-mask CFG inputs; no gradients, training or decoding")
    assert len(rows) == 83, "Incomplete full forward comparison set"
    if args.precision_run:
        report["precision_variant"] = output.name
        report["rope_constants_sha256"] = file_hash(args.output / "official_rope.npz")
    write_json(output / "alignment.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def environment(gpu, jittor):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), NVIDIA_TF32_OVERRIDE="0", PYTHONUNBUFFERED="1",
               CUBLAS_WORKSPACE_CONFIG=":4096:8", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false")
    if jittor:
        view = Path.home() / "Show-o-Jittor/deps/cuda-jittor"
        env.update(CUDA_HOME=str(view), nvcc_path=str(view / "bin/nvcc"),
                   JITTOR_HOME="/tmp/xxt-showo-jittor", DISABLE_MULTIPROCESSING="1")
        env["PATH"] = str(view / "bin") + ":" + env.get("PATH", "")
        old = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = str(view / "lib64") + (":" + old if old else "")
    return env


def run_stage(command, env, logfile):
    with logfile.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace")
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        stream.write("\nExit code: " + str(code) + "\n")
    if code:
        raise RuntimeError("Stage failed: " + str(code) + "; log: " + str(logfile))


def main():
    root = Path.home() / "Show-o-Jittor"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--mmu-run", type=Path, default=root / "results" / DEFAULT_MMU)
    parser.add_argument("--t2i-run", type=Path, default=root / "results" / DEFAULT_T2I)
    parser.add_argument("--reference-run", type=Path,
                        help="Reuse an intact completed PyTorch reference; run only the Jittor stage")
    parser.add_argument("--diagnose-run", type=Path,
                        help="Diagnose an existing failed alignment using independent same-input layer replay")
    parser.add_argument("--precision-run", type=Path,
                        help="Check FP32 RoPE/residual precision candidates using a completed numerical diagnostic")
    parser.add_argument("--operator-candidate-run", type=Path,
                        help="Test a previously passing local LayerNorm/Linear candidate across all 83 full-model comparisons")
    parser.add_argument("--worker", choices=("reference", "diagnostic_reference", "precision_reference", "jittor"), help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if Path(sys.prefix).resolve() != (Path.home() / "envs/showo-jittor").resolve():
        raise RuntimeError("Run in the configured showo-jittor environment")
    if args.worker:
        {"reference": reference_worker, "diagnostic_reference": diagnostic_reference_worker,
         "precision_reference": precision_reference_worker,
         "jittor": jittor_worker}[args.worker](args)
        return
    selected_modes = sum(value is not None for value in (args.diagnose_run, args.precision_run, args.operator_candidate_run))
    if selected_modes > 1:
        parser.error("Choose only one diagnostic, precision, or full operator-candidate mode")
    if args.operator_candidate_run:
        args.operator_candidate_run = args.operator_candidate_run.resolve(strict=True)
        candidate = json.loads((args.operator_candidate_run / "candidate_summary.json").read_text(encoding="utf-8"))
        candidate_config = json.loads((args.operator_candidate_run / "run_config.json").read_text(encoding="utf-8"))
        assert candidate["status"] == "local_candidates_completed"
        assert candidate["tolerances"] == dict(atol=ATOL, rtol=RTOL)
        assert "centered_layernorm_flat_linear" in candidate["local_passed_variants"]
        assert candidate_config["checkpoint_sha256"] == SHA256 and candidate_config["commit"] == COMMIT
        assert file_hash(args.operator_candidate_run / "operator_script.py") == candidate_config["runner_sha256"]
        diagnostic_path = Path(candidate_config["diagnostic"]).resolve(strict=True)
        diagnostic_config = json.loads((diagnostic_path / "run_config.json").read_text(encoding="utf-8"))
        diagnostic_summary = json.loads((diagnostic_path / "local_replay.json").read_text(encoding="utf-8"))
        assert diagnostic_summary["status"] == "diagnostic_completed"
        assert diagnostic_config["checkpoint_sha256"] == SHA256 and diagnostic_config["commit"] == COMMIT
        assert Path(diagnostic_config["reference_run"]).is_absolute()
        args.reference_run = Path(diagnostic_config["reference_run"]).resolve(strict=True)
    if args.precision_run:
        args.precision_run = args.precision_run.resolve(strict=True)
        diagnostic = json.loads((args.precision_run / "local_replay.json").read_text())
        assert diagnostic["status"] == "diagnostic_completed"
        assert diagnostic["tolerances"] == dict(atol=ATOL, rtol=RTOL)
        for component in ("self_attn", "mlp"):
            path = args.precision_run / "diagnostic_reference/t2i_half_masks" / ("model.layers.23." + component + ".npy")
            if not path.is_file():
                raise FileNotFoundError("Missing residual diagnostic: " + str(path))
        baseline = json.loads((Path(diagnostic["baseline_run"]) / "alignment.json").read_text())
        assert baseline["checkpoint_sha256"] == SHA256 and baseline["commit"] == COMMIT
        assert baseline["parameters"] == PARAMETERS and baseline["status"] == "failed"
        reference_path = Path(baseline["reference_run"]).resolve(strict=True)
        if args.reference_run and args.reference_run.resolve(strict=True) != reference_path:
            raise ValueError("Precision diagnostic points to a different reference")
        args.reference_run = reference_path
    if args.diagnose_run:
        args.diagnose_run = args.diagnose_run.resolve(strict=True)
        baseline = json.loads((args.diagnose_run / "alignment.json").read_text(encoding="utf-8"))
        assert baseline["status"] == "failed" and baseline["parameters"] == PARAMETERS
        assert baseline["checkpoint_sha256"] == SHA256 and baseline["commit"] == COMMIT
        assert baseline["tolerances"] == dict(atol=ATOL, rtol=RTOL)
        reference_path = Path(baseline["reference_run"]).resolve(strict=True)
        if args.reference_run and args.reference_run.resolve(strict=True) != reference_path:
            raise ValueError("Diagnostic baseline points to a different reference")
        args.reference_run = reference_path
    native_path = root / "align_tiny_jittor.py"
    if args.reference_run:
        args.reference_run = args.reference_run.resolve(strict=True)
        reference = json.loads((args.reference_run / "reference_status.json").read_text(encoding="utf-8"))
        saved_config = json.loads((args.reference_run / "run_config.json").read_text(encoding="utf-8"))
        assert reference["status"] == "passed" and reference["parameters"] == PARAMETERS
        assert reference["commit"] == COMMIT and reference["checkpoint_sha256"] == SHA256
        assert reference["checkpoint_revision"] == REVISION and reference["attention"] == "eager"
        assert reference["dtype"] == "float32"
        assert saved_config["atol"] == ATOL and saved_config["rtol"] == RTOL
        native_path = args.reference_run / "native_phi.py"
        assert file_hash(native_path) == saved_config["native_source_sha256"], "Native source snapshot changed"
        assert file_hash(args.reference_run / "alignment_script.py") == saved_config["runner_sha256"], "Original runner snapshot changed"
        if args.diagnose_run:
            assert file_hash(args.diagnose_run / "native_phi.py") == saved_config["native_source_sha256"], "Diagnostic native source differs"
        if args.precision_run:
            assert file_hash(args.precision_run / "native_phi.py") == saved_config["native_source_sha256"], "Precision native source differs"
        if args.operator_candidate_run:
            candidate_config = json.loads((args.operator_candidate_run / "run_config.json").read_text(encoding="utf-8"))
            assert candidate_config["native_sha256"] == saved_config["native_source_sha256"]
            diagnostic_config = json.loads((Path(candidate_config["diagnostic"]) / "run_config.json").read_text(encoding="utf-8"))
            assert Path(diagnostic_config["reference_run"]).resolve() == args.reference_run
        cases = json.loads((args.reference_run / "cases.json").read_text(encoding="utf-8"))
        assert [case["name"] for case in cases] == ["mmu", "t2i_all_masks", "t2i_half_masks"]
        for case in cases:
            directory = args.reference_run / "reference" / case["name"]
            required = [directory / "input.npz"] + [directory / (key + ".npy") for key in trace_names()]
            if case["name"] != "mmu":
                required.append(directory / "guided_logits.npy")
            for path in required:
                if not path.is_file():
                    raise FileNotFoundError("Incomplete reference; preserved: " + str(path))
        print("REUSING_COMPLETED_TORCH_REFERENCE", args.reference_run, flush=True)
    required = [native_path, root / "deps/cuda-jittor/bin/nvcc"]
    if not args.reference_run:
        required.extend([args.mmu_run / "input.npz", args.t2i_run / "image_tokens.npz", args.t2i_run / "settings.json"])
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(str(path) + "; preserve files and report this output")
    check_gpu(args.gpu)
    repo = root / "official"
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    assert commit == COMMIT, "Official source revision differs"
    subprocess.run(["git", "-C", str(repo), "diff", "--quiet", "HEAD", "--",
                    "models/phi.py", "models/modeling_showo.py", "training/prompting_utils.py"], check=True)
    checkpoint = root / "weights/show-o-512x512"
    verification = json.loads((checkpoint / "verification.json").read_text(encoding="utf-8"))
    assert verification["verified"] is True and verification["sha256"] == SHA256
    assert verification["revision"] == REVISION and verification["repo"] == "showlab/show-o-512x512"
    print("VERIFYING_FULL_WEIGHT_SHA256", flush=True)
    assert file_hash(checkpoint / "pytorch_model.safetensors") == SHA256
    needed_bytes = 8 * 1024 ** 3 if args.diagnose_run or not args.reference_run else 128 * 1024 ** 2
    if shutil.disk_usage(root).free < needed_bytes:
        raise RuntimeError("Insufficient free space for this alignment run")
    prefix = ("official512-full-operator-candidate-" if args.operator_candidate_run else
              "official512-precision-" if args.precision_run else
              "official512-numerics-" if args.diagnose_run else "official512-jittor-align-")
    args.output = root / "results" / (prefix + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    args.output.mkdir(parents=True, exist_ok=False)
    script = args.output / "alignment_script.py"
    source = globals().get("showo_source")
    if source is None:
        source = Path(__file__).read_text(encoding="utf-8")
    compile(source, str(script), "exec")
    with script.open("x", encoding="utf-8") as stream:
        stream.write(source)
    native_source = native_path.read_text(encoding="utf-8")
    compile(native_source, "native_phi.py", "exec")
    with (args.output / "native_phi.py").open("x", encoding="utf-8") as stream:
        stream.write(native_source)
    write_json(args.output / "run_config.json", dict(
        gpu=args.gpu, mmu_run=str(args.mmu_run), t2i_run=str(args.t2i_run), commit=COMMIT,
        reference_run=str(args.reference_run) if args.reference_run else None,
        diagnose_run=str(args.diagnose_run) if args.diagnose_run else None,
        precision_run=str(args.precision_run) if args.precision_run else None,
        operator_candidate_run=str(args.operator_candidate_run) if args.operator_candidate_run else None,
        checkpoint_revision=REVISION, checkpoint_sha256=SHA256, parameters=PARAMETERS,
        atol=ATOL, rtol=RTOL, tolerances_note="FP32 full-depth pretrained model; absolute plus relative elementwise bound",
        runner_sha256=file_hash(script), native_source_sha256=file_hash(args.output / "native_phi.py")))
    print("RUN_DIR", args.output, flush=True)
    base = [sys.executable, "-u", str(script), "--gpu", str(args.gpu), "--output", str(args.output),
            "--mmu-run", str(args.mmu_run), "--t2i-run", str(args.t2i_run)]
    if args.reference_run:
        base.extend(["--reference-run", str(args.reference_run)])
    if args.diagnose_run:
        base.extend(["--diagnose-run", str(args.diagnose_run)])
    if args.precision_run:
        base.extend(["--precision-run", str(args.precision_run)])
    if args.operator_candidate_run:
        base.extend(["--operator-candidate-run", str(args.operator_candidate_run)])
    stages = (("jittor",) if args.operator_candidate_run else
              ("precision_reference", "jittor") if args.precision_run else
              ("diagnostic_reference", "jittor") if args.diagnose_run else
              ("jittor",) if args.reference_run else ("reference", "jittor"))
    try:
        for stage in stages:
            check_gpu(args.gpu)
            run_stage(base + ["--worker", stage], environment(args.gpu, stage == "jittor"),
                      args.output / (stage + ".log"))
    except BaseException as error:
        write_json(args.output / "run_status.json", dict(status="failed", error=repr(error)))
        raise
    scope = ("Full-depth centered LayerNorm plus flattened Linear candidate; original 83 tolerances retained" if args.operator_candidate_run else
             "FP32 precision candidates; inspect precision_summary.json" if args.precision_run else
             "same-input numerical diagnostic; alignment failure preserved" if args.diagnose_run else "full-weight FP32 forward alignment only")
    write_json(args.output / "run_status.json", dict(status="completed", scope=scope))
    marker = ("FULL_OPERATOR_CANDIDATE_RUN_COMPLETE" if args.operator_candidate_run else
              "PRECISION_SWEEP_RUN_COMPLETE" if args.precision_run else
              "NUMERICAL_DIAGNOSTIC_RUN_COMPLETE" if args.diagnose_run else "FULL_ALIGNMENT_RUN_COMPLETE")
    print(marker, args.output, flush=True)


if __name__ == "__main__":
    main()
