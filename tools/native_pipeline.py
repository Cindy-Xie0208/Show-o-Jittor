"""Show-o 1 native Jittor inference and independent PyTorch verification.

Called through infer_full_jittor.py --native-pipeline. Reference preparation,
native inference, and reference verification run in separate sequential
processes. The native process uses no PyTorch tensor operations; tokenizer,
image I/O and prompt construction use tokenizers, Pillow and NumPy.
"""

import argparse
import ast
from datetime import datetime
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import types

import numpy as np

COMMIT = "45a5a2de01d1ebd10cd5864d29310a76476cdf23"
SHOWO_SHA = "638739e52da4f5cba220b7cd59c04819f852dda180124c5a32c9c48b35a7299f"
MAGVIT_SHA = "79656bf974d99c215465a61df5e93b97e03282b0941f62f6294719527873b778"
PARAMETERS = 1448176770
ATOL = RTOL = 1e-4
DEFAULT_PROMPT = "A photo of a golden retriever sitting in a field of yellow flowers."
QUESTION = "Describe this image in detail."


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def native_guard(stage):
    torch = sys.modules.get("torch")
    initialized = getattr(getattr(torch, "cuda", None), "is_initialized", None)
    state = dict(module_loaded=torch is not None,
                 cuda_initialized=bool(initialized()) if callable(initialized) else False)
    if state["cuda_initialized"]:
        raise RuntimeError("PyTorch CUDA initialized in native worker: " + stage)
    return state


def t2i_prompt(encoded, specials, bos, eos):
    text = list(encoded)
    if not text or text[0] != bos:
        text.insert(0, bos)
    prefix = [specials["<|t2i|>"]] + text + [eos]
    if len(prefix) > 129:
        prefix = prefix[:128] + [eos]
    prefix = [specials["<|pad|>"]] * (129 - len(prefix)) + prefix
    return np.array([prefix + [specials["<|soi|>"]] + [58497] * 1024 + [specials["<|eoi|>"]]], dtype=np.int32)


def t2i_mask(ids, pad, soi, eoi):
    start = ids == soi
    end = ids == eoi
    image = (np.cumsum(start, axis=1) > np.cumsum(end, axis=1)) | start | end
    length = ids.shape[1]
    allowed = (~image)[:, :, None] & np.tril(np.ones((length, length), dtype=bool))[None]
    bidirectional = np.ones_like(allowed)
    for batch in range(ids.shape[0]):
        padding = np.flatnonzero(ids[batch] == pad)
        if len(padding):
            last = int(padding[-1])
            allowed[batch, last + 1:, :last + 1] = False
        image_start = int(np.flatnonzero(start[batch])[0])
        bidirectional[batch, image_start:, padding] = False
    allowed[image] = bidirectional[image]
    # Official masks are int64 minimum cast to FP32, not float32 minimum.
    return np.where(allowed[:, None], np.float32(0), np.float32(np.iinfo(np.int64).min)).astype(np.float32)


def mmu_prompt(codes, encoded_question, specials, bos):
    values = [specials["<|mmu|>"], specials["<|soi|>"]]
    values += (codes[0].astype(np.int64) + 50305).tolist()
    values += [specials["<|eoi|>"], bos] + list(encoded_question)
    ids = np.array([values], dtype=np.int32)
    allowed = np.tril(np.ones((len(values), len(values)), dtype=bool))
    eoi = values.index(specials["<|eoi|>"])
    allowed[:, :eoi + 1] = True
    mask = np.where(allowed[None, None], np.float32(0), np.float32(np.iinfo(np.int64).min)).astype(np.float32)
    return ids, mask


def image_pixels(path, resolution=512):
    from PIL import Image
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        if width <= height:
            size = (resolution, int(resolution * height / width))
        else:
            size = (int(resolution * width / height), resolution)
        if image.size != size:
            image = image.resize(size, Image.Resampling.BICUBIC)
        left = int(round((image.width - resolution) / 2.0))
        top = int(round((image.height - resolution) / 2.0))
        image = image.crop((left, top, left + resolution, top + resolution))
        pixels = np.array(image, dtype=np.float32).transpose(2, 0, 1)[None] / np.float32(255)
    return (pixels - np.float32(0.5)) / np.float32(0.5)


def prepare(args):
    import torch
    from PIL import Image
    from transformers import AutoTokenizer
    from torchvision import transforms

    root, output = args.root, args.output
    prompting = load_module("native_pipeline_official_prompt", root / "official/training/prompting_utils.py")
    tokenizer = AutoTokenizer.from_pretrained(str(root / "weights/phi-1_5-original"), local_files_only=True)
    prompt = prompting.UniversalPrompting(tokenizer, max_text_len=128, cond_dropout_prob=0.0)
    if len(tokenizer) != 50305 or tokenizer.pad_token_id != 50295:
        raise RuntimeError("Tokenizer vocabulary differs")
    specials = {key: int(value.item()) for key, value in prompt.sptids_dict.items()}
    tokenizer.save_pretrained(output / "tokenizer")
    masked = torch.full((1, 1024), 58497, dtype=torch.long)
    ids, _ = prompt(([args.prompt], masked), "t2i_gen")
    uncond, _ = prompt(([""], masked), "t2i_gen")
    mask = prompting.create_attention_mask_predict_next(torch.cat([ids, uncond]),
        pad_id=tokenizer.pad_token_id, soi_id=specials["<|soi|>"],
        eoi_id=specials["<|eoi|>"], rm_pad_in_image=True).float()
    np.savez_compressed(output / "official_inputs.npz", input_ids=ids.numpy(),
                        uncond_input_ids=uncond.numpy(), attention_mask=mask.numpy())
    source = (root / "official/training/utils.py").read_text(encoding="utf-8")
    functions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "image_transform"]
    if len(functions) != 1:
        raise RuntimeError("Official image transform not found")
    namespace = {"transforms": transforms}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "pinned_image_transform", "exec"), namespace)
    with Image.open(root / "official/mmu_validation/dog.png") as image:
        pixels = namespace["image_transform"](image.convert("RGB"), resolution=512).unsqueeze(0).numpy()
    np.save(output / "official_pixels.npy", pixels, allow_pickle=False)
    baseline = read_json(args.mmu_baseline / "mmu_result.json")
    if baseline.get("image_resolution") != 512 or baseline.get("image_tokens") != 1024:
        raise RuntimeError("MMU reference must use the 512px/1024-code dog input")
    with np.load(args.mmu_baseline / "mmu_input.npz", allow_pickle=False) as archive:
        codes = archive["image_codes"].copy()
        mmu_ids, mmu_mask = mmu_prompt(codes, tokenizer("USER: \n" + QUESTION + " ASSISTANT:")["input_ids"], specials, tokenizer.bos_token_id)
        if not np.array_equal(mmu_ids, archive["input_ids"]) or not np.array_equal(mmu_mask, archive["attention_mask"]):
            raise RuntimeError("MMU baseline input differs from the expected 512px dog case")
    expected_mask = prompting.create_attention_mask_for_mmu(torch.from_numpy(mmu_ids.astype(np.int64)),
        eoi_id=specials["<|eoi|>"]).float().numpy()
    if not np.array_equal(mmu_mask, expected_mask):
        raise RuntimeError("Native MMU initial mask differs from official")
    np.savez_compressed(output / "official_mmu.npz", input_ids=mmu_ids, attention_mask=mmu_mask, codes=codes)
    reference_ids = baseline["reference_generated_ids"]
    write_json(output / "prepare.json", dict(specials=specials, bos=tokenizer.bos_token_id,
        eos=tokenizer.eos_token_id, source_sha256=file_hash(root / "official/mmu_validation/dog.png"),
        reference_generated_ids=reference_ids[:args.max_new_tokens],
        reference_answer=baseline["reference_answer"],
        preprocessing_source=ast.get_source_segment(source, functions[0])))
    print("NATIVE_PIPELINE_REFERENCE_INPUTS_READY", flush=True)


def load_transformer(args, jt):
    from safetensors import safe_open
    port = load_module("pipeline_native_phi", args.reference_run / "native_phi.py")
    model = port.model_class(jt, np)(read_json(args.reference_run / "phi_config.json"))
    model.eval()
    state = model.state_dict()
    if sum(int(np.prod(value.shape)) for value in state.values()) != PARAMETERS:
        raise RuntimeError("Native Transformer parameter count differs")
    with safe_open(str(args.root / "weights/show-o-512x512/pytorch_model.safetensors"), framework="np") as archive:
        keys = {key.removeprefix("showo."): key for key in archive.keys()}
        if set(keys) != set(state):
            raise RuntimeError("Native Transformer weight keys differ")
        for index, (key, parameter) in enumerate(state.items(), 1):
            value = archive.get_tensor(keys[key])
            if value.dtype != np.float32 or tuple(value.shape) != tuple(parameter.shape):
                raise RuntimeError("Native Transformer shape/dtype differs: " + key)
            parameter.assign(jt.array(value)).stop_grad()
            if not np.array_equal(parameter.numpy(), value):
                raise RuntimeError("Native Transformer exact load failed: " + key)
            if index == 1 or index % 48 == 0 or index == len(state):
                print("NATIVE_PIPELINE_LOAD", index, "/", len(state), key, flush=True)
    source = (args.candidate_run / "operator_script.py").read_text(encoding="utf-8")
    functions = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.FunctionDef) and node.name == "flat_linear"]
    if len(functions) != 1 or functions[0].decorator_list:
        raise RuntimeError("Cannot extract verified flat_linear")
    namespace = {"jt": jt}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "verified_flat_linear", "exec"), namespace)
    type(model.lm_head).execute = namespace["flat_linear"]
    return model


def native(args):
    import jittor as jt
    from PIL import Image
    from tokenizers import Tokenizer
    import jittor_sampling as sampler
    from validate_native_magvit import MAGVIT_SHA256, load_exact, setup_native

    setup_native()
    from jittor_magvit import MAGVITv2
    jt.flags.amp_level = 0
    jt.set_global_seed(args.seed)
    state_before = native_guard("imports")
    started = time.monotonic()
    output = args.output
    info = read_json(output / "prepare.json")
    specials = info["specials"]
    tokenizer = Tokenizer.from_file(str(output / "tokenizer/tokenizer.json"))
    ids = t2i_prompt(tokenizer.encode(args.prompt).ids, specials, info["bos"], info["eos"])
    uncond = t2i_prompt(tokenizer.encode("").ids, specials, info["bos"], info["eos"])
    mask = t2i_mask(np.concatenate([ids, uncond]), specials["<|pad|>"],
                    specials["<|soi|>"], specials["<|eoi|>"])
    with np.load(output / "official_inputs.npz", allow_pickle=False) as fixture:
        for key, value in (("input_ids", ids), ("uncond_input_ids", uncond), ("attention_mask", mask)):
            if not np.array_equal(value, fixture[key]):
                raise RuntimeError("Native prompt/mask differs from official: " + key)
    pixels = image_pixels(args.root / "official/mmu_validation/dog.png")
    if not np.array_equal(pixels, np.load(output / "official_pixels.npy", allow_pickle=False)):
        raise RuntimeError("Native PIL/NumPy image preprocessing differs from official")
    print("NATIVE_PROMPTS_MASKS_PIXELS_EXACT", flush=True)
    magvit = MAGVITv2()
    magvit.eval()
    entries, elements = load_exact(magvit, args.root / "weights/magvitv2/pytorch_model.safetensors")
    with jt.no_grad():
        image_codes = magvit.get_code(jt.array(pixels).stop_grad()).numpy().astype(np.int32)
    with np.load(output / "official_mmu.npz", allow_pickle=False) as fixture:
        code_exact = np.array_equal(image_codes, fixture["codes"])
        mmu_ids, mmu_mask = mmu_prompt(image_codes, tokenizer.encode("USER: \n" + QUESTION + " ASSISTANT:").ids,
                                       specials, info["bos"])
        if not code_exact or not np.array_equal(mmu_ids, fixture["input_ids"]) or not np.array_equal(mmu_mask, fixture["attention_mask"]):
            raise RuntimeError("Native 512px MMU encoding/input differs from reference")
    np.savez_compressed(output / "native_mmu_input.npz", input_ids=mmu_ids, attention_mask=mmu_mask, codes=image_codes)
    Image.fromarray(np.clip((pixels[0].transpose(1, 2, 0) + 1) * 127.5, 0, 255).astype(np.uint8)).save(output / "mmu_input.png")
    print("NATIVE_MMU_512_INPUT_EXACT", flush=True)
    del pixels
    jt.sync_all(True)
    jt.gc()
    model = load_transformer(args, jt)
    calls = [0]
    replay = []
    # Save first/middle/last inputs and guided logits for independent same-input
    # reference replay. This is not a cross-framework random-seed comparison.
    def t2i_forward(current_ids, attention_mask):
        jt.sync_all(True)
        jt.gc()
        step = calls[0]
        logits, trace = model(current_ids, attention_mask)
        if step in (0, 24, 49):
            branch = logits[:, -1025:-1, 50305:58497]
            cond, uncond_logits = branch.chunk(2, dim=0)
            guided = (6.0 * cond - 5.0 * uncond_logits).numpy()
            if not np.isfinite(guided).all():
                raise RuntimeError("Nonfinite T2I guided logits")
            name = "t2i-replay-%02d.npz" % (step + 1)
            np.savez_compressed(output / name, input_ids=current_ids.numpy(),
                                guided_logits=guided)
            replay.append(name)
        del trace
        calls[0] += 1
        if calls[0] == 1 or calls[0] % 10 == 0:
            print("NATIVE_T2I_FORWARD", calls[0], "/ 50", flush=True)
        native_guard("T2I forward")
        return logits
    # Re-seed after weight initialization so sampling RNG does not depend on it.
    jt.set_global_seed(args.seed)
    with jt.no_grad():
        codes, history, masks = sampler.t2i_generate(t2i_forward, jt.array(ids).stop_grad(),
            jt.array(uncond).stop_grad(), jt.array(mask).stop_grad())
    if codes.shape != (1, 1024) or codes.min() < 0 or codes.max() >= 8192 or calls[0] != 50:
        raise RuntimeError("Incomplete native T2I generation or invalid final codes")
    np.savez_compressed(output / "image_tokens.npz", codes=codes, history=np.stack(history), masks=np.stack(masks))
    with jt.no_grad():
        decoded = magvit.decode_code(jt.array(codes).stop_grad()).numpy()
    if decoded.shape != (1, 3, 512, 512) or not np.isfinite(decoded).all():
        raise RuntimeError("Invalid native T2I decoded image")
    np.save(output / "native_decoded.npy", decoded, allow_pickle=False)
    Image.fromarray(np.clip((decoded[0].transpose(1, 2, 0) + 1) * 127.5, 0, 255).astype(np.uint8)).save(output / "generated.png")
    print("NATIVE_T2I_IMAGE_READY", output / "generated.png", flush=True)
    mmu_calls = [0]
    mmu_replay = []
    def mmu_forward(current_ids, attention_mask):
        jt.sync_all(True)
        jt.gc()
        step = mmu_calls[0]
        logits, trace = model(current_ids, attention_mask)
        if step in (0, args.max_new_tokens // 2, args.max_new_tokens - 1):
            last = logits[:, -1, :].numpy()
            if not np.isfinite(last).all():
                raise RuntimeError("Nonfinite native MMU logits")
            name = "mmu-replay-%02d.npz" % (step + 1)
            np.savez_compressed(output / name, input_ids=current_ids.numpy(),
                                attention_mask=attention_mask.numpy(), logits=last)
            mmu_replay.append(name)
        del trace
        mmu_calls[0] += 1
        if mmu_calls[0] == 1 or mmu_calls[0] % 10 == 0:
            print("NATIVE_MMU_FORWARD", mmu_calls[0], "/", args.max_new_tokens, flush=True)
        native_guard("MMU forward")
        return logits
    with jt.no_grad():
        tokens, margins = sampler.mmu_generate(mmu_forward, jt.array(mmu_ids).stop_grad(),
            jt.array(mmu_mask).stop_grad(), max_new_tokens=args.max_new_tokens, eot_token=info["eos"])
    expected = info["reference_generated_ids"]
    agreement = sum(a == b for a, b in zip(tokens, expected)) / max(len(tokens), len(expected))
    first = next((i + 1 for i, (a, b) in enumerate(zip(tokens, expected)) if a != b), None)
    if first is None and len(tokens) != len(expected):
        first = min(len(tokens), len(expected)) + 1
    report = dict(status="completed", t2i_calls=calls[0], mmu_calls=mmu_calls[0],
        native_import_state_before=state_before, native_import_state_after=native_guard("generation complete"),
        seed=args.seed, random_sampling_framework="jittor", t2i_steps=50,
        t2i_replay=replay, mmu_replay=mmu_replay,
        mmu=dict(question=QUESTION, generated_ids=tokens, answer=tokenizer.decode(tokens, skip_special_tokens=True),
                 reference_generated_ids=expected, exact_token_match=tokens == expected,
                 token_agreement=agreement, first_divergence_step=first, top1_margins=margins),
        magvit=dict(state_entries=entries, state_elements_including_buffers=elements,
                    encoding_exact=bool(code_exact), checkpoint_sha256=MAGVIT_SHA256),
        elapsed_seconds=time.monotonic() - started,
        scope="Jittor Transformer, sampler, MagViT encoding/decoding; PIL/NumPy preprocessing and tokenizers; no PyTorch tensor computation in native worker")
    write_json(output / "native_result.json", report)
    print("NATIVE_MMU_ANSWER", report["mmu"]["answer"], flush=True)
    print("NATIVE_MMU_EXACT_TOKEN_MATCH", report["mmu"]["exact_token_match"], flush=True)
    print("NATIVE_PIPELINE_GENERATION_COMPLETE", output, flush=True)


def score_comparison(actual, reference):
    if actual.shape != reference.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise RuntimeError("Invalid same-input replay outputs")
    delta = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    limit = ATOL + RTOL * np.abs(reference.astype(np.float64))
    return dict(atol=ATOL, rtol=RTOL, max_abs=float(delta.max()), mean_abs=float(delta.mean()),
        relative_l2=float(np.linalg.norm(delta.ravel())) / max(float(np.linalg.norm(reference.astype(np.float64).ravel())), np.finfo(np.float64).tiny),
        outside_tolerance=int(np.count_nonzero(delta > limit)), within_tolerance=bool(np.all(delta <= limit)),
        argmax_agreement=float(np.mean(actual.argmax(axis=-1) == reference.argmax(axis=-1))))


def verify(args):
    import gc
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from transformers import PhiConfig
    from validate_native_magvit import compare_decodes, preview

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    native_report = read_json(args.output / "native_result.json")
    phi = load_module("pipeline_verification_phi", args.root / "official/models/phi.py")
    config = PhiConfig.from_dict(read_json(args.reference_run / "phi_config.json"))
    config._attn_implementation = "eager"
    with torch.device("cuda"):
        model = phi.PhiForCausalLM(config)
    weights = load_file(str(args.root / "weights/show-o-512x512/pytorch_model.safetensors"), device="cpu")
    print(model.load_state_dict({key.removeprefix("showo."): value for key, value in weights.items()}, strict=True), flush=True)
    del weights
    model.eval().requires_grad_(False)
    with np.load(args.output / "official_inputs.npz", allow_pickle=False) as fixture:
        t2i_attention = fixture["attention_mask"].copy()
    rows = []
    for task, names in (("t2i", native_report["t2i_replay"]), ("mmu", native_report["mmu_replay"])):
        for name in names:
            print("PYTORCH_SAME_INPUT_REPLAY", name, flush=True)
            with np.load(args.output / name, allow_pickle=False) as fixture:
                ids = torch.from_numpy(fixture["input_ids"].astype(np.int64)).cuda()
                raw_mask = t2i_attention if task == "t2i" else fixture["attention_mask"].copy()
                attention = torch.from_numpy(raw_mask).cuda()
                expected = fixture["guided_logits" if task == "t2i" else "logits"].copy()
            with torch.inference_mode():
                logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits
                if task == "t2i":
                    branches = logits[:, -1025:-1, 50305:58497]
                    cond, uncond = branches.chunk(2, dim=0)
                    actual = (6.0 * cond - 5.0 * uncond).cpu().numpy()
                    del branches, cond, uncond
                else:
                    actual = logits[:, -1, :].cpu().numpy()
            metrics = score_comparison(expected, actual)
            rows.append(dict(task=task, input=name, metrics=metrics))
            del logits, ids, attention, expected, actual
            gc.collect()
            torch.cuda.empty_cache()
            print("REPLAY_CHECK", name, json.dumps(metrics), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    package = types.ModuleType("pipeline_verification_magvit")
    package.__path__ = [str(args.root / "official/models")]
    sys.modules[package.__name__] = package
    cls = importlib.import_module(package.__name__ + ".modeling_magvitv2").MAGVITv2
    model = cls()
    model.load_state_dict(load_file(str(args.root / "weights/magvitv2/pytorch_model.safetensors")), strict=True)
    model.eval().requires_grad_(False).cuda()
    with np.load(args.output / "image_tokens.npz", allow_pickle=False) as archive:
        codes = torch.from_numpy(archive["codes"].astype(np.int64)).cuda()
    with torch.inference_mode():
        reference = model.decode_code(codes).cpu().numpy()
    native_decoded = np.load(args.output / "native_decoded.npy", allow_pickle=False)
    decoding = compare_decodes(native_decoded, reference)
    Image.fromarray(preview(reference)).save(args.output / "pytorch_same_codes.png")
    pair = Image.new("RGB", (1024, 512))
    pair.paste(Image.fromarray(preview(reference)), (0, 0))
    pair.paste(Image.fromarray(preview(native_decoded)), (512, 0))
    pair.save(args.output / "comparison.png")
    functional = native_report["mmu"]["exact_token_match"] and decoding["passed"] and all(
        row["metrics"]["argmax_agreement"] == 1.0 for row in rows)
    strict = all(row["metrics"]["within_tolerance"] for row in rows)
    result = dict(status="completed", functional_checks_passed=bool(functional),
        sampled_forward_checks_within_tolerance=bool(strict), replay=rows, same_codes_decode=decoding,
        mmu=native_report["mmu"], native_process=native_report,
        original_83_checks=dict(checked=83, passed=78, failed=5, rerun=False,
            source=str(args.alignment_run), atol=ATOL, rtol=RTOL),
        comparison_layout="left: independent PyTorch decode of native generated codes; right: native Jittor decode of the same codes",
        limits="One 512px T2I and one dog MMU case; replay only first/middle/last forwards, no claim about all steps or inputs. Independent Jittor RNG: identical seed does not imply identical PyTorch T2I codes. Existing 83-check failures retained.")
    write_json(args.output / "native_pipeline_report.json", result)
    print("FULL_NATIVE_PIPELINE_VERIFICATION_COMPLETE", args.output,
          "functional_checks_passed", bool(functional), "sampled_forward_checks_within_tolerance", bool(strict), flush=True)


def preflight(args):
    used = int(subprocess.check_output(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=memory.used",
        "--format=csv,noheader,nounits"], text=True, timeout=10).strip())
    if used > 1024:
        raise RuntimeError("GPU occupied; stop without terminating jobs: %s MiB" % used)
    commit = subprocess.check_output(["git", "-C", str(args.root / "official"), "rev-parse", "HEAD"], text=True).strip()
    if commit != COMMIT:
        raise RuntimeError("Official revision differs")
    subprocess.run(["git", "-C", str(args.root / "official"), "diff", "--quiet", "HEAD", "--",
        "models", "training/prompting_utils.py", "training/utils.py"], check=True)
    reference = read_json(args.reference_run / "run_config.json")
    candidate = read_json(args.candidate_run / "run_config.json")
    summary = read_json(args.candidate_run / "candidate_summary.json")
    for saved in (reference, candidate):
        if saved["commit"] != COMMIT or saved["checkpoint_sha256"] != SHOWO_SHA:
            raise RuntimeError("Reference/candidate metadata differs")
    if file_hash(args.reference_run / "native_phi.py") != reference["native_source_sha256"]:
        raise RuntimeError("Native Transformer snapshot SHA differs")
    if file_hash(args.reference_run / "native_phi.py") != candidate["native_sha256"]:
        raise RuntimeError("Candidate native Transformer SHA differs")
    if Path(candidate["reference"]).resolve() != args.reference_run.resolve():
        raise RuntimeError("Candidate belongs to a different reference run")
    if read_json(args.reference_run / "reference_status.json")["status"] != "passed":
        raise RuntimeError("Original reference preparation has not passed")
    if file_hash(args.candidate_run / "operator_script.py") != candidate["runner_sha256"]:
        raise RuntimeError("Operator candidate source SHA differs")
    if "flat_linear" not in summary["local_passed_variants"]:
        raise RuntimeError("Verified flat_linear candidate missing")
    operators = read_json(args.candidate_run / "flat_linear/operators.json")
    if operators["checked"] != 24 or operators["outside_tolerance"] != 0:
        raise RuntimeError("Verified flat_linear operator checks differ")
    alignment = read_json(args.alignment_run / "alignment/alignment.json")
    if alignment["checked"] != 83 or alignment["failed"] != 5:
        raise RuntimeError("Expected existing 78/83 alignment record")
    rows = [json.loads(line) for line in (args.alignment_run / "alignment/alignment.jsonl").read_text().splitlines() if line.strip()]
    if len(rows) != 83 or not all(row["finite"] for row in rows):
        raise RuntimeError("Existing full alignment contains nonfinite or incomplete results")
    sampler = read_json(args.sampler_run / "sampler_contract_report.json")
    if sampler["passed"] is not True:
        raise RuntimeError("Native sampler contract has not passed")
    magvit = read_json(args.magvit_run / "magvit_alignment_report.json")
    if not magvit["all_encode_exact"] or not magvit["all_decode_within_tolerance"]:
        raise RuntimeError("Native MagViT paired validation has not passed")
    tools = Path(__file__).resolve().parent
    sampler_config = read_json(args.sampler_run / "run_config.json")
    for name in ("jittor_sampling.py", "validate_native_sampling.py"):
        if file_hash(tools / name) != sampler_config["source_sha256"][name]:
            raise RuntimeError("Current sampler sources differ from passed contract: " + name)
    magvit_config = read_json(args.magvit_run / "run_config.json")
    if file_hash(tools / "jittor_magvit.py") != magvit_config["native_sha256"]:
        raise RuntimeError("Current MagViT source differs from passed validation")
    if file_hash(tools / "validate_native_magvit.py") != magvit_config["script_sha256"]:
        raise RuntimeError("Current MagViT loader differs from passed validation")
    for name, expected in (("show-o-512x512", SHOWO_SHA), ("magvitv2", MAGVIT_SHA)):
        print("VERIFYING_NATIVE_PIPELINE_WEIGHT", name, flush=True)
        if file_hash(args.root / "weights" / name / "pytorch_model.safetensors") != expected:
            raise RuntimeError("Weight SHA differs: " + name)


def main(argv=None):
    root = Path.home() / "Show-o-Jittor"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=64, choices=range(1, 65))
    parser.add_argument("--candidate-run", type=Path, default=root / "results/official512-operator-candidates-20261027-022054-436199")
    parser.add_argument("--reference-run", type=Path, default=root / "results/official512-jittor-align-20261027-004837-924843")
    parser.add_argument("--alignment-run", type=Path, default=root / "results/jittor-t2i-512-20261027-023446-631846")
    parser.add_argument("--mmu-baseline", type=Path, default=root / "results/jittor-mmu-512-20261027-030340-162478")
    parser.add_argument("--magvit-run", type=Path, default=root / "results/native-magvit-paired-20261114-145717-007496")
    parser.add_argument("--sampler-run", type=Path, default=root / "results/native-sampler-contract-20261114-151720-646497")
    parser.add_argument("--worker", choices=("prepare", "native", "verify"), help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.root = root
    if args.worker:
        if args.output is None:
            parser.error("worker requires --output")
        {"prepare": prepare, "native": native, "verify": verify}[args.worker](args)
        return
    preflight(args)
    args.output = root / "results" / ("full-native-512-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    args.output.mkdir(parents=True, exist_ok=False)
    tools = Path(__file__).resolve().parent
    files = ("native_pipeline.py", "infer_full_jittor.py", "jittor_sampling.py",
             "jittor_magvit.py", "validate_native_magvit.py", "validate_native_sampling.py")
    for name in files:
        shutil.copyfile(tools / name, args.output / name)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(official_commit=COMMIT, showo_sha256=SHOWO_SHA, magvit_sha256=MAGVIT_SHA,
        atol=ATOL, rtol=RTOL, original_83_checks_rerun=False,
        source_sha256={name: file_hash(tools / name) for name in files})
    write_json(args.output / "run_config.json", config)
    print("RUN_DIR", args.output, flush=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), NVIDIA_TF32_OVERRIDE="0", PYTHONUNBUFFERED="1")
    common = [sys.executable, "-u", str(tools / "native_pipeline.py"), "--output", str(args.output),
        "--gpu", str(args.gpu), "--seed", str(args.seed), "--prompt", args.prompt,
        "--max-new-tokens", str(args.max_new_tokens)]
    for option in ("candidate_run", "reference_run", "alignment_run", "mmu_baseline", "magvit_run", "sampler_run"):
        common += ["--" + option.replace("_", "-"), str(getattr(args, option))]
    for worker in ("prepare", "native", "verify"):
        started = time.monotonic()
        print("NATIVE_PIPELINE_STAGE_START", worker, flush=True)
        with (args.output / (worker + ".log")).open("x", encoding="utf-8") as log:
            with subprocess.Popen(common + ["--worker", worker], env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
        if code != 0:
            write_json(args.output / "run_status.json", dict(status="failed", worker=worker, returncode=code))
            raise SystemExit(code)
        print("NATIVE_PIPELINE_STAGE_DONE", worker, "seconds", round(time.monotonic() - started, 2), flush=True)
    report = read_json(args.output / "native_pipeline_report.json")
    write_json(args.output / "run_status.json", dict(status="completed",
        functional_checks_passed=report["functional_checks_passed"],
        sampled_forward_checks_within_tolerance=report["sampled_forward_checks_within_tolerance"],
        original_alignment_passed=78, original_alignment_checked=83))
    if not report["functional_checks_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
