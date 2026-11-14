"""Server-only Show-o 1 native Transformer with official T2I/MMU sampling."""

import argparse
import ast
from datetime import datetime
import importlib
import importlib.util
import json
from multiprocessing.connection import Client, Listener
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import types


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def mmu_input_path(args):
    if args.mmu_resolution == 512:
        return args.output / "mmu_input.npz"
    return args.reference_run / "reference/mmu/input.npz"


def prepare_mmu_512(args, helpers):
    import numpy as np
    import torch
    from PIL import Image
    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    from torchvision import transforms
    from transformers import AutoTokenizer

    root = Path.home() / "Show-o-Jittor"
    config = OmegaConf.load(root / "official/configs/showo_demo_512x512.yaml")
    assert config.dataset.params.resolution == 512 and config.model.showo.num_vq_tokens == 1024
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    package = types.ModuleType("mmu_image_encoder")
    package.__path__ = [str(root / "official/models")]
    sys.modules[package.__name__] = package
    Magvit = importlib.import_module("mmu_image_encoder.modeling_magvitv2").MAGVITv2
    model = Magvit()
    weight = root / "weights/magvitv2/pytorch_model.safetensors"
    assert helpers.file_hash(weight) == "79656bf974d99c215465a61df5e93b97e03282b0941f62f6294719527873b778"
    print(model.load_state_dict(load_file(str(weight)), strict=True), flush=True)
    model.eval().requires_grad_(False).cuda()
    source = (root / "official/training/utils.py").read_text(encoding="utf-8")
    functions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "image_transform"]
    assert len(functions) == 1 and not functions[0].decorator_list
    namespace = {"transforms": transforms}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "official_image_transform", "exec"), namespace)
    image_path = root / "official/mmu_validation/dog.png"
    with Image.open(image_path) as image:
        pixels = namespace["image_transform"](image.convert("RGB"), resolution=512).unsqueeze(0).cuda()
    assert pixels.shape == (1, 3, 512, 512) and torch.isfinite(pixels).all().item()
    with torch.inference_mode():
        codes = model.get_code(pixels).cpu().numpy().copy()
    assert codes.shape == (1, 1024) and codes.min() >= 0 and codes.max() < 8192
    preview = ((pixels[0] + 1) / 2).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    Image.fromarray((preview * 255).astype(np.uint8)).save(args.output / "mmu_input.png")
    del model, pixels
    torch.cuda.empty_cache()
    prompting = load_module("mmu_image_prompting", root / "official/training/prompting_utils.py")
    tokenizer = AutoTokenizer.from_pretrained(str(root / "weights/phi-1_5-original"), local_files_only=True)
    prompt = prompting.UniversalPrompting(tokenizer, max_text_len=128, cond_dropout_prob=0.0)
    assert len(tokenizer) == 50305
    special = lambda name: int(prompt.sptids_dict[name].item())
    question = "Describe this image in detail."
    prefix = [special("<|mmu|>"), special("<|soi|>")] + (codes[0] + len(tokenizer)).tolist()
    prefix += [special("<|eoi|>"), special("<|sot|>")]
    assert special("<|sot|>") == tokenizer.bos_token_id and special("<|eot|>") == tokenizer.eos_token_id
    ids = torch.tensor([prefix + tokenizer("USER: \n" + question + " ASSISTANT:")["input_ids"]], dtype=torch.long)
    mask = prompting.create_attention_mask_for_mmu(ids, eoi_id=special("<|eoi|>")).float()
    np.savez_compressed(mmu_input_path(args), input_ids=ids.numpy(), attention_mask=mask.numpy(), image_codes=codes)
    helpers.write_json(args.output / "mmu_input.json", dict(
        resolution=512, image_tokens=1024, source_image=str(image_path), source_sha256=helpers.file_hash(image_path),
        input_sha256=helpers.file_hash(mmu_input_path(args)), question=question,
        preprocessing=ast.get_source_segment(source, functions[0]),
        scope="Frozen official PyTorch MagViT encoding; identical saved input for both Transformers"))
    print("MAGVIT_512_MMU_INPUT_READY", list(ids.shape), "image_tokens:", codes.size, flush=True)


def native_worker(args, helpers):
    imports = {"before_dependencies": helpers.torch_cuda_state()}
    import numpy as np
    import jittor as jt
    from safetensors import safe_open

    imports["after_jittor"] = helpers.check_torch_cuda("native imports")
    jt.flags.use_cuda = 1
    jt.flags.amp_level = 0
    jt.set_global_seed(args.seed)
    port = load_module("generation_native_phi", args.reference_run / "native_phi.py")
    config = json.loads((args.reference_run / "phi_config.json").read_text())
    model = port.model_class(jt, np)(config)
    model.eval()
    state = model.state_dict()
    assert sum(int(np.prod(value.shape)) for value in state.values()) == helpers.PARAMETERS
    weight = Path.home() / "Show-o-Jittor/weights/show-o-512x512/pytorch_model.safetensors"
    with safe_open(str(weight), framework="np") as archive:
        mapping = {key.removeprefix("showo."): key for key in archive.keys()}
        assert set(mapping) == set(state)
        for index, (key, parameter) in enumerate(state.items(), 1):
            array = archive.get_tensor(mapping[key])
            assert array.dtype == np.float32 and tuple(array.shape) == tuple(parameter.shape), key
            parameter.assign(jt.array(array)).stop_grad()
            assert np.array_equal(parameter.numpy(), array), key
            if index == 1 or index % 48 == 0 or index == len(state):
                print("FULL_JITTOR_LOAD", index, "/", len(state), key, flush=True)
    del state
    print("FULL_JITTOR_WEIGHTS_LOADED_EXACTLY", helpers.PARAMETERS, flush=True)

    candidate_source = (args.candidate_run / "operator_script.py").read_text(encoding="utf-8")
    tree = ast.parse(candidate_source)
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "flat_linear"]
    assert len(functions) == 1 and not functions[0].decorator_list
    namespace = {"jt": jt}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "verified_flat_linear", "exec"), namespace)
    type(model.lm_head).execute = namespace["flat_linear"]
    helpers.write_json(args.output / "effective_linear.json", dict(
        variant="flat_linear", source=ast.get_source_segment(candidate_source, functions[0]),
        tested_candidate_run=str(args.candidate_run), candidate_sha256=helpers.file_hash(args.candidate_run / "operator_script.py")))

    if args.task == "mmu":
        report = json.loads((args.baseline_run / "alignment/alignment.json").read_text())
        print("REUSING_RECORDED_FULL_ALIGNMENT", args.baseline_run, flush=True)
    else:
        alignment = args.output / "alignment"
        alignment.mkdir()
        report = helpers.forward_alignment(args, jt, np, model, alignment, imports)
    assert report["checked"] == 83
    rows_path = (args.baseline_run if args.task == "mmu" else args.output) / "alignment/alignment.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    if not all(row["finite"] for row in rows):
        raise RuntimeError("Non-finite alignment values; generation stopped")
    print("FULL_FORWARD_STATUS", report["status"], "failed:", report["failed"], "/ 83", flush=True)
    if report["status"] != "passed":
        print("GENERATION_SCOPE: diagnostic demonstration; original numerical acceptance remains failed", flush=True)

    if args.task == "t2i":
        mask = None
    else:
        with np.load(mmu_input_path(args), allow_pickle=False) as fixture:
            initial_length = fixture["input_ids"].shape[1]
    calls = 0
    started = time.perf_counter()
    # The socket lives in a new owner-only /tmp directory, never on the shared NFS.
    with Listener(str(args.bridge), family="AF_UNIX") as listener:
        helpers.write_json(args.output / "native_ready.json", dict(status="ready", bridge=str(args.bridge)))
        with listener.accept() as connection:
            while True:
                request = connection.recv()
                if request["command"] == "stop":
                    assert (calls == 50 if args.task == "t2i" else 1 <= calls <= args.max_new_tokens), "Incomplete sampling run"
                    connection.send(dict(status="stopped", calls=calls))
                    break
                limit = 50 if args.task == "t2i" else args.max_new_tokens
                assert request["command"] == "forward" and calls < limit
                raw = request["input_ids"]
                assert isinstance(raw, np.ndarray)
                if args.task == "t2i":
                    assert raw.shape == (2, 1155)
                    raw_mask = request["attention_mask"]
                    assert raw_mask.shape == (2, 1155, 1155)
                    assert raw_mask.dtype == np.float32 and np.isfinite(raw_mask).all()
                    mask = jt.array(raw_mask).stop_grad()
                else:
                    assert raw.shape == (1, initial_length + calls)
                    raw_mask = request["attention_mask"]
                    assert raw_mask.shape == (1, 1, raw.shape[1], raw.shape[1])
                    assert raw_mask.dtype == np.float32 and np.isfinite(raw_mask).all()
                    mask = jt.array(raw_mask).stop_grad()
                assert np.issubdtype(raw.dtype, np.integer) and raw.min() >= 0 and raw.max() < 58498
                with jt.no_grad():
                    ids = jt.array(raw.astype(np.int32)).stop_grad()
                    logits, trace = model(ids, mask)
                    selected = logits[:, -1025:-1, 50305:58497] if args.task == "t2i" else logits[:, -1:, :]
                    result = selected.numpy()
                    expected_shape = (2, 1024, 8192) if args.task == "t2i" else (1, 1, 58498)
                    assert result.shape == expected_shape
                    assert result.dtype == np.float32 and np.isfinite(result).all()
                    connection.send(result)
                    del ids, logits, trace, selected, result
                calls += 1
                jt.sync_all(True)
                jt.gc()
                helpers.check_torch_cuda("native generation " + str(calls))
                if calls == 1 or calls % 10 == 0:
                    print("JITTOR_" + args.task.upper() + "_FORWARD", calls, "/", limit, flush=True)
    helpers.write_json(args.output / "native_generation.json", dict(
        status="completed", task=args.task, calls=calls, parameters=helpers.PARAMETERS, linear_variant="flat_linear",
        elapsed_seconds=time.perf_counter() - started,
        torch_cuda_state=helpers.check_torch_cuda("native generation complete"),
        alignment_status=report["status"], alignment_failed=report["failed"],
        alignment_source=str(args.baseline_run) if args.task == "mmu" else str(args.output),
        scope="All full-depth Transformer forwards use native Jittor; sampling is a separate official PyTorch process"))


def mmu_components(args):
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    root = Path.home() / "Show-o-Jittor"
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    package = types.ModuleType("mmu_official")
    package.__path__ = [str(root / "official/models")]
    sys.modules[package.__name__] = package
    Showo = importlib.import_module("mmu_official.modeling_showo").Showo
    prompting = load_module("mmu_prompting", root / "official/training/prompting_utils.py")
    tokenizer = AutoTokenizer.from_pretrained(str(root / "weights/phi-1_5-original"), local_files_only=True)
    prompting.UniversalPrompting(tokenizer, max_text_len=128, cond_dropout_prob=0.0)
    assert len(tokenizer) == 50305
    with np.load(mmu_input_path(args), allow_pickle=False) as fixture:
        ids = torch.from_numpy(fixture["input_ids"].astype(np.int64)).cuda()
        mask = torch.from_numpy(fixture["attention_mask"].astype(np.float32)).cuda()
    image_tokens = (mmu_input_path(args) if args.mmu_resolution == 512 else
                    root / "results/magvit-reconstruction-20261026-203006-738585/image_tokens.npz")
    with np.load(image_tokens, allow_pickle=False) as saved:
        dog = saved["image_codes" if args.mmu_resolution == 512 else "dog"].copy()
    assert dog.shape == (1, (args.mmu_resolution // 16) ** 2) and dog.min() >= 0 and dog.max() < 8192
    question_ids = tokenizer("USER: \nDescribe this image in detail. ASSISTANT:")["input_ids"]
    expected = [50301, 50296] + (dog[0] + len(tokenizer)).tolist() + [50297, tokenizer.bos_token_id] + question_ids
    assert np.array_equal(ids.cpu().numpy(), np.asarray([expected])), "Expected original dog image/question fixture"
    assert torch.equal(mask, prompting.create_attention_mask_for_mmu(ids, eoi_id=50297).float())
    assert ids.shape[1] + args.max_new_tokens <= 2048
    return np, torch, Showo, tokenizer, ids, mask


def mmu_reference_worker(args, helpers):
    if args.mmu_resolution == 512:
        prepare_mmu_512(args, helpers)
    np, torch, Showo, tokenizer, ids, mask = mmu_components(args)
    from safetensors.torch import load_file
    from transformers import PhiConfig

    root = Path.home() / "Show-o-Jittor"
    phi = load_module("mmu_reference_phi", root / "official/models/phi.py")
    config = PhiConfig.from_dict(json.loads((args.reference_run / "phi_config.json").read_text()))
    config._attn_implementation = "eager"
    with torch.device("cuda"):
        model = phi.PhiForCausalLM(config)
    weights = load_file(str(root / "weights/show-o-512x512/pytorch_model.safetensors"), device="cpu")
    assert all(key.startswith("showo.") for key in weights)
    print(model.load_state_dict({key.removeprefix("showo."): value for key, value in weights.items()}, strict=True), flush=True)
    del weights
    model.eval().requires_grad_(False)
    assert sum(p.numel() for p in model.parameters()) == helpers.PARAMETERS
    assert all(p.dtype == torch.float32 for p in model.parameters())
    next_logits, margins = [], []

    class ReferenceForward:
        config = types.SimpleNamespace(w_clip_vit=False)

        def __call__(self, input_ids, input_embeddings=None, attention_mask=None):
            assert input_embeddings is None
            # The official sampler squeezes the growing mask after its first step.
            length = input_ids.shape[1]
            attention_mask = attention_mask.reshape((1, 1, length, length))
            result = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
            last = result[:, -1, :]
            next_logits.append(last.cpu().numpy().copy())
            top = last.topk(2).values
            margins.append(float((top[0, 0] - top[0, 1]).item()))
            if len(next_logits) == 1 or len(next_logits) % 10 == 0:
                print("PYTORCH_MMU_FORWARD", len(next_logits), "/", args.max_new_tokens, flush=True)
            return result

    with torch.inference_mode():
        generated = Showo.mmu_generate(ReferenceForward(), idx=ids, attention_mask=mask,
                                      max_new_tokens=args.max_new_tokens, top_k=1, eot_token=tokenizer.eos_token_id)
    generated_ids = torch.stack(generated).cpu().tolist()
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    directory = args.output / "mmu_reference"
    directory.mkdir()
    scores = np.concatenate(next_logits, axis=0)
    assert scores.shape == (len(generated_ids), 58498) and np.isfinite(scores).all()
    np.save(directory / "next_logits.npy", scores, allow_pickle=False)
    report = dict(status="completed", generated_ids=generated_ids, answer=answer, steps=len(generated_ids),
                  eos_terminated=generated_ids[-1] == tokenizer.eos_token_id,
                  min_top1_margin=min(margins), checkpoint_sha256=helpers.SHA256,
                  dtype="float32", attention="eager", image_resolution=args.mmu_resolution,
                  image_tokens=(args.mmu_resolution // 16) ** 2,
                  scope="Same 512 weights and dog input as native MMU; official generation loop, top_k=1")
    helpers.write_json(directory / "result.json", report)
    print("PYTORCH_MMU_ANSWER:", answer, flush=True)


def mmu_sample_worker(args, helpers):
    np, torch, Showo, tokenizer, ids, mask = mmu_components(args)
    reference = json.loads((args.output / "mmu_reference/result.json").read_text())
    scores = np.load(args.output / "mmu_reference/next_logits.npy", mmap_mode="r", allow_pickle=False)
    base_ids = ids.cpu().numpy().copy()
    comparisons = []
    calls = [0]
    with Client(str(args.bridge), family="AF_UNIX") as connection:
        class NativeForward:
            config = types.SimpleNamespace(w_clip_vit=False)

            def __call__(self, input_ids, input_embeddings=None, attention_mask=None):
                assert input_embeddings is None
                raw = input_ids.cpu().numpy()
                length = raw.shape[1]
                raw_mask = attention_mask.cpu().numpy().reshape((1, 1, length, length)).astype(np.float32)
                end = int(np.flatnonzero(raw[0] == 50297)[0])
                positions = np.arange(length)
                allowed = (positions[None, :] <= positions[:, None]) | (positions[None, :] <= end)
                assert np.array_equal(raw_mask == 0, allowed[None, None]), "Official growing MMU mask differs from independent oracle"
                connection.send(dict(command="forward", input_ids=raw, attention_mask=raw_mask))
                result = connection.recv()
                assert result.shape == (1, 1, 58498) and result.dtype == np.float32 and np.isfinite(result).all()
                step = calls[0]
                expected_ids = np.concatenate((base_ids, np.asarray([reference["generated_ids"][:step]], dtype=np.int64)), axis=1)
                shared = step < len(scores) and np.array_equal(raw, expected_ids)
                if shared:
                    row = helpers.compare(np, result[:, -1, :], scores[step:step + 1], "mmu_decode", str(step + 1))
                else:
                    row = dict(case="mmu_decode", name=str(step + 1), status="not_compared_different_prefix")
                row["shared_prefix"] = bool(shared)
                comparisons.append(row)
                calls[0] += 1
                return torch.from_numpy(result).cuda()

        with torch.inference_mode():
            generated = Showo.mmu_generate(NativeForward(), idx=ids, attention_mask=mask,
                                          max_new_tokens=args.max_new_tokens, top_k=1, eot_token=tokenizer.eos_token_id)
        generated_ids = torch.stack(generated).cpu().tolist()
        connection.send(dict(command="stop"))
        assert connection.recv() == dict(status="stopped", calls=calls[0])
    exact = generated_ids == reference["generated_ids"]
    common = sum(a == b for a, b in zip(generated_ids, reference["generated_ids"]))
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    report = dict(status="generation_completed", question="Describe this image in detail.", answer=answer,
                  reference_answer=reference["answer"], generated_ids=generated_ids,
                  reference_generated_ids=reference["generated_ids"], exact_token_match=exact,
                  token_agreement=common / max(len(generated_ids), len(reference["generated_ids"])),
                  eos_terminated=generated_ids[-1] == tokenizer.eos_token_id, max_new_tokens=args.max_new_tokens,
                  source_image="official/mmu_validation/dog.png", image_resolution=args.mmu_resolution,
                  image_tokens=(args.mmu_resolution // 16) ** 2,
                  checkpoint="show-o-512x512", checkpoint_sha256=helpers.SHA256,
                  next_logit_comparisons=comparisons,
                  shared_prefix_checks=sum(row["shared_prefix"] for row in comparisons),
                  shared_prefix_checks_failed=sum(row["shared_prefix"] and not row["passed"] for row in comparisons),
                  scope="Single dog image; full Jittor Transformer with official PyTorch top_k=1 sampler; "
                        "same-weight eager reference; not a quality benchmark or full numerical acceptance")
    helpers.write_json(args.output / "mmu_result.json", report)
    print("PYTORCH_MMU_ANSWER:", reference["answer"], flush=True)
    print("JITTOR_MMU_ANSWER:", answer, flush=True)
    print("MMU_EXACT_TOKEN_MATCH:", exact, "token_agreement:", report["token_agreement"], flush=True)
    print("MMU_NEXT_LOGIT_CHECKS:", report["shared_prefix_checks"], "failed:", report["shared_prefix_checks_failed"], flush=True)
    print("MMU_RESULT:", args.output / "mmu_result.json", flush=True)


def sample_worker(args, helpers):
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    root = Path.home() / "Show-o-Jittor"
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    package = types.ModuleType("bridge_official")
    package.__path__ = [str(root / "official/models")]
    sys.modules[package.__name__] = package
    Showo = importlib.import_module("bridge_official.modeling_showo").Showo
    config = OmegaConf.load(root / "official/configs/showo_demo_512x512.yaml")
    assert config.model.showo.num_vq_tokens == 1024
    assert config.model.showo.llm_vocab_size + config.model.showo.num_new_special_tokens == 50305
    assert config.dataset.preprocessing.max_seq_length == 128
    prompting = load_module("sample_prompting", root / "official/training/prompting_utils.py")
    tokenizer = AutoTokenizer.from_pretrained(str(root / "weights/phi-1_5-original"), local_files_only=True)
    prompt = prompting.UniversalPrompting(tokenizer, max_text_len=128, cond_dropout_prob=0.0)
    masked = torch.full((1, 1024), 58497, dtype=torch.long, device="cuda")
    ids, _ = prompt(([args.prompt], masked), "t2i_gen")
    uncond, _ = prompt(([""], masked), "t2i_gen")
    mask = prompting.create_attention_mask_predict_next(
        torch.cat([ids, uncond]), pad_id=tokenizer.pad_token_id,
        soi_id=int(prompt.sptids_dict["<|soi|>"].item()),
        eoi_id=int(prompt.sptids_dict["<|eoi|>"].item()), rm_pad_in_image=True).float()
    ids, uncond, mask = ids.cuda(), uncond.cuda(), mask.cuda()
    assert ids.shape == (2, 1155) and torch.all(ids[:, -1025:-1] == 58497).item()
    progress = []
    with Client(str(args.bridge), family="AF_UNIX") as connection:
        class NativeForward:
            config = types.SimpleNamespace(mask_token_id=58497)

            def __call__(self, input_ids, attention_mask=None):
                assert attention_mask is mask and input_ids.shape == (2, 1155)
                raw = input_ids.detach().cpu().numpy()
                connection.send(dict(command="forward", input_ids=raw,
                                     attention_mask=mask.detach().cpu().numpy()))
                image_logits = connection.recv()
                assert image_logits.shape == (2, 1024, 8192) and image_logits.dtype == np.float32
                assert np.isfinite(image_logits).all()
                # The pinned sampler reads only this slice after CFG. Unused entries are zero.
                logits = torch.zeros((2, 1155, 58498), dtype=torch.float32, device="cuda")
                logits[:, -1025:-1, 50305:58497] = torch.from_numpy(image_logits).cuda()
                number = len(progress) + 1
                progress.append(dict(step=number, remaining_masks=int((input_ids[0, -1025:-1] == 58497).sum().item())))
                return logits

        with torch.inference_mode():
            codes = Showo.t2i_generate(
                NativeForward(), ids[:1].clone(), uncond_input_ids=ids[1:].clone(), attention_mask=mask,
                timesteps=50, guidance_scale=5.0, temperature=1.0,
                generator=torch.Generator(device="cuda").manual_seed(args.seed), config=config)
        assert codes.shape == (1, 1024) and codes.min().item() >= 0 and codes.max().item() < 8192
        connection.send(dict(command="stop"))
        response = connection.recv()
        assert response == dict(status="stopped", calls=50)
    np.savez_compressed(args.output / "image_tokens.npz", codes=codes.cpu().numpy())
    baseline_path = root / "results/official-t2i-512-20261027-001553-017691"
    token_agreement = None
    settings = dict(prompt=args.prompt)
    if args.prompt == "A photo of a golden retriever sitting in a field of yellow flowers." and args.seed == 42:
        with np.load(baseline_path / "image_tokens.npz", allow_pickle=False) as saved:
            token_agreement = float((saved["codes"] == codes.cpu().numpy()).mean())
        settings = json.loads((baseline_path / "settings.json").read_text())
    helpers.write_json(args.output / "sampling.json", dict(
        status="completed", seed=args.seed, timesteps=50, guidance_scale=5.0, temperature=1.0,
        progress=progress, torch_version=torch.__version__, prompt=settings["prompt"],
        original_sdpa_baseline_token_agreement=token_agreement,
        peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
        scope="Unmodified pinned official t2i_generate and CUDA random sampler; Transformer logits supplied by Jittor over a server-local socket; "
              "token agreement with earlier SDPA run is observational, not a strict acceptance test"))
    print("FULL_JITTOR_IMAGE_TOKENS_READY", "SDPA_baseline_token_agreement:", token_agreement, flush=True)


def decode_worker(args, helpers):
    import numpy as np
    import torch
    from PIL import Image
    from safetensors.torch import load_file

    root = Path.home() / "Show-o-Jittor"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    package = types.ModuleType("bridge_decoder")
    package.__path__ = [str(root / "official/models")]
    sys.modules[package.__name__] = package
    Magvit = importlib.import_module("bridge_decoder.modeling_magvitv2").MAGVITv2
    model = Magvit()
    model.load_state_dict(load_file(str(root / "weights/magvitv2/pytorch_model.safetensors")), strict=True)
    model.eval().requires_grad_(False).cuda()
    with np.load(args.output / "image_tokens.npz", allow_pickle=False) as saved:
        codes = torch.from_numpy(saved["codes"].astype(np.int64)).cuda()
    with torch.inference_mode():
        decoded = model.decode_code(codes)
    assert decoded.shape == (1, 3, 512, 512) and torch.isfinite(decoded).all().item()
    pixels = ((decoded[0] + 1) / 2).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    image = Image.fromarray((pixels * 255).astype(np.uint8))
    image.save(args.output / "generated.png")
    previous = root / "results/official-t2i-512-20261027-001553-017691/generated.png"
    if args.prompt == "A photo of a golden retriever sitting in a field of yellow flowers." and args.seed == 42 and previous.is_file():
        with Image.open(previous) as original:
            assert original.size == (512, 512)
            comparison = Image.new("RGB", (1024, 512))
            comparison.paste(original.convert("RGB"), (0, 0))
            comparison.paste(image, (512, 0))
            comparison.save(args.output / "comparison.png")
    report = json.loads((args.output / "native_generation.json").read_text())
    report.update(generated_image=str(args.output / "generated.png"),
                  comparison=(str(args.output / "comparison.png") if (args.output / "comparison.png").is_file() else None),
                  comparison_layout=("left: earlier official PyTorch SDPA; right: Jittor Transformer with official sampler and MagViT"
                                     if (args.output / "comparison.png").is_file() else None),
                  scope="512x512 single-sample Jittor Transformer generation demonstration; official PyTorch sampler and MagViT retained; "
                        "not complete native-Jittor pipeline, training reproduction or quality benchmark")
    helpers.write_json(args.output / "generation_report.json", report)
    print("GENERATED_IMAGE:", args.output / "generated.png", flush=True)
    print("COMPARISON_IMAGE:", args.output / "comparison.png", flush=True)


def wait_ready(process, output):
    deadline = time.monotonic() + 1800
    while not (output / "native_ready.json").is_file():
        if process.poll() is not None:
            raise RuntimeError("Native stage exited before sampler bridge was ready; inspect native.log")
        if time.monotonic() >= deadline:
            raise TimeoutError("Native initialization/alignment exceeded 30 minutes; inspect native.log")
        time.sleep(0.25)


def relay_native(process, log):
    for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line)
        log.flush()


def main():
    if "--native-pipeline" in sys.argv:
        from native_pipeline import main as native_pipeline_main
        return native_pipeline_main([value for value in sys.argv[1:] if value != "--native-pipeline"])
    root = Path.home() / "Show-o-Jittor"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--task", choices=("t2i", "mmu"), default="t2i")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--prompt", default="A photo of a golden retriever sitting in a field of yellow flowers.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mmu-resolution", type=int, choices=(256, 512), default=256)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, default=root / "results/jittor-t2i-512-20261027-023446-631846")
    parser.add_argument("--helpers-run", type=Path, default=root / "results/official512-precision-20261027-015419-310675")
    parser.add_argument("--reference-run", type=Path, default=root / "results/official512-jittor-align-20261027-004837-924843")
    parser.add_argument("--worker", choices=("native", "sample", "decode", "mmu_reference"), help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--bridge", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 128:
        parser.error("max-new-tokens must be between 1 and 128")
    if Path(sys.prefix).resolve() != (Path.home() / "envs/showo-jittor").resolve():
        raise RuntimeError("Run in the configured showo-jittor environment")
    helpers = load_module("generation_helpers", args.helpers_run / "alignment_script.py")
    args.precision_run = None
    if args.worker:
        {"native": native_worker, "sample": mmu_sample_worker if args.task == "mmu" else sample_worker,
         "decode": decode_worker, "mmu_reference": mmu_reference_worker}[args.worker](args, helpers)
        return
    for directory in (args.helpers_run, args.reference_run):
        saved = json.loads((directory / "run_config.json").read_text())
        assert saved["commit"] == helpers.COMMIT and saved["checkpoint_sha256"] == helpers.SHA256
        assert saved["atol"] == helpers.ATOL and saved["rtol"] == helpers.RTOL
        assert helpers.file_hash(directory / "alignment_script.py") == saved["runner_sha256"]
        assert helpers.file_hash(directory / "native_phi.py") == saved["native_source_sha256"]
    assert json.loads((args.reference_run / "reference_status.json").read_text())["status"] == "passed"
    candidate = json.loads((args.candidate_run / "candidate_summary.json").read_text())
    saved = json.loads((args.candidate_run / "run_config.json").read_text())
    assert "flat_linear" in candidate["local_passed_variants"]
    assert candidate["tolerances"] == dict(atol=helpers.ATOL, rtol=helpers.RTOL)
    assert Path(saved["reference"]).resolve() == args.reference_run.resolve()
    assert saved["checkpoint_sha256"] == helpers.SHA256 and saved["commit"] == helpers.COMMIT
    assert helpers.file_hash(args.candidate_run / "operator_script.py") == saved["runner_sha256"]
    assert helpers.file_hash(args.reference_run / "native_phi.py") == saved["native_sha256"]
    passed = json.loads((args.candidate_run / "flat_linear/operators.json").read_text())
    assert passed["checked"] == 24 and passed["outside_tolerance"] == 0
    if args.task == "mmu":
        baseline_config = json.loads((args.baseline_run / "run_config.json").read_text())
        baseline = json.loads((args.baseline_run / "native_generation.json").read_text())
        assert baseline["status"] == "completed" and baseline["calls"] == 50
        assert baseline_config["checkpoint_sha256"] == helpers.SHA256 and baseline_config["variant"] == "flat_linear"
        assert Path(baseline_config["reference_run"]).resolve() == args.reference_run.resolve()
        assert Path(baseline_config["candidate_run"]).resolve() == args.candidate_run.resolve()
        assert helpers.file_hash(args.baseline_run / "generation_script.py") == baseline_config["runner_sha256"]
    helpers.check_gpu(args.gpu)
    assert subprocess.check_output(["git", "-C", str(root / "official"), "rev-parse", "HEAD"], text=True).strip() == helpers.COMMIT
    subprocess.run(["git", "-C", str(root / "official"), "diff", "--quiet", "HEAD", "--",
                    "models", "training/prompting_utils.py", "training/utils.py", "configs/showo_demo_512x512.yaml"], check=True)
    print("VERIFYING_EXISTING_WEIGHT_SHA256", flush=True)
    assert helpers.file_hash(root / "weights/show-o-512x512/pytorch_model.safetensors") == helpers.SHA256
    args.output = root / "results" / ("jittor-" + args.task + "-512-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    args.output.mkdir(parents=True, exist_ok=False)
    args.bridge = Path(tempfile.mkdtemp(prefix="xxt-showo-bridge-", dir="/tmp")) / "native.sock"
    source = globals().get("showo_source")
    if source is None:
        source = Path(__file__).read_text(encoding="utf-8")
    script = args.output / "generation_script.py"
    compile(source, str(script), "exec")
    with script.open("x", encoding="utf-8") as stream:
        stream.write(source)
    helpers.write_json(args.output / "run_config.json", dict(
        gpu=args.gpu, candidate_run=str(args.candidate_run), helpers_run=str(args.helpers_run),
        task=args.task, max_new_tokens=args.max_new_tokens, mmu_resolution=args.mmu_resolution,
        prompt=args.prompt, seed=args.seed,
        baseline_run=str(args.baseline_run) if args.task == "mmu" else None,
        reference_run=str(args.reference_run), bridge=str(args.bridge), variant="flat_linear",
        runner_sha256=helpers.file_hash(script), checkpoint_sha256=helpers.SHA256, commit=helpers.COMMIT,
        candidate_sha256=saved["runner_sha256"], native_sha256=saved["native_sha256"],
        atol=helpers.ATOL, rtol=helpers.RTOL,
        policy="Record full 83-check numerical status; finite tolerance failures permit explicitly labelled diagnostic generation"))
    print("RUN_DIR", args.output, flush=True)
    base = [sys.executable, "-u", str(script), "--gpu", str(args.gpu), "--candidate-run", str(args.candidate_run),
            "--helpers-run", str(args.helpers_run), "--reference-run", str(args.reference_run),
            "--task", args.task, "--max-new-tokens", str(args.max_new_tokens), "--baseline-run", str(args.baseline_run),
            "--mmu-resolution", str(args.mmu_resolution),
            "--prompt", args.prompt, "--seed", str(args.seed),
            "--output", str(args.output), "--bridge", str(args.bridge)]
    native = None
    relay = None
    log = (args.output / "native.log").open("x", encoding="utf-8")
    try:
        if args.task == "mmu":
            helpers.run_stage(base + ["--worker", "mmu_reference"], helpers.environment(args.gpu, False), args.output / "mmu_reference.log")
            helpers.check_gpu(args.gpu)
        native = subprocess.Popen(base + ["--worker", "native"], env=helpers.environment(args.gpu, True),
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace")
        relay = threading.Thread(target=relay_native, args=(native, log), daemon=True)
        relay.start()
        wait_ready(native, args.output)
        alignment_source = args.baseline_run if args.task == "mmu" else args.output
        alignment = json.loads((alignment_source / "alignment/alignment.json").read_text())
        print("FULL_FORWARD_STATUS", alignment["status"], "failed:", alignment["failed"], "/ 83", flush=True)
        print("GENERATING_" + args.task.upper() + "_WITH_JITTOR_TRANSFORMER", flush=True)
        helpers.run_stage(base + ["--worker", "sample"], helpers.environment(args.gpu, False), args.output / "sample.log")
        if native.wait(timeout=90) != 0:
            raise RuntimeError("Native worker failed; inspect native.log")
        relay.join(timeout=5)
        if args.task == "t2i":
            helpers.check_gpu(args.gpu)
            helpers.run_stage(base + ["--worker", "decode"], helpers.environment(args.gpu, False), args.output / "decode.log")
    except BaseException as error:
        if native is not None and native.poll() is None:
            native.terminate()
            try:
                native.wait(timeout=20)
            except subprocess.TimeoutExpired:
                native.kill()
                native.wait()
        helpers.write_json(args.output / "run_status.json", dict(status="failed", error=repr(error)))
        raise
    finally:
        if relay is not None:
            relay.join(timeout=5)
        log.close()
    helpers.write_json(args.output / "run_status.json", dict(
        status="generation_completed", alignment_status=alignment["status"], alignment_failed=alignment["failed"],
        task=args.task, alignment_source=str(alignment_source),
        scope="Jittor Transformer with official PyTorch sampler; image decoding for T2I only; full forward tolerance status recorded separately"))
    print("JITTOR_TRANSFORMER_" + args.task.upper() + "_DEMO_COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
