"""Full pretrained Show-o Transformer loss/all-parameter gradient validation.

One real Flickr training image/caption, 512px / 1024 image codes. Frozen
official MagViT prepares the shared data oracle; separate PyTorch and Jittor
workers backpropagate through all 24 Transformer blocks. No optimizer steps.
"""

import argparse
import ast
from datetime import datetime
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import types

import numpy as np

from native_pipeline import COMMIT, MAGVIT_SHA, PARAMETERS, SHOWO_SHA, file_hash, load_module, native_guard, read_json, write_json


GRAD_ATOL, GRAD_RTOL = 1e-5, 1e-3
LOSS_ATOL, LOSS_RTOL = 1e-4, 1e-4
MANIFEST_SHA = "dce49014745a0a6c2ba9e5c7a75058a2aafa7a72c616a1d94cd124e1d664b47e"


def compare_arrays(actual, reference, atol=GRAD_ATOL, rtol=GRAD_RTOL):
    if actual.shape != reference.shape:
        raise RuntimeError("Gradient shape differs")
    a, b = actual.reshape(-1), reference.reshape(-1)
    outside, maximum, absolute_sum, error_sq, reference_sq = 0, 0., 0., 0., 0.
    finite = True
    for start in range(0, a.size, 1024 * 1024):
        x, y = a[start:start + 1024 * 1024].astype(np.float64), b[start:start + 1024 * 1024].astype(np.float64)
        valid = np.isfinite(x) & np.isfinite(y)
        finite = finite and bool(valid.all())
        delta = np.abs(x - y)
        outside += int(np.count_nonzero(~valid | (delta > atol + rtol * np.abs(y))))
        if valid.all():
            maximum = max(maximum, float(delta.max()))
            absolute_sum += float(delta.sum())
            error_sq += float(np.square(delta).sum())
            reference_sq += float(np.square(y).sum())
    return dict(shape=list(actual.shape), values=int(a.size), finite=finite,
        max_abs=maximum if finite else None, mean_abs=absolute_sum / a.size if finite else None,
        relative_l2=(error_sq / max(reference_sq, 1e-300)) ** 0.5 if finite else None,
        error_squared_sum=error_sq if finite else None,
        reference_squared_sum=reference_sq if finite else None,
        outside_tolerance=outside, atol=atol, rtol=rtol, passed=finite and outside == 0)


def check_gpu(gpu):
    used = int(subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used",
        "--format=csv,noheader,nounits"], text=True, timeout=10).strip())
    if used > 1024:
        raise RuntimeError("GPU occupied: %d MiB; no existing jobs terminated" % used)


def validate_fixture(ids, labels, attention, task):
    if ids.shape != labels.shape or ids.ndim != 2 or ids.shape[0] != 1:
        raise RuntimeError("Expected one identically shaped input/label sequence")
    length = ids.shape[1]
    if length > 2048 or attention.shape != (1, 1, length, length):
        raise RuntimeError("Unexpected attention mask shape or sequence length")
    if attention.dtype != np.float32 or not np.isfinite(attention).all():
        raise RuntimeError("Attention mask must be finite FP32")
    if ids.min() < 0 or ids.max() >= 58498:
        raise RuntimeError("Input outside the official vocabulary")
    supervised = labels[labels != -100]
    if supervised.size == 0 or supervised.min() < 0 or supervised.max() >= 58498:
        raise RuntimeError("Invalid or empty supervised targets")
    if task == "t2i" and (ids.shape != (1, 1155) or int(np.count_nonzero(ids == 58497)) != 512):
        raise RuntimeError("Expected 512 masked codes in the 512px T2I fixture")


def prepare(args):
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from torchvision import transforms

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    manifest = read_json(args.data / "manifest.json")
    if file_hash(args.data / "manifest.json") != MANIFEST_SHA:
        raise RuntimeError("Flickr manifest differs from the verified 128/32 subset")
    row = next(x for x in manifest["samples"] if x["split"] == "train")
    image_path = (args.data / row["image"]).resolve()
    image_path.relative_to(args.data.resolve())
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
    import hashlib
    if hashlib.sha256(str(rgb.size).encode("ascii") + rgb.tobytes()).hexdigest() != row["rgb_sha256"]:
        raise RuntimeError("Flickr image contents differ")
    source = (args.root / "official/training/utils.py").read_text(encoding="utf-8")
    functions = [x for x in ast.parse(source).body if isinstance(x, ast.FunctionDef) and x.name == "image_transform"]
    if len(functions) != 1:
        raise RuntimeError("Official image transform not unique")
    namespace = {"transforms": transforms}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "pinned_image_transform", "exec"), namespace)
    pixels = namespace["image_transform"](rgb, resolution=512).unsqueeze(0)
    package = types.ModuleType("gradient_fixture_magvit")
    package.__path__ = [str(args.root / "official/models")]
    sys.modules[package.__name__] = package
    cls = importlib.import_module(package.__name__ + ".modeling_magvitv2").MAGVITv2
    model = cls()
    model.load_state_dict(load_file(str(args.root / "weights/magvitv2/pytorch_model.safetensors")), strict=True)
    model.eval().requires_grad_(False).cuda()
    with torch.inference_mode():
        codes = model.get_code(pixels.cuda()).cpu().long()
    if tuple(codes.shape) != (1, 1024) or codes.min() < 0 or codes.max() >= 8192:
        raise RuntimeError("Invalid 512px codes")
    prompting = load_module("gradient_fixture_prompting", args.root / "official/training/prompting_utils.py")
    tokenizer = AutoTokenizer.from_pretrained(str(args.root / "weights/phi-1_5-original"), local_files_only=True)
    t2i_prompt = prompting.UniversalPrompting(tokenizer, max_text_len=128, cond_dropout_prob=0.)
    if len(tokenizer) != 50305:
        raise RuntimeError("Unexpected official vocabulary")
    caption = row["captions"][0]
    image_ids = codes + len(tokenizer)
    masked = image_ids.clone()
    selected = torch.zeros_like(masked, dtype=torch.bool)
    selected[:, np.random.default_rng(args.seed).choice(1024, 512, replace=False)] = True
    masked[selected] = 58497
    targets = torch.where(selected, image_ids, -100)
    ids, _, labels = t2i_prompt(([caption], masked, targets), "t2i")
    attention = prompting.create_attention_mask_predict_next(ids, pad_id=tokenizer.pad_token_id,
        soi_id=int(t2i_prompt.sptids_dict["<|soi|>"].item()), eoi_id=int(t2i_prompt.sptids_dict["<|eoi|>"].item()),
        rm_pad_in_image=True).float()
    validate_fixture(ids.numpy(), labels.numpy(), attention.numpy(), "t2i")
    np.savez(args.output / "t2i_input.npz", input_ids=ids.numpy(), labels=labels.numpy(), attention_mask=attention.numpy())
    mmu_prompt = prompting.UniversalPrompting(tokenizer, max_text_len=64, cond_dropout_prob=0.)
    ids, _, labels = mmu_prompt((image_ids, [caption]), "mmu")
    attention = prompting.create_attention_mask_for_mmu(ids, eoi_id=int(mmu_prompt.sptids_dict["<|eoi|>"].item())).float()
    validate_fixture(ids.numpy(), labels.numpy(), attention.numpy(), "mmu")
    np.savez(args.output / "mmu_input.npz", input_ids=ids.numpy(), labels=labels.numpy(), attention_mask=attention.numpy())
    write_json(args.output / "fixture.json", dict(manifest_sha256=MANIFEST_SHA, image=row["image"],
        image_rgb_sha256=row["rgb_sha256"], split="train", caption_index=0, caption=caption,
        image_resolution=512, image_codes=1024, mask_count=512,
        t2i_loss="Official UniversalPrompting t2i labels: non-padding text/special tokens plus masked image targets; unshifted full-vocabulary CE",
        mmu_loss="Official UniversalPrompting mmu labels, shifted full-vocabulary CE; text max length 64",
        oracle="Frozen official PyTorch MagViT prepares shared image codes; Transformer gradient workers use identical stored inputs",
        not_validation_metric=True,
        fixture_sha256={task: file_hash(args.output / (task + "_input.npz")) for task in ("mmu", "t2i")}))
    print("FULL_GRADIENT_FIXTURE_READY", flush=True)


def torch_worker(args):
    import torch
    from safetensors.torch import load_file
    from transformers import PhiConfig

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    phi = load_module("full_gradient_official_phi", args.root / "official/models/phi.py")
    config = PhiConfig.from_dict(read_json(args.reference_run / "phi_config.json"))
    config._attn_implementation = "eager"
    with torch.device("cuda"):
        model = phi.PhiForCausalLM(config)
    weights = load_file(str(args.root / "weights/show-o-512x512/pytorch_model.safetensors"), device="cpu")
    model.load_state_dict({k.removeprefix("showo."): v for k, v in weights.items()}, strict=True)
    del weights
    model.train().requires_grad_(True)
    params = list(model.named_parameters())
    if sum(p.numel() for _, p in params) != PARAMETERS or len(params) != 437:
        raise RuntimeError("Unexpected full parameter set")
    if any(getattr(config, k, 0.) for k in ("attention_dropout", "resid_pdrop", "embd_pdrop")):
        raise RuntimeError("Gradient alignment requires dropout=0")
    with np.load(args.output / (args.task + "_input.npz"), allow_pickle=False) as data:
        validate_fixture(data["input_ids"], data["labels"], data["attention_mask"], args.task)
        ids = torch.from_numpy(data["input_ids"].astype(np.int64)).cuda()
        labels = torch.from_numpy(data["labels"].astype(np.int64)).cuda()
        mask = torch.from_numpy(data["attention_mask"]).cuda()
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    if args.task == "mmu":
        scores, targets = logits[:, :-1], labels[:, 1:]
    else:
        scores, targets = logits, labels
    loss = torch.nn.functional.cross_entropy(scores.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
    print("TORCH_FULL_BACKWARD_START", args.task, "loss", float(loss.detach().cpu()), flush=True)
    loss.backward()
    torch.cuda.synchronize()
    output = args.output / ("torch_" + args.task)
    output.mkdir(exist_ok=False)
    names = []
    for index, (name, parameter) in enumerate(params, 1):
        if parameter.grad is None:
            raise RuntimeError("Missing Torch gradient: " + name)
        value = parameter.grad.detach().cpu().numpy()
        if value.dtype != np.float32 or not np.isfinite(value).all():
            raise RuntimeError("Invalid Torch gradient: " + name)
        np.save(output / (name + ".npy"), value, allow_pickle=False)
        names.append(name)
        if index == 1 or index % 48 == 0 or index == len(params):
            print("TORCH_GRADIENT_SAVED", args.task, index, "/", len(params), name, flush=True)
    write_json(output / "report.json", dict(task=args.task, loss=float(loss.detach().cpu()),
        parameter_names=names, parameters=PARAMETERS, state_entries=len(params),
        supervised_tokens=int((targets != -100).sum().item()),
        input_shape=list(ids.shape), seconds=time.monotonic() - started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(), gradients_finite=True))
    print("TORCH_FULL_GRADIENT_COMPLETE", args.task, flush=True)


def native_worker(args):
    import jittor as jt
    from native_pipeline import load_transformer

    jt.flags.use_cuda = 1
    jt.flags.amp_level = 0
    model = load_transformer(args, jt)
    # Jittor eval() and the inference loader stop gradients. Restore BOTH.
    model.train()
    state = model.state_dict()
    for parameter in state.values():
        parameter.start_grad()
    params = sorted(state.items())
    expected = read_json(args.output / ("torch_" + args.task) / "report.json")
    if set(state) != set(expected["parameter_names"]) or sum(int(np.prod(p.shape)) for _, p in params) != PARAMETERS:
        raise RuntimeError("Native/Torch full trainable parameter sets differ")
    config = read_json(args.reference_run / "phi_config.json")
    if any(config.get(k, 0.) for k in ("attention_dropout", "resid_pdrop", "embd_pdrop")):
        raise RuntimeError("Gradient alignment requires dropout=0")
    native_guard("full gradients imports")
    with np.load(args.output / (args.task + "_input.npz"), allow_pickle=False) as data:
        validate_fixture(data["input_ids"], data["labels"], data["attention_mask"], args.task)
        ids = jt.array(data["input_ids"].astype(np.int32)).stop_grad()
        labels = jt.array(data["labels"].astype(np.int32)).stop_grad()
        mask = jt.array(data["attention_mask"]).stop_grad()
    started = time.monotonic()
    logits, trace = model(ids, mask)
    trace.clear()
    if args.task == "mmu":
        scores, targets = logits[:, :-1], labels[:, 1:]
    else:
        scores, targets = logits, labels
    loss = jt.nn.cross_entropy_loss(scores.reshape((-1, logits.shape[-1])), targets.reshape((-1,)), ignore_index=-100)
    actual_loss = float(loss.numpy().item())
    print("JITTOR_FULL_BACKWARD_START", args.task, "loss", actual_loss, flush=True)
    gradients = jt.grad(loss, [p for _, p in params])
    if len(gradients) != len(params):
        raise RuntimeError("Native gradient count differs from full parameter count")
    rows = []
    with (args.output / ("gradients_" + args.task + ".jsonl")).open("x", encoding="utf-8") as stream:
        for index, ((name, parameter), gradient) in enumerate(zip(params, gradients), 1):
            actual = gradient.numpy()
            reference = np.load(args.output / ("torch_" + args.task) / (name + ".npy"), mmap_mode="r", allow_pickle=False)
            if actual.dtype != np.float32:
                raise RuntimeError("Unexpected native gradient dtype: " + name)
            row = dict(name=name, **compare_arrays(actual, reference))
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            del actual, reference
            if index == 1 or index % 48 == 0 or index == len(params):
                print("JITTOR_GRADIENT_CHECK", args.task, index, "/", len(params), name, "passed", row["passed"], flush=True)
    finite = all(x["finite"] for x in rows)
    loss_comparison = compare_arrays(np.array(actual_loss), np.array(expected["loss"]), LOSS_ATOL, LOSS_RTOL)
    result = dict(task=args.task, parameters=PARAMETERS, gradient_tensors_checked=len(rows),
        gradient_values_checked=sum(x["values"] for x in rows),
        gradient_tensors_passed=sum(x["passed"] for x in rows),
        outside_tolerance=sum(x["outside_tolerance"] for x in rows), gradients_finite=finite,
        global_relative_l2=(sum(x["error_squared_sum"] for x in rows) / max(sum(x["reference_squared_sum"] for x in rows), 1e-300)) ** .5 if finite else None,
        actual_loss=actual_loss, reference_loss=expected["loss"], loss_comparison=loss_comparison,
        passed=loss_comparison["passed"] and all(x["passed"] for x in rows),
        tolerances=dict(gradient_atol=GRAD_ATOL, gradient_rtol=GRAD_RTOL, loss_atol=LOSS_ATOL, loss_rtol=LOSS_RTOL),
        seconds=time.monotonic() - started, native_process=native_guard("full gradients complete"))
    write_json(args.output / ("native_" + args.task + ".json"), result)
    print("FULL_GRADIENT_TASK_COMPLETE", json.dumps(result, allow_nan=False), flush=True)


def main(argv=None):
    root = Path.home() / "Show-o-Jittor"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=Path, default=root / "data/flickr8k-128-32-20261026-203910-164291")
    parser.add_argument("--reference-run", type=Path, default=root / "results/official512-jittor-align-20261027-004837-924843")
    parser.add_argument("--candidate-run", type=Path, default=root / "results/official512-operator-candidates-20261027-022054-436199")
    parser.add_argument("--worker", choices=("prepare", "torch", "native"), help=argparse.SUPPRESS)
    parser.add_argument("--task", choices=("mmu", "t2i"), help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.root = root
    if args.worker:
        if args.output is None or (args.worker != "prepare" and args.task is None):
            parser.error("Worker needs output/task")
        {"prepare": prepare, "torch": torch_worker, "native": native_worker}[args.worker](args)
        return 0
    check_gpu(args.gpu)
    if shutil.disk_usage(root).free < 18 * 1024 ** 3:
        raise RuntimeError("Need 18 GiB free in the project filesystem for two full Torch gradient fixtures; not system /tmp")
    commit = subprocess.check_output(["git", "-C", str(root / "official"), "rev-parse", "HEAD"], text=True).strip()
    if commit != COMMIT:
        raise RuntimeError("Official commit differs")
    subprocess.run(["git", "-C", str(root / "official"), "diff", "--quiet", "HEAD", "--", "models", "training/prompting_utils.py", "training/utils.py"], check=True)
    baseline = read_json(args.reference_run / "run_config.json")
    candidate = read_json(args.candidate_run / "run_config.json")
    if (file_hash(args.reference_run / "native_phi.py") != baseline["native_source_sha256"]
        or baseline["native_source_sha256"] != candidate["native_sha256"]
        or file_hash(args.candidate_run / "operator_script.py") != candidate["runner_sha256"]):
        raise RuntimeError("Validated native/candidate sources differ")
    if (baseline["commit"] != COMMIT or baseline["checkpoint_sha256"] != SHOWO_SHA
        or candidate["commit"] != COMMIT or candidate["checkpoint_sha256"] != SHOWO_SHA
        or read_json(args.reference_run / "reference_status.json")["status"] != "passed"):
        raise RuntimeError("Prior full-weight reference provenance differs")
    operators = read_json(args.candidate_run / "flat_linear/operators.json")
    if operators["checked"] != 24 or operators["outside_tolerance"] != 0:
        raise RuntimeError("Prior flat_linear operator acceptance differs")
    if file_hash(args.data / "manifest.json") != MANIFEST_SHA:
        raise RuntimeError("Expected verified Flickr subset manifest")
    for name, expected in (("show-o-512x512", SHOWO_SHA), ("magvitv2", MAGVIT_SHA)):
        print("VERIFYING_FULL_GRADIENT_WEIGHT", name, flush=True)
        if file_hash(root / "weights" / name / "pytorch_model.safetensors") != expected:
            raise RuntimeError("Weight checksum differs: " + name)
    args.output = root / "results" / ("full-gradient-512-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    args.output.mkdir(parents=True, exist_ok=False)
    for name in (Path(__file__).name, "native_pipeline.py"):
        shutil.copyfile(Path(__file__).parent / name, args.output / name)
    write_json(args.output / "run_config.json", dict(
        official_commit=COMMIT, checkpoint_sha256=SHOWO_SHA, parameters=PARAMETERS,
        data=str(args.data), reference_run=str(args.reference_run), candidate_run=str(args.candidate_run),
        seed=args.seed, gpu=args.gpu, precision="float32", trainable_scope="all 437 Transformer parameter tensors, all 24 blocks; frozen MagViT data oracle",
        tolerances=dict(gradient_atol=GRAD_ATOL, gradient_rtol=GRAD_RTOL, loss_atol=LOSS_ATOL, loss_rtol=LOSS_RTOL),
        native_source_sha256=baseline["native_source_sha256"], script_sha256=file_hash(Path(__file__)),
        optimization_steps=0, original_83_checks_rerun=False))
    print("RUN_DIR", args.output, flush=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), NVIDIA_TF32_OVERRIDE="0", PYTHONUNBUFFERED="1",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    common = [sys.executable, "-u", str(Path(__file__).resolve()), "--gpu", str(args.gpu),
        "--seed", str(args.seed), "--data", str(args.data), "--reference-run", str(args.reference_run),
        "--candidate-run", str(args.candidate_run), "--output", str(args.output)]
    stages = [("prepare", None)] + [(worker, task) for task in ("mmu", "t2i") for worker in ("torch", "native")]
    for worker, task in stages:
        label = worker + ("_" + task if task else "")
        check_gpu(args.gpu)
        started = time.monotonic()
        print("FULL_GRADIENT_STAGE_START", label, flush=True)
        command = common + ["--worker", worker] + (["--task", task] if task else [])
        with (args.output / (label + ".log")).open("x", encoding="utf-8") as log:
            with subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
        if code:
            write_json(args.output / "run_status.json", dict(status="failed", stage=label, returncode=code))
            return code
        print("FULL_GRADIENT_STAGE_DONE", label, "seconds", round(time.monotonic() - started, 2), flush=True)
    tasks = [read_json(args.output / ("native_" + task + ".json")) for task in ("mmu", "t2i")]
    passed = all(row["passed"] for row in tasks)
    write_json(args.output / "full_gradient_report.json", dict(status="completed", passed=passed,
        tasks=tasks, parameters=PARAMETERS, optimization_steps=0,
        limits="Two fixed real-image training fixtures, all Transformer parameter gradients. No claim of training convergence, full MagViT gradients or validation metrics; original 78/83 forward result unchanged."))
    write_json(args.output / "run_status.json", dict(status="completed", gradient_acceptance_passed=passed))
    print("FULL_GRADIENT_VALIDATION_COMPLETE", args.output, "passed", passed, flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
