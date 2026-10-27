"""Server-only synthetic-token decoding checks for both Tiny backends."""

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def attention_mask(np, ids, task, tokens):
    batch, length = ids.shape
    allowed = np.broadcast_to(np.tri(length, dtype=bool), (batch, length, length)).copy()
    for row in range(batch):
        end = int(np.flatnonzero(ids[row] == tokens["eoi"])[0])
        if task == "mmu":
            allowed[row, :, : end + 1] = True
        else:
            start = int(np.flatnonzero(ids[row] == tokens["soi"])[0])
            allowed[row, start : end + 1, :] = True
    return np.where(allowed[:, None], np.float32(0.0), np.finfo(np.float32).min)


def dataset(np, reference, tokens):
    with np.load(reference / "initial_fixture.npz", allow_pickle=False) as fixture:
        mmu = fixture["mmu.input_ids"].copy()
        mmu_labels = fixture["mmu.labels"].copy()
        t2i = fixture["t2i.input_ids"].copy()
        t2i_labels = fixture["t2i.labels"].copy()
    end = int(np.flatnonzero(mmu[0] == tokens["eoi"])[0])
    start = int(np.flatnonzero(t2i[0] == tokens["soi"])[0]) + 1
    image_end = int(np.flatnonzero(t2i[0] == tokens["eoi"])[0])
    image = mmu[:, 2:end]
    if image.shape[1] != image_end - start:
        raise ValueError("Image spans disagree between tasks")
    observed = t2i[:, start:image_end]
    if not np.array_equal(observed[observed != tokens["mask"]], image[observed != tokens["mask"]]):
        raise ValueError("MMU and T2I fixture rows do not correspond")
    selected = t2i_labels[:, start:image_end] != -100
    if not np.array_equal(t2i_labels[:, start:image_end][selected], image[selected]):
        raise ValueError("T2I target mismatch")
    return dict(mmu=mmu, text=mmu_labels[:, end + 1 :], prefix_length=end + 1, t2i=t2i, image=image, image_start=start, image_end=image_end)


def decode(np, predict, data, tokens):
    generated = []
    for row in range(len(data["mmu"])):
        ids = data["mmu"][row : row + 1, : data["prefix_length"]].copy()
        sequence = []
        for step in range(12):
            logits = predict(ids, "mmu", f"mmu.{row}.{step}")
            token = int(logits[0, -1].argmax())
            sequence.append(token)
            ids = np.concatenate((ids, np.array([[token]], dtype=np.int64)), axis=1)
            if token == tokens["eos"]:
                break
        generated.append(sequence)
    targets = data["text"].tolist()
    correct = sum(sum(i < len(actual) and actual[i] == token for i, token in enumerate(expected)) for actual, expected in zip(generated, targets))
    denominator = sum(max(len(actual), len(expected)) for actual, expected in zip(generated, targets))
    result = dict(
        scope="Decoding the same four synthetic training samples; no held-out or image-quality claim",
        mmu=dict(
            generated=generated,
            targets=targets,
            exact_matches=sum(actual == expected for actual, expected in zip(generated, targets)),
            samples=len(generated),
            token_accuracy=correct / denominator,
            eos_terminated=[bool(sequence and sequence[-1] == tokens["eos"]) for sequence in generated],
        ),
        t2i={},
    )
    start, end = data["image_start"], data["image_end"]
    for case in ("masked_completion", "all_masks"):
        ids = data["t2i"].copy()
        if case == "all_masks":
            ids[:, start:end] = tokens["mask"]
        initial_unknown = ids[:, start:end] == tokens["mask"]
        initial_count = initial_unknown.sum(axis=1)
        given = ids[:, start:end].copy()
        progress = []
        for step in range(8):
            image = ids[:, start:end].copy()
            unknown = image == tokens["mask"]
            if not unknown.any():
                break
            scores = predict(ids, "t2i", f"t2i.{case}.{step}")[:, start:end, 128:192]
            logits = scores.astype(np.float64)
            exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probabilities = exp / exp.sum(axis=-1, keepdims=True)
            predicted = scores.argmax(axis=-1)
            confidence = np.take_along_axis(probabilities, predicted[..., None], axis=-1)[..., 0]
            candidate = np.where(unknown, predicted + 128, image)
            next_unknown = np.zeros_like(unknown)
            for row in range(len(image)):
                indices = np.flatnonzero(unknown[row])
                scheduled = math.floor(int(initial_count[row]) * math.cos(math.pi * (step + 1) / 16))
                remaining = min(scheduled, max(0, len(indices) - 1)) if step < 7 else 0
                order = np.argsort(confidence[row, indices], kind="stable")
                next_unknown[row, indices[order[:remaining]]] = True
            candidate[next_unknown] = tokens["mask"]
            if not np.array_equal(candidate[~unknown], image[~unknown]):
                raise ValueError("A previously committed image token was modified")
            ids[:, start:end] = candidate
            progress.append(dict(step=step + 1, remaining_masks=next_unknown.sum(axis=1).tolist(), image_tokens=candidate.tolist()))
        image = ids[:, start:end]
        if (image == tokens["mask"]).any() or not ((image >= 128) & (image < 192)).all():
            raise ValueError("Image decoding left masks or invalid codebook tokens")
        if not np.array_equal(image[~initial_unknown], given[~initial_unknown]):
            raise ValueError("Completion changed a supplied image token")
        result["t2i"][case] = dict(
            tokens=image.tolist(),
            targets=data["image"].tolist(),
            full_token_accuracy=float((image == data["image"]).mean()),
            generated_position_accuracy=float((image == data["image"])[initial_unknown].mean()),
            exact_matches=int(np.all(image == data["image"], axis=1).sum()),
            remaining_masks=int((image == tokens["mask"]).sum()),
            progress=progress,
        )
    return result


def worker(args):
    import numpy as np

    from train_tiny_reference import COMMIT, load_file

    config = json.loads((args.reference / "config.json").read_text())
    metadata = json.loads((args.reference / "metrics.json").read_text())
    if metadata["commit"] != COMMIT or metadata["status"] != "passed":
        raise ValueError("Expected successful pinned reference")
    tokens = metadata["tokens"]
    data = dataset(np, args.reference, tokens)
    traces = {}
    if args.backend == "torch":
        import hashlib
        import torch
        from transformers import PhiConfig

        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        official = Path.home() / "Show-o-Jittor/official"
        for name, expected in metadata["sources"].items():
            if hashlib.sha256((official / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Official source changed: {name}")
        phi = load_file("decode_reference_phi", official / "models/phi.py")
        prompting = load_file("decode_reference_prompting", official / "training/prompting_utils.py")
        cfg = PhiConfig.from_dict(config)
        cfg._attn_implementation = "eager"
        model = phi.PhiForCausalLM(cfg).cuda().eval()
        model.load_state_dict(torch.load(args.reference / "model.pt", map_location="cpu", weights_only=True))
        np.savez(args.output / "shared_trained_weights.npz", **{name: value.detach().cpu().numpy() for name, value in model.state_dict().items()})

        def forward(ids, task):
            ids_tensor = torch.as_tensor(ids, dtype=torch.long, device="cuda")
            if task == "mmu":
                allowed = prompting.create_attention_mask_for_mmu(ids_tensor, eoi_id=tokens["eoi"], return_inverse_mask=False)
            else:
                allowed = prompting.create_attention_mask_predict_next(ids_tensor, pad_id=tokens["pad"], soi_id=tokens["soi"], eoi_id=tokens["eoi"], return_inverse_mask=False)
            additive = attention_mask(np, ids, task, tokens)
            if not np.array_equal(allowed.cpu().numpy().astype(bool), additive == 0):
                raise ValueError("Decoding mask differs from official implementation")
            with torch.no_grad():
                return model(input_ids=ids_tensor, attention_mask=torch.as_tensor(additive, device="cuda"), use_cache=False).logits.cpu().numpy()

    else:
        import jittor as jt
        from align_tiny_jittor import model_class, strict_load

        jt.flags.use_cuda = 1
        model = model_class(jt, np)(config)
        weights = args.output / "shared_trained_weights.npz" if args.backend == "shared-jittor" else args.jittor_run / "jittor_weights.npz"
        with np.load(weights, allow_pickle=False) as archive:
            strict_load(model, {key: archive[key] for key in archive.files}, jt, np)
        model.eval()

        def forward(ids, task):
            with jt.no_grad():
                logits, _ = model(jt.array(ids.astype(np.int32)), jt.array(attention_mask(np, ids, task, tokens)))
                return logits.numpy()

    def predict(ids, task, key):
        logits = forward(ids, task)
        if not np.isfinite(logits).all():
            raise ValueError(f"Non-finite inference output: {key}")
        if args.backend != "trained-jittor":
            traces[key] = logits
        return logits

    result = decode(np, predict, data, tokens)
    result["backend"] = args.backend
    result["decoding"] = dict(mmu="greedy full-vocabulary next token", t2i="greedy image-codebook prediction with cosine confidence-based unmasking; no CFG or sampling noise")
    (args.output / f"{args.backend}_tokens.json").write_text(json.dumps(result, indent=2) + "\n")
    if traces:
        np.savez(args.output / f"{args.backend}_logits.npz", **traces)
    summary = dict(
        backend=args.backend,
        mmu_exact_matches=result["mmu"]["exact_matches"],
        mmu_samples=result["mmu"]["samples"],
        mmu_token_accuracy=result["mmu"]["token_accuracy"],
        t2i_completion_accuracy=result["t2i"]["masked_completion"]["generated_position_accuracy"],
        t2i_all_masks_accuracy=result["t2i"]["all_masks"]["full_token_accuracy"],
        remaining_masks=result["t2i"]["all_masks"]["remaining_masks"],
    )
    print(json.dumps(summary, indent=2), flush=True)


def run_backend(args, backend):
    project = Path.home() / "Show-o-Jittor"
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), NVIDIA_TF32_OVERRIDE="0", HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")
    if backend != "torch":
        view = project / "deps/cuda-jittor"
        env.update(CUDA_HOME=str(view), nvcc_path=str(view / "bin/nvcc"), JITTOR_HOME="/tmp/xxt-showo-jittor", DISABLE_MULTIPROCESSING="1")
        env["PATH"] = str(view / "bin") + ":" + env.get("PATH", "")
        old_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = str(view / "lib64") + (":" + old_ld if old_ld else "")
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--backend", backend, "--reference", str(args.reference), "--jittor-run", str(args.jittor_run), "--output", str(args.output), "--gpu", str(args.gpu)]
    with (args.output / f"{backend}.log").open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
            stream.flush()
        code = process.wait()
        stream.write(f"\nExit code: {code}\n")
    if code:
        raise RuntimeError(f"Backend {backend} failed ({code}); see {args.output / (backend + '.log')}")


def compare_shared(np, output):
    expected = json.loads((output / "torch_tokens.json").read_text())
    actual = json.loads((output / "shared-jittor_tokens.json").read_text())
    tokens_equal = expected["mmu"]["generated"] == actual["mmu"]["generated"]
    for case in expected["t2i"]:
        tokens_equal = tokens_equal and expected["t2i"][case]["tokens"] == actual["t2i"][case]["tokens"]
        tokens_equal = tokens_equal and expected["t2i"][case]["progress"] == actual["t2i"][case]["progress"]
    rows = []
    with np.load(output / "torch_logits.npz", allow_pickle=False) as torch_data, np.load(output / "shared-jittor_logits.npz", allow_pickle=False) as jt_data:
        keys_equal = set(torch_data.files) == set(jt_data.files)
        for name in sorted(set(torch_data.files) & set(jt_data.files)):
            reference, result = torch_data[name], jt_data[name]
            same_shape = reference.shape == result.shape
            error = float(np.abs(reference - result).max()) if same_shape else None
            passed = same_shape and bool(np.allclose(reference, result, atol=1e-4, rtol=0))
            rows.append(dict(name=name, max_abs=error, passed=passed))
    report = dict(
        status="passed" if rows and tokens_equal and keys_equal and all(row["passed"] for row in rows) else "failed",
        scope="Same trained PyTorch weights in both backends; not independently trained weight equality",
        token_trajectories_equal=tokens_equal,
        trace_keys_equal=keys_equal,
        checked=len(rows),
        atol=1e-4,
        rtol=0,
        worst_max_abs=max((row["max_abs"] for row in rows if row["max_abs"] is not None), default=None),
        rows=rows,
    )
    (output / "decode_alignment.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2), flush=True)
    if report["status"] != "passed":
        raise RuntimeError("Shared-weight decode alignment failed; inspect decode_alignment.json")
    print("TINY_SHARED_WEIGHT_DECODE_ALIGN_OK", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--jittor-run", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--backend", choices=["torch", "shared-jittor", "trained-jittor"], help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if Path(sys.prefix).resolve() != (Path.home() / "envs/showo-jittor").resolve():
        raise RuntimeError("Activate the showo-jittor environment")
    args.reference = args.reference.resolve(strict=True)
    args.jittor_run = args.jittor_run.resolve(strict=True)
    if args.backend:
        worker(args)
        return
    import numpy as np

    alignment = json.loads((args.jittor_run / "alignment.json").read_text())
    training = json.loads((args.jittor_run / "jittor_training.json").read_text())
    if alignment["status"] != "passed" or Path(alignment["reference"]).resolve() != args.reference:
        raise ValueError("Jittor run does not match the successful reference")
    if not all(training["final_loss"][task] < training["initial_loss"][task] for task in ("mmu", "t2i")):
        raise ValueError("Expected a successfully trained Jittor checkpoint")
    args.output = Path.home() / "Show-o-Jittor/results" / ("tiny-inference-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    args.output.mkdir(parents=True)
    print("Results:", args.output, flush=True)
    run_backend(args, "torch")
    run_backend(args, "shared-jittor")
    compare_shared(np, args.output)
    run_backend(args, "trained-jittor")
    print("TINY_TOKEN_INFERENCE_OK", flush=True)
    print("Scope: synthetic training samples, token outputs only; no real-image generation.", flush=True)
    print("Results:", args.output, flush=True)


if __name__ == "__main__":
    main()
