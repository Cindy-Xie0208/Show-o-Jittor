"""Server-only real-token reference alignment and small Jittor training run."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


COMMIT = "45a5a2de01d1ebd10cd5864d29310a76476cdf23"
DEFAULT_DATA = "flickr8k-tokens-20261026-204336-019378"
MANIFEST_SHA = "dce49014745a0a6c2ba9e5c7a75058a2aafa7a72c616a1d94cd124e1d664b47e"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def load_data(reference):
    import numpy as np

    with np.load(reference / "training_data.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    data["special"] = json.loads((reference / "dataset.json").read_text())["special"]
    return data


def make_batch(data, indices, captions, task, rng, rate=None):
    import numpy as np

    special = data["special"]
    sequences, targets, regions = [], [], []
    for index, caption in zip(indices, captions):
        length = int(data["text_lengths"][index, caption])
        text = data["text_ids"][index, caption, :length].tolist()
        picture = data["image_tokens"][index].astype(np.int32) + special["image_offset"]
        if task == "mmu":
            prefix = [special["mmu"], special["soi"]] + picture.tolist() + [special["eoi"]]
            sequence = prefix + text + [special["eos"]]
            labels = [-100] * len(prefix) + text + [special["eos"]]
            region = (len(prefix), 0)
        else:
            if rate is None:
                fraction = math.cos(float(rng.random()) * math.pi / 2)
                if rng.random() < 0.1:
                    fraction = 1.0
            else:
                fraction = rate
            count = max(1, min(256, int(math.ceil(256 * fraction))))
            selected = np.zeros(256, dtype=bool)
            selected[rng.choice(256, count, replace=False)] = True
            corrupted = np.where(selected, special["mask"], picture).tolist()
            prefix = [special["t2i"]] + text + [special["eos"], special["soi"]]
            sequence = prefix + corrupted + [special["eoi"]]
            labels = [-100] * len(prefix) + np.where(selected, picture, -100).tolist() + [-100]
            region = (len(prefix) - 1, len(sequence) - 1)
        sequences.append(sequence)
        targets.append(labels)
        regions.append(region)
    size = int(data.get("sequence_length", max(map(len, sequences))))
    if max(map(len, sequences)) > size:
        raise ValueError("Sequence exceeds the configured fixed length")
    ids = np.full((len(sequences), size), special["pad"], dtype=np.int32)
    labels = np.full_like(ids, -100)
    allowed = np.broadcast_to(np.tri(size, dtype=bool), (len(sequences), size, size)).copy()
    for row, (sequence, target, region) in enumerate(zip(sequences, targets, regions)):
        ids[row, :len(sequence)] = sequence
        labels[row, :len(target)] = target
        if task == "mmu":
            allowed[row, :, :region[0]] = True
        else:
            begin, end = region
            allowed[row, begin:end + 1, :len(sequence)] = True
            allowed[row, begin:end + 1, len(sequence):] = False
    mask = np.where(allowed[:, None], np.float32(0), np.finfo(np.float32).min)
    return dict(input_ids=ids, labels=labels, attention_mask=mask)


def verify_masks(torch, prompting, data):
    import numpy as np

    first = int(data["train_indices"][0])
    original_length = data["text_lengths"][first, 0]
    other = next((int(i), c) for i in data["train_indices"] for c in range(5)
                 if data["text_lengths"][i, c] != original_length)
    checked = 0
    for indices, captions in (([first], [0]), ([first, other[0]], [0, other[1]])):
        for task in ("mmu", "t2i"):
            for rate in (0.01, 0.5, 1.0):
                batch = make_batch(data, indices, captions, task, np.random.default_rng(123), rate)
                ids = torch.tensor(batch["input_ids"], dtype=torch.long)
                if task == "mmu":
                    actual = prompting.create_attention_mask_for_mmu(
                        ids, eoi_id=data["special"]["eoi"], return_inverse_mask=False)
                else:
                    actual = prompting.create_attention_mask_predict_next(
                        ids, pad_id=data["special"]["pad"], soi_id=data["special"]["soi"],
                        eoi_id=data["special"]["eoi"], rm_pad_in_image=True, return_inverse_mask=False)
                    selected = batch["labels"] != -100
                    if not np.all(batch["input_ids"][selected] == data["special"]["mask"]):
                        raise ValueError("T2I labels must select only masked image positions")
                if not np.array_equal(actual.bool().numpy(), batch["attention_mask"] == 0):
                    raise ValueError("Attention mask differs from the pinned official implementation")
                checked += 1
    print("REAL_DATA_OFFICIAL_MASK_ORACLE_OK", checked, flush=True)


def prepare_reference(args):
    import numpy as np
    import torch
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoTokenizer, PhiConfig
    import train_tiny_reference as helper

    raw = (args.tokens / "source_manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA:
        raise ValueError("Source manifest differs from the verified Flickr subset")
    source = json.loads(raw)
    with np.load(args.tokens / "tokens.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    train_ids, val_ids = data["train_indices"], data["validation_indices"]
    if data["image_tokens"].shape != (160, 256) or len(source["samples"]) != 160:
        raise ValueError("Unexpected real image dataset shape")
    if len(train_ids) != 128 or len(val_ids) != 32 or set(train_ids) & set(val_ids):
        raise ValueError("Unexpected or overlapping splits")
    if sorted(np.concatenate((train_ids, val_ids)).tolist()) != list(range(160)):
        raise ValueError("Split indices must partition all image rows")
    for split, indices in (("train", train_ids), ("validation", val_ids)):
        if any(source["samples"][i]["split"] != split for i in indices):
            raise ValueError("Caption and token split mismatch")
    if data["image_tokens"].min() < 0 or data["image_tokens"].max() >= 8192:
        raise ValueError("Image code outside the MagViT codebook")

    stage = Path(tempfile.mkdtemp(prefix="xxt-showo-phi-tokenizer-", dir="/tmp"))
    print("TOKENIZER_DOWNLOAD_DIR", stage, flush=True)
    tokenizer_repo = "microsoft/phi-1_5"
    revision = HfApi().model_info(tokenizer_repo).sha
    folder = snapshot_download(
        repo_id=tokenizer_repo, revision=revision,
        allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json",
                        "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json"],
        local_dir=str(stage / "files"), cache_dir=str(stage / "cache"), max_workers=1)
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
    reference = args.output / "reference"
    reference.mkdir()
    tokenizer.save_pretrained(reference / "tokenizer")
    base = len(tokenizer)
    special = dict(pad=base, mmu=base + 1, t2i=base + 2, soi=base + 3,
                   eoi=base + 4, eos=base + 5, image_offset=base + 6, mask=base + 6 + 8192)
    text_ids = np.full((160, 5, 64), special["pad"], dtype=np.int32)
    lengths = np.zeros((160, 5), dtype=np.int32)
    truncated = 0
    for i, sample in enumerate(source["samples"]):
        if len(sample["captions"]) != 5:
            raise ValueError("Expected five captions per image")
        for c, caption in enumerate(sample["captions"]):
            ids = tokenizer.encode(caption, add_special_tokens=False)
            if not ids or min(ids) < 0 or max(ids) >= base:
                raise ValueError("Unexpected caption token IDs")
            truncated += len(ids) > 64
            ids = ids[:64]
            lengths[i, c] = len(ids)
            text_ids[i, c, :len(ids)] = ids
    data.update(text_ids=text_ids, text_lengths=lengths, special=special,
                sequence_length=np.array(324, dtype=np.int32))
    np.savez_compressed(reference / "training_data.npz", **{k: v for k, v in data.items() if k != "special"})
    write_json(reference / "dataset.json", dict(
        source_tokens=str(args.tokens), source_manifest_sha256=MANIFEST_SHA, special=special,
        tokenizer_repo=tokenizer_repo, tokenizer_revision=revision, base_vocab_size=base,
        text_max_tokens=64, sequence_length=324, truncated_captions=int(truncated), train_images=128, validation_images=32,
        source_scope=source["scope"], text_condition_dropout=0,
        mask_schedule="ceil(256*cos(pi*U/2)); 10 percent explicitly all-masked"))
    (reference / "source_manifest.json").write_bytes(raw)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    repo = helper.reference_repository(Path.home() / "Show-o-Jittor")
    phi = helper.load_file("flickr_official_phi", repo / "models/phi.py")
    prompting = helper.load_file("flickr_official_prompting", repo / "training/prompting_utils.py")
    verify_masks(torch, prompting, data)
    config = PhiConfig(
        vocab_size=special["mask"] + 1, hidden_size=128, intermediate_size=512,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        partial_rotary_factor=0.5, max_position_embeddings=512, hidden_act="gelu_new",
        layer_norm_eps=1e-5, attention_dropout=0.0, resid_pdrop=0.0, embd_pdrop=0.0,
        pad_token_id=special["pad"], bos_token_id=special["mmu"], eos_token_id=special["eos"],
        tie_word_embeddings=False, use_cache=False, qk_layernorm=True)
    config._attn_implementation = "eager"
    config.to_json_file(reference / "config.json")
    model = phi.PhiForCausalLM(config).cuda()
    batches = {}
    for task in ("mmu", "t2i"):
        batch = make_batch(data, [int(train_ids[0])], [0], task, np.random.default_rng(345), 0.5)
        batches[task] = {key: torch.tensor(value, device="cuda", dtype=(
            torch.float32 if key == "attention_mask" else torch.long)) for key, value in batch.items()}
    losses = helper.export_fixture(torch, np, model, batches, reference)
    if not all(np.isfinite(value) for value in losses.values()):
        raise ValueError("Non-finite real-data reference loss")
    report = dict(status="passed", commit=COMMIT, parameters=sum(p.numel() for p in model.parameters()),
                  fixture_scope="Initial FP32 real Flickr image/caption fixture; newly initialized small model",
                  scope="Reference export and mask oracle; no reference optimizer steps", initial_loss=losses,
                  sources={name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
                           for name in ("models/phi.py", "training/prompting_utils.py")})
    write_json(reference / "metrics.json", report)
    print(json.dumps(report, indent=2), flush=True)
    print("FLICKR_REFERENCE_EXPORT_OK", flush=True)


def alignment_worker(args):
    import numpy as np
    import jittor as jt
    import align_tiny_jittor as port

    jt.flags.use_cuda = 1
    jt.set_global_seed(args.seed)
    reference = args.output / "reference"
    out = args.output / "alignment"
    config = json.loads((reference / "config.json").read_text())
    model = port.model_class(jt, np)(config)
    model.train()
    with np.load(reference / "initial_weights.npz", allow_pickle=False) as archive:
        port.strict_load(model, {key: archive[key] for key in archive.files}, jt, np)
    parameters = sorted(model.named_parameters())
    rows = []
    with np.load(reference / "initial_fixture.npz", allow_pickle=False) as fixture:
        for task in ("mmu", "t2i"):
            batch = port.batch_from_fixture(jt, np, fixture, task)
            logits, loss, trace = port.loss_for(jt, model, task, batch)
            expected = {key.removeprefix(task + ".") for key in fixture.files
                        if key.startswith(task + ".") and not key.startswith(task + ".grad.")
                        and key.removeprefix(task + ".") not in batch}
            if set(trace) != expected:
                raise ValueError("Real-fixture forward trace keys differ")
            for name, value in trace.items():
                kind = "loss" if name == "loss" else "forward"
                rows.append(port.comparison(np, value.numpy(), fixture[task + "." + name], task, kind, name))
            names = {key.removeprefix(task + ".grad.") for key in fixture.files if key.startswith(task + ".grad.")}
            if names != {name for name, _ in parameters}:
                raise ValueError("Real-fixture gradient parameter names differ")
            gradients = jt.grad(loss, [parameter for _, parameter in parameters])
            for (name, _), gradient in zip(parameters, gradients):
                rows.append(port.comparison(np, gradient.numpy(), fixture[task + ".grad." + name],
                                            task, "gradient", name))
            subset = [row for row in rows if row["task"] == task]
            print(json.dumps(dict(task=task, checked=len(subset), failed=sum(not r["passed"] for r in subset))), flush=True)
            del logits, loss, trace, gradients
    with (out / "alignment.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    failed = [row for row in rows if not row["passed"]]
    report = dict(status="failed" if failed else "passed", reference=str(reference), checked=len(rows),
                  scope="Initial FP32 real Flickr fixture, fixed 324 positions including padding; forward/loss/all parameter gradients",
                  forward_atol=1e-4, loss_atol=1e-5, gradient_atol=1e-5, gradient_rtol=1e-3,
                  failed=len(failed), worst=max(rows, key=lambda row: row["max_abs"]))
    write_json(out / "alignment.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if failed:
        raise RuntimeError("Real-data alignment exceeded preset tolerances; inspect alignment.csv")
    print("FLICKR_REAL_TOKEN_ALIGNMENT_OK", flush=True)


def jt_batch(jt, batch):
    return {key: jt.array(value).stop_grad() for key, value in batch.items()}


def evaluate(jt, np, port, model, data, indices, batch_size):
    report = {}
    for case in ("mmu", "t2i", "t2i_all_masks"):
        task = "mmu" if case == "mmu" else "t2i"
        rng = np.random.default_rng(81031)
        loss_sum, count, correct = 0.0, 0, 0
        with jt.no_grad():
            for start in range(0, len(indices), batch_size):
                rows = indices[start:start + batch_size]
                raw = make_batch(data, rows, np.zeros(len(rows), dtype=int), task, rng,
                                 1.0 if case == "t2i_all_masks" else 0.5)
                logits, loss, trace = port.loss_for(jt, model, task, jt_batch(jt, raw))
                predictions = logits.argmax(dim=-1)[0].numpy()
                labels = raw["labels"]
                if task == "mmu":
                    predictions, labels = predictions[:, :-1], labels[:, 1:]
                valid = labels != -100
                n = int(valid.sum())
                value = float(loss.numpy().item())
                if not np.isfinite(value):
                    raise ValueError("Non-finite fixed evaluation loss")
                loss_sum += value * n
                count += n
                correct += int((predictions[valid] == labels[valid]).sum())
                del logits, loss, trace
        report[case] = dict(loss=loss_sum / count, token_accuracy=correct / count, tokens=count)
    return report


def save_weights(np, model, path):
    state = {name: value.numpy() for name, value in model.state_dict().items()}
    if not all(np.isfinite(value).all() for value in state.values()):
        raise ValueError("Non-finite checkpoint parameter")
    with path.open("xb") as stream:
        np.savez(stream, **state)


def train_worker(args):
    import numpy as np
    import jittor as jt
    import align_tiny_jittor as port

    alignment = json.loads((args.output / "alignment/alignment.json").read_text())
    if alignment["status"] != "passed":
        raise ValueError("Real-data initial forward and gradient alignment must pass first")
    reference = args.output / "reference"
    config = json.loads((reference / "config.json").read_text())
    data = load_data(reference)
    jt.flags.use_cuda = 1
    jt.set_global_seed(args.seed)
    Model = port.model_class(jt, np)
    model = Model(config)
    model.train()
    with np.load(reference / "initial_weights.npz", allow_pickle=False) as archive:
        port.strict_load(model, {key: archive[key] for key in archive.files}, jt, np)
    optimizer = jt.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    rng = np.random.default_rng(args.seed)
    train_ids, val_ids = data["train_indices"], data["validation_indices"]
    monitor = train_ids[:16]
    history = []
    best_score, best_checkpoint, best_step = float("inf"), "reference/initial_weights.npz", 0

    def evaluate_step(step, checkpoint):
        nonlocal best_score, best_checkpoint, best_step
        row = dict(step=step, train_monitor_images=16, validation_images=32,
                   train_monitor=evaluate(jt, np, port, model, data, monitor, args.batch_size),
                   validation=evaluate(jt, np, port, model, data, val_ids, args.batch_size))
        score = sum(row["validation"][task]["loss"] for task in ("mmu", "t2i"))
        if score < best_score:
            best_score, best_checkpoint, best_step = score, checkpoint, step
        history.append(row)
        with (args.output / "evaluation.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        write_json(args.output / "best_checkpoint.json", dict(
            checkpoint=best_checkpoint, selection="validation MMU loss + fixed 50%-mask T2I loss",
            score=best_score, best_step=best_step, last_evaluated_step=step, weights_only=True))
        print("FIXED_EVALUATION", json.dumps(row), flush=True)

    started = time.perf_counter()
    evaluate_step(0, best_checkpoint)
    with (args.output / "loss.csv").open("x", newline="") as stream:
        fields = ["step", "mmu_loss", "t2i_loss", "total_loss", "elapsed_seconds"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for step in range(1, args.steps + 1):
            indices = rng.choice(train_ids, size=args.batch_size, replace=False)
            captions = rng.integers(0, 5, size=args.batch_size)
            losses = {}
            for task in ("mmu", "t2i"):
                batch = make_batch(data, indices, captions, task, rng)
                losses[task] = port.loss_for(jt, model, task, jt_batch(jt, batch))[1]
            total = losses["mmu"] + losses["t2i"]
            values = {task: float(loss.numpy().item()) for task, loss in losses.items()}
            if not all(np.isfinite(value) for value in values.values()):
                raise ValueError("Non-finite training loss")
            optimizer.zero_grad()
            optimizer.backward(total)
            optimizer.clip_grad_norm(1.0, 2)
            optimizer.step()
            jt.sync_all(True)
            row = dict(step=step, mmu_loss=values["mmu"], t2i_loss=values["t2i"],
                       total_loss=sum(values.values()), elapsed_seconds=time.perf_counter() - started)
            writer.writerow(row)
            stream.flush()
            if step == 1 or step % 10 == 0:
                print(json.dumps(row), flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                name = "weights-step-" + str(step).zfill(6) + ".npz"
                save_weights(np, model, args.output / name)
                evaluate_step(step, name)
    probe_raw = make_batch(data, [int(train_ids[0])], [0], "mmu", np.random.default_rng(100), 0.5)
    probe = jt_batch(jt, probe_raw)
    with jt.no_grad():
        expected = port.loss_for(jt, model, "mmu", probe)[0].numpy()
    restored = Model(config)
    with np.load(args.output / name, allow_pickle=False) as archive:
        port.strict_load(restored, {key: archive[key] for key in archive.files}, jt, np)
    with jt.no_grad():
        actual = port.loss_for(jt, restored, "mmu", probe)[0].numpy()
    reload_error = float(np.abs(expected - actual).max())
    if not np.allclose(expected, actual, atol=1e-6, rtol=1e-6):
        raise ValueError("Final checkpoint reload output mismatch")
    improved = all(history[-1]["train_monitor"][task]["loss"] < history[0]["train_monitor"][task]["loss"]
                   for task in ("mmu", "t2i"))
    report = dict(status="passed" if improved else "train_monitor_not_improved", steps=args.steps,
                  seed=args.seed, batch_size=args.batch_size, learning_rate=args.lr,
                  scope="Small newly initialized Jittor model on real Flickr tokens; not paper benchmark reproduction",
                  parameters=sum(int(np.prod(p.shape)) for p in model.parameters()),
                  initial=history[0], final=history[-1], best_checkpoint=best_checkpoint,
                  final_checkpoint=name, checkpoint_max_abs_error=reload_error,
                  elapsed_seconds=time.perf_counter() - started, weights_only_checkpoints=True)
    write_json(args.output / "training_report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if not improved:
        raise RuntimeError("Training ended but fixed train-monitor losses did not both improve; inspect logs")
    print("FLICKR_JITTOR_TRAIN_OK", flush=True)
    print("REAL_CHECKPOINT_RELOAD_OK", flush=True)


def worker_environment(args, jittor):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), NVIDIA_TF32_OVERRIDE="0",
               CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONUNBUFFERED="1",
               TOKENIZERS_PARALLELISM="false", HF_HUB_DISABLE_XET="1")
    if jittor:
        view = Path.home() / "Show-o-Jittor/deps/cuda-jittor"
        env.update(CUDA_HOME=str(view), nvcc_path=str(view / "bin/nvcc"),
                   JITTOR_HOME="/tmp/xxt-showo-jittor", DISABLE_MULTIPROCESSING="1", HF_HUB_OFFLINE="1")
        env["PATH"] = str(view / "bin") + ":" + env.get("PATH", "")
        old = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = str(view / "lib64") + (":" + old if old else "")
    return env


def run_stage(command, env, log):
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace")
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
            stream.flush()
        code = process.wait()
        stream.write("\nExit code: " + str(code) + "\n")
    if code:
        raise RuntimeError("Stage failed with exit code " + str(code) + "; log: " + str(log))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path, default=Path.home() / "Show-o-Jittor/data" / DEFAULT_DATA)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--worker", choices=["reference", "alignment", "train"], help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if Path(sys.prefix).resolve() != (Path.home() / "envs/showo-jittor").resolve():
        raise RuntimeError("Activate the configured showo-jittor environment")
    if args.steps < 1 or args.eval_every < 1 or not 1 <= args.batch_size <= 128 or not args.lr > 0:
        parser.error("Steps/eval interval/lr must be positive; batch size must be 1..128")
    if args.worker:
        {"reference": prepare_reference, "alignment": alignment_worker, "train": train_worker}[args.worker](args)
        return
    project = Path.home() / "Show-o-Jittor"
    for required in ("train_tiny_reference.py", "align_tiny_jittor.py", "deps/cuda-jittor/bin/nvcc"):
        if not (project / required).is_file():
            raise FileNotFoundError(project / required)
    args.tokens = args.tokens.resolve(strict=True)
    usage = subprocess.check_output(
        ["nvidia-smi", "-i", str(args.gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True).strip()
    if int(usage) > 1024:
        raise RuntimeError("Selected GPU already uses " + usage + " MiB; inspect allocation before starting")
    args.output = project / "results" / ("flickr-jittor-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), args.output / "training_script.py")
    write_json(args.output / "arguments.json", {key: str(value) if isinstance(value, Path) else value
                                              for key, value in vars(args).items()})
    print("RUN_DIR", args.output, flush=True)
    base = [sys.executable, "-u", str(Path(__file__).resolve()), "--output", str(args.output),
            "--tokens", str(args.tokens), "--gpu", str(args.gpu), "--steps", str(args.steps),
            "--batch-size", str(args.batch_size), "--eval-every", str(args.eval_every),
            "--lr", str(args.lr), "--seed", str(args.seed)]
    try:
        run_stage(base + ["--worker", "reference"], worker_environment(args, False), args.output / "reference.log")
        alignment = args.output / "alignment"
        alignment.mkdir()
        run_stage(base + ["--worker", "alignment"], worker_environment(args, True), args.output / "alignment.log")
        run_stage(base + ["--worker", "train"], worker_environment(args, True), args.output / "train.log")
    except Exception as error:
        write_json(args.output / "run_status.json", dict(status="failed", error=str(error)))
        raise
    write_json(args.output / "run_status.json", dict(status="completed", steps=args.steps))
    print("FLICKR_RUN_COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
