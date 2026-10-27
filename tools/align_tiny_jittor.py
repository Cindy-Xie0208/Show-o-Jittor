# Adapted from Show-o models/phi.py at 45a5a2d.
# Original copyright 2023 Microsoft and the HuggingFace Inc. team.
# Original Show-o copyright 2024 NUS Show Lab, HuggingFace.
# Licensed under the Apache License, Version 2.0.
# https://www.apache.org/licenses/LICENSE-2.0
# Changes: native Jittor Tiny model, strict fixture alignment and smoke training.
"""Run only on the server; compare native Jittor with exported official Phi."""

import argparse
import csv
from datetime import datetime
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


COMMIT = "45a5a2de01d1ebd10cd5864d29310a76476cdf23"


def model_class(jt, np):
    nn = jt.nn

    class Attention(nn.Module):
        def __init__(self, config):
            self.hidden = config["hidden_size"]
            self.heads = config["num_attention_heads"]
            self.head_dim = self.hidden // self.heads
            self.rot_dim = int(config["partial_rotary_factor"] * self.head_dim)
            self.q_proj = nn.Linear(self.hidden, self.hidden)
            self.k_proj = nn.Linear(self.hidden, self.hidden)
            self.v_proj = nn.Linear(self.hidden, self.hidden)
            self.dense = nn.Linear(self.hidden, self.hidden)
            self.q_layernorm = nn.LayerNorm(self.head_dim, eps=config["layer_norm_eps"])
            self.k_layernorm = nn.LayerNorm(self.head_dim, eps=config["layer_norm_eps"])
            if self.rot_dim < 2 or self.rot_dim % 2:
                raise ValueError("Rotary dimension must be positive and even")
            inv = np.float32(1.0) / np.power(
                np.float32(config["rope_theta"]),
                np.arange(0, self.rot_dim, 2, dtype=np.float32) / np.float32(self.rot_dim),
            )
            phase = np.outer(np.arange(config["max_position_embeddings"], dtype=np.float32), inv)
            phase = np.concatenate((phase, phase), axis=-1)
            self._cos = jt.array(np.cos(phase).astype(np.float32))[None, None].stop_grad()
            self._sin = jt.array(np.sin(phase).astype(np.float32))[None, None].stop_grad()

        def rotate(self, x):
            part, tail = x[..., : self.rot_dim], x[..., self.rot_dim :]
            half = self.rot_dim // 2
            rotated = jt.concat((-part[..., half:], part[..., :half]), dim=-1)
            length = x.shape[2]
            part = part * self._cos[:, :, :length] + rotated * self._sin[:, :, :length]
            return jt.concat((part, tail), dim=-1)

        def execute(self, hidden, mask, trace, prefix):
            batch, length, _ = hidden.shape
            projected = []
            for name in ("q_proj", "k_proj", "v_proj"):
                value = getattr(self, name)(hidden)
                trace[prefix + "." + name] = value
                projected.append(value.reshape((batch, length, self.heads, self.head_dim)).transpose((0, 2, 1, 3)))
            query, key, value = projected
            query = self.q_layernorm(query)
            key = self.k_layernorm(key)
            trace[prefix + ".q_layernorm"] = query
            trace[prefix + ".k_layernorm"] = key
            query, key = self.rotate(query), self.rotate(key)
            scores = jt.matmul(query.float32(), key.transpose((0, 1, 3, 2)).float32())
            scores = scores / math.sqrt(self.head_dim) + mask
            probabilities = nn.softmax(scores, dim=-1)
            attended = jt.matmul(probabilities, value)
            attended = attended.transpose((0, 2, 1, 3)).reshape((batch, length, self.hidden))
            output = self.dense(attended)
            trace[prefix] = output
            return output

    class MLP(nn.Module):
        def __init__(self, config):
            self.fc1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
            self.fc2 = nn.Linear(config["intermediate_size"], config["hidden_size"])

        def execute(self, hidden):
            x = self.fc1(hidden)
            x = 0.5 * x * (1.0 + jt.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))
            return self.fc2(x)

    class Block(nn.Module):
        def __init__(self, config):
            self.input_layernorm = nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"])
            self.self_attn = Attention(config)
            self.mlp = MLP(config)

        def execute(self, hidden, mask, trace, prefix):
            normalized = self.input_layernorm(hidden)
            trace[prefix + ".input_layernorm"] = normalized
            attention = self.self_attn(normalized, mask, trace, prefix + ".self_attn")
            feed_forward = self.mlp(normalized)
            trace[prefix + ".mlp"] = feed_forward
            output = attention + feed_forward + hidden
            trace[prefix] = output
            return output

    class Backbone(nn.Module):
        def __init__(self, config):
            self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"], padding_idx=config["pad_token_id"])
            self.layers = nn.ModuleList([Block(config) for _ in range(config["num_hidden_layers"])])
            self.final_layernorm = nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"])

        def execute(self, ids, mask, trace):
            hidden = self.embed_tokens(ids)
            trace["model.embed_tokens"] = hidden
            for index, layer in enumerate(self.layers):
                hidden = layer(hidden, mask, trace, f"model.layers.{index}")
            hidden = self.final_layernorm(hidden)
            trace["model.final_layernorm"] = hidden
            return hidden

    class TinyPhi(nn.Module):
        def __init__(self, config):
            super().__init__()
            if config["num_key_value_heads"] != config["num_attention_heads"]:
                raise ValueError("This Tiny configuration requires equal Q and KV heads")
            if not config.get("qk_layernorm") or config["hidden_act"] != "gelu_new":
                raise ValueError("Expected Show-o QK normalization and gelu_new")
            if config["hidden_size"] % config["num_attention_heads"]:
                raise ValueError("Hidden size must be divisible by head count")
            if any(config.get(key, 0.0) for key in ("attention_dropout", "resid_pdrop", "embd_pdrop")):
                raise ValueError("Reference alignment requires dropout=0")
            if config.get("tie_word_embeddings") or config.get("rope_scaling"):
                raise ValueError("Tied embeddings and scaled RoPE are outside this Tiny configuration")
            self.model = Backbone(config)
            self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"])

        def execute(self, ids, mask):
            trace = {}
            logits = self.lm_head(self.model(ids, mask, trace)).float32()
            trace["lm_head"] = logits
            trace["logits"] = logits
            return logits, trace

    return TinyPhi


def strict_load(model, weights, jt, np):
    state = model.state_dict()
    if set(state) != set(weights):
        raise ValueError(
            f"Parameter mismatch: missing={sorted(set(state) - set(weights))}, "
            f"extra={sorted(set(weights) - set(state))}"
        )
    for name, value in weights.items():
        if tuple(state[name].shape) != tuple(value.shape):
            raise ValueError(f"Wrong shape for {name}: {state[name].shape} vs {value.shape}")
        state[name].assign(jt.array(value))
    for name, parameter in state.items():
        if not np.array_equal(parameter.numpy(), weights[name]):
            raise ValueError(f"Parameter did not load exactly: {name}")


def batch_from_fixture(jt, np, fixture, task):
    batch = {}
    for name in ("input_ids", "labels", "attention_mask"):
        array = fixture[f"{task}.{name}"]
        dtype = np.float32 if name == "attention_mask" else np.int32
        batch[name] = jt.array(array.astype(dtype)).stop_grad()
    return batch


def loss_for(jt, model, task, batch):
    logits, trace = model(batch["input_ids"], batch["attention_mask"])
    if task == "mmu":
        scores, labels = logits[:, :-1], batch["labels"][:, 1:]
    else:
        scores, labels = logits, batch["labels"]
    loss = jt.nn.cross_entropy_loss(scores.reshape((-1, logits.shape[-1])), labels.reshape((-1,)), ignore_index=-100)
    trace["loss"] = loss
    return logits, loss, trace


def comparison(np, actual, expected, task, kind, name):
    if actual.shape != expected.shape:
        if actual.size == expected.size == 1:
            actual, expected = actual.reshape(()), expected.reshape(())
        else:
            raise ValueError(f"Shape mismatch at {task}/{name}: {actual.shape} vs {expected.shape}")
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    absolute = np.abs(difference)
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    atol = 1e-5 if kind in ("loss", "gradient") else 1e-4
    rtol = 1e-3 if kind == "gradient" else 0.0
    passed = finite and bool(np.allclose(actual, expected, atol=atol, rtol=rtol))
    return dict(
        task=task,
        kind=kind,
        name=name,
        shape=str(tuple(expected.shape)),
        max_abs=float(absolute.max()),
        mean_abs=float(absolute.mean()),
        relative_l2=float(np.linalg.norm(difference.ravel()) / max(np.linalg.norm(expected.ravel()), 1e-12)),
        atol=atol,
        rtol=rtol,
        passed=passed,
    )


def train_smoke(jt, np, model, config, batches, steps, out, TinyPhi):
    initial = {task: float(loss_for(jt, model, task, batch)[1].numpy().item()) for task, batch in batches.items()}
    optimizer = jt.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    jt.sync_all(True)
    started = time.perf_counter()
    with (out / "jittor_loss.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["step", "mmu_loss", "t2i_loss", "total_loss"])
        writer.writeheader()
        for step in range(1, steps + 1):
            losses = {task: loss_for(jt, model, task, batch)[1] for task, batch in batches.items()}
            total = losses["mmu"] + losses["t2i"]
            values = {task: float(loss.numpy().item()) for task, loss in losses.items()}
            if not all(np.isfinite(value) for value in values.values()):
                raise ValueError("Non-finite training loss")
            optimizer.zero_grad()
            optimizer.backward(total)
            optimizer.clip_grad_norm(1.0, 2)
            optimizer.step()
            jt.sync_all(True)
            row = dict(step=step, mmu_loss=values["mmu"], t2i_loss=values["t2i"], total_loss=sum(values.values()))
            writer.writerow(row)
            stream.flush()
            if step == 1 or step % 10 == 0 or step == steps:
                print(json.dumps(row), flush=True)
    elapsed = time.perf_counter() - started
    outputs = {task: loss_for(jt, model, task, batch) for task, batch in batches.items()}
    final = {task: float(result[1].numpy().item()) for task, result in outputs.items()}
    saved = {name: value.numpy() for name, value in model.state_dict().items()}
    if not all(np.isfinite(value).all() for value in saved.values()):
        raise ValueError("Non-finite trained parameter")
    np.savez(out / "jittor_weights.npz", **saved)
    restored = TinyPhi(config)
    with np.load(out / "jittor_weights.npz", allow_pickle=False) as archive:
        strict_load(restored, {key: archive[key] for key in archive.files}, jt, np)
    reload_error = 0.0
    for task, batch in batches.items():
        expected = outputs[task][0].numpy()
        actual = loss_for(jt, restored, task, batch)[0].numpy()
        reload_error = max(reload_error, float(np.abs(actual - expected).max()))
        if not np.allclose(actual, expected, atol=1e-6, rtol=1e-6):
            raise ValueError("Jittor checkpoint reload mismatch")
    report = dict(initial_loss=initial, final_loss=final, steps=steps, train_seconds=elapsed, checkpoint_max_abs_error=reload_error)
    (out / "jittor_training.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not all(final[task] < initial[task] for task in final):
        raise ValueError("Both Jittor task losses must decrease")
    print("TINY_JITTOR_TRAIN_OK", flush=True)
    print("JITTOR_CHECKPOINT_RELOAD_OK", flush=True)


def worker(args):
    import numpy as np
    import jittor as jt

    jt.flags.use_cuda = 1
    jt.set_global_seed(42)
    config = json.loads((args.reference / "config.json").read_text())
    metadata = json.loads((args.reference / "metrics.json").read_text())
    if metadata["commit"] != COMMIT or metadata["status"] != "passed":
        raise ValueError("Expected a successful reference at the pinned official commit")
    TinyPhi = model_class(jt, np)
    model = TinyPhi(config)
    model.train()
    with np.load(args.reference / "initial_weights.npz", allow_pickle=False) as archive:
        strict_load(model, {key: archive[key] for key in archive.files}, jt, np)
    count = sum(int(np.prod(value.shape)) for value in model.state_dict().values())
    if count != metadata["parameters"]:
        raise ValueError(f"Parameter count mismatch: {count} vs {metadata['parameters']}")
    print("JITTOR_WEIGHTS_LOADED_EXACTLY", count, flush=True)
    rows = []
    outputs = {}
    batches = {}
    parameters = sorted(model.named_parameters())
    with np.load(args.reference / "initial_fixture.npz", allow_pickle=False) as fixture:
        for task in ("mmu", "t2i"):
            batch = batch_from_fixture(jt, np, fixture, task)
            batches[task] = batch
            if np.any(fixture[f"{task}.input_ids"] == config["pad_token_id"]):
                raise ValueError("This first alignment fixture must not contain padding")
            logits, loss, trace = loss_for(jt, model, task, batch)
            expected_keys = {
                key.removeprefix(task + ".")
                for key in fixture.files
                if key.startswith(task + ".")
                and not key.startswith(task + ".grad.")
                and key.removeprefix(task + ".") not in batch
            }
            if set(trace) != expected_keys:
                raise ValueError(f"Trace keys mismatch: {set(trace) ^ expected_keys}")
            for name, value in trace.items():
                actual = value.numpy()
                outputs[f"{task}.{name}"] = actual
                kind = "loss" if name == "loss" else "forward"
                rows.append(comparison(np, actual, fixture[f"{task}.{name}"], task, kind, name))
            expected_grads = {key.removeprefix(task + ".grad.") for key in fixture.files if key.startswith(task + ".grad.")}
            if expected_grads != {name for name, _ in parameters}:
                raise ValueError("Gradient parameter names do not match the fixture")
            gradients = jt.grad(loss, [parameter for _, parameter in parameters])
            for (name, _), gradient in zip(parameters, gradients):
                actual = gradient.numpy()
                outputs[f"{task}.grad.{name}"] = actual
                rows.append(comparison(np, actual, fixture[f"{task}.grad.{name}"], task, "gradient", name))
            subset = [row for row in rows if row["task"] == task]
            summary = dict(task=task, checked=len(subset), failed=sum(not row["passed"] for row in subset))
            print(json.dumps(summary), flush=True)
    with (args.output / "alignment.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez(args.output / "jittor_initial_outputs.npz", **outputs)
    failed = [row for row in rows if not row["passed"]]
    report = dict(
        status="passed" if not failed else "failed",
        scope="Initial FP32 outputs and parameter gradients on fixed synthetic token fixtures",
        reference=str(args.reference),
        commit=COMMIT,
        jittor_version=importlib.metadata.version("jittor"),
        use_cuda=int(jt.flags.use_cuda),
        parameters=count,
        rows=len(rows),
        failed=len(failed),
        tolerances=dict(forward_atol=1e-4, loss_atol=1e-5, grad_atol=1e-5, grad_rtol=1e-3),
        worst_forward=max((row for row in rows if row["kind"] == "forward"), key=lambda row: row["max_abs"]),
        worst_gradient=max((row for row in rows if row["kind"] == "gradient"), key=lambda row: row["max_abs"]),
    )
    (args.output / "alignment.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if failed:
        for row in sorted(failed, key=lambda row: row["max_abs"], reverse=True)[:10]:
            print("ALIGNMENT_FAILURE", json.dumps(row), flush=True)
        raise RuntimeError("Alignment exceeded preset tolerances; inspect alignment.csv")
    print("TINY_JITTOR_FORWARD_ALIGN_OK", flush=True)
    print("TINY_JITTOR_GRAD_ALIGN_OK", flush=True)
    if args.train_steps:
        train_smoke(jt, np, model, config, batches, args.train_steps, args.output, TinyPhi)
    print("Results:", args.output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.train_steps < 0:
        parser.error("--train-steps must be nonnegative")
    if Path(sys.prefix).resolve() != (Path.home() / "envs/showo-jittor").resolve():
        raise RuntimeError("Activate the showo-jittor environment")
    args.reference = args.reference.resolve(strict=True)
    if args.worker:
        worker(args)
        return 0
    project = Path.home() / "Show-o-Jittor"
    view = project / "deps/cuda-jittor"
    if not (view / "bin/nvcc").is_file():
        raise FileNotFoundError("Run check_jittor_gpu.py to prepare the private CUDA view first")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = project / "results" / ("tiny-jittor-" + stamp)
    output.mkdir(parents=True)
    env = os.environ.copy()
    env.update(
        CUDA_HOME=str(view),
        nvcc_path=str(view / "bin/nvcc"),
        JITTOR_HOME="/tmp/xxt-showo-jittor",
        DISABLE_MULTIPROCESSING="1",
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        PYTHONUNBUFFERED="1",
        NVIDIA_TF32_OVERRIDE="0",
    )
    env["PATH"] = str(view / "bin") + ":" + env.get("PATH", "")
    old_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = str(view / "lib64") + (":" + old_ld if old_ld else "")
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--reference", str(args.reference), "--output", str(output), "--gpu", str(args.gpu), "--train-steps", str(args.train_steps)]
    with (output / "run.log").open("x", encoding="utf-8") as stream:
        print("Results:", output, flush=True)
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
            stream.flush()
        code = process.wait()
        stream.write(f"\nExit code: {code}\n")
    print(f"Exit code: {code}; log: {output / 'run.log'}", flush=True)
    return code if code >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
