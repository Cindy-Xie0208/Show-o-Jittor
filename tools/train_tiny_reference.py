"""Server-only Tiny reference using the pinned official Show-o Phi and masks."""

import argparse
import csv
from datetime import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


COMMIT = "45a5a2de01d1ebd10cd5864d29310a76476cdf23"
UPSTREAM = "https://github.com/showlab/Show-o.git"
TOKENS = dict(pad=0, mmu=1, t2i=2, soi=3, eoi=4, eos=5, mask=255)


def reference_repository(project):
    repo = project / "official"
    if not repo.exists():
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", UPSTREAM, str(repo)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repo), "checkout", "-b", "feature/tiny-reference", COMMIT],
            check=True,
        )
    actual = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != COMMIT:
        raise RuntimeError(f"Expected {COMMIT}, found {actual}; existing checkout preserved")
    subprocess.run(["git", "-C", str(repo), "diff", "--quiet", "HEAD", "--"], check=True)
    return repo


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_batches(torch, prompting, device):
    generator = torch.Generator().manual_seed(20261026)
    batch_size, image_length, text_length = 4, 16, 8
    text = torch.stack([torch.arange(text_length) + 16 + i * 8 for i in range(batch_size)])
    images = torch.stack(
        [128 + (torch.arange(image_length) + i * 7) % 64 for i in range(batch_size)]
    )

    def special(name):
        return torch.full((batch_size, 1), TOKENS[name], dtype=torch.long)

    mmu_ids = torch.cat(
        [special("mmu"), special("soi"), images, special("eoi"), text, special("eos")], 1
    )
    mmu_labels = mmu_ids.clone()
    prefix = 3 + image_length
    mmu_labels[:, :prefix] = -100
    mmu_allowed = prompting.create_attention_mask_for_mmu(
        mmu_ids, eoi_id=TOKENS["eoi"], return_inverse_mask=False
    ).bool()
    positions = torch.arange(mmu_ids.shape[1])
    expected = (positions[None, :] <= positions[:, None]) | (positions[None, :] < prefix)
    assert torch.equal(mmu_allowed, expected[None, None].expand(batch_size, 1, -1, -1))

    selected = torch.rand(images.shape, generator=generator) < 0.5
    selected[:, 0] = True
    corrupted = images.masked_fill(selected, TOKENS["mask"])
    t2i_ids = torch.cat(
        [special("t2i"), text, special("soi"), corrupted, special("eoi")], 1
    )
    t2i_labels = torch.full_like(t2i_ids, -100)
    start = text_length + 2
    t2i_labels[:, start : start + image_length] = images.masked_fill(~selected, -100)
    t2i_allowed = prompting.create_attention_mask_predict_next(
        t2i_ids,
        pad_id=TOKENS["pad"],
        soi_id=TOKENS["soi"],
        eoi_id=TOKENS["eoi"],
        return_inverse_mask=False,
    ).bool()
    expected = torch.ones((t2i_ids.shape[1], t2i_ids.shape[1]), dtype=torch.bool).tril()
    expected[:, :] |= (torch.arange(t2i_ids.shape[1])[:, None] >= start - 1)
    assert torch.equal(t2i_allowed, expected[None, None].expand(batch_size, 1, -1, -1))

    batches = {}
    for name, ids, labels, allowed in (
        ("mmu", mmu_ids, mmu_labels, mmu_allowed),
        ("t2i", t2i_ids, t2i_labels, t2i_allowed),
    ):
        additive = torch.zeros(allowed.shape, dtype=torch.float32)
        additive.masked_fill_(~allowed, torch.finfo(torch.float32).min)
        batches[name] = {
            "input_ids": ids.to(device),
            "labels": labels.to(device),
            "attention_mask": additive.to(device),
        }
    print("OFFICIAL_MASK_CHECK_OK", flush=True)
    return batches


def forward_loss(torch, model, task, batch):
    logits = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    ).logits
    labels = batch["labels"]
    if task == "mmu":
        logits_for_loss, labels = logits[:, :-1], labels[:, 1:]
    else:
        logits_for_loss = logits
    loss = torch.nn.functional.cross_entropy(
        logits_for_loss.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100
    )
    return logits, loss


def export_fixture(torch, np, model, batches, out):
    tensors = {}
    model.eval()
    for task, batch in batches.items():
        handles = []

        def hook(name):
            def capture(module, inputs, result):
                value = result[0] if isinstance(result, (tuple, list)) else result
                tensors[f"{task}.{name}"] = value.detach().cpu().numpy()

            return capture

        wanted = {"model.embed_tokens", "model.final_layernorm", "lm_head"}
        for i in range(model.config.num_hidden_layers):
            base = f"model.layers.{i}"
            wanted.update(
                {
                    base,
                    base + ".input_layernorm",
                    base + ".self_attn",
                    base + ".self_attn.q_proj",
                    base + ".self_attn.k_proj",
                    base + ".self_attn.v_proj",
                    base + ".self_attn.q_layernorm",
                    base + ".self_attn.k_layernorm",
                    base + ".mlp",
                }
            )
        for name, module in model.named_modules():
            if name in wanted:
                handles.append(module.register_forward_hook(hook(name)))
        model.zero_grad(set_to_none=True)
        try:
            logits, loss = forward_loss(torch, model, task, batch)
            loss.backward()
            tensors[f"{task}.logits"] = logits.detach().cpu().numpy()
            tensors[f"{task}.loss"] = loss.detach().cpu().numpy()
            for name, value in batch.items():
                tensors[f"{task}.{name}"] = value.detach().cpu().numpy()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    grad = parameter.grad.detach().cpu().numpy()
                    assert np.isfinite(grad).all(), f"Non-finite initial gradient: {name}"
                    tensors[f"{task}.grad.{name}"] = grad
        finally:
            for handle in handles:
                handle.remove()
    np.savez(out / "initial_fixture.npz", **tensors)
    np.savez(
        out / "initial_weights.npz",
        **{name: value.detach().cpu().numpy() for name, value in model.state_dict().items()},
    )
    model.zero_grad(set_to_none=True)
    return {task: float(tensors[f"{task}.loss"]) for task in batches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if Path(sys.prefix).resolve() != (Path.home() / "envs/showo-jittor").resolve():
        raise RuntimeError("Activate the showo-jittor environment first")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import numpy as np
    import torch
    from transformers import PhiConfig

    if importlib.metadata.version("transformers") != "4.41.1":
        raise RuntimeError("This reference requires transformers==4.41.1")
    assert torch.cuda.is_available(), "CUDA unavailable"
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    project = Path.home() / "Show-o-Jittor"
    repo = reference_repository(project)
    phi = load_file("official_showo_phi", repo / "models/phi.py")
    prompting = load_file("official_showo_prompting", repo / "training/prompting_utils.py")
    config = PhiConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        partial_rotary_factor=0.5,
        max_position_embeddings=128,
        hidden_act="gelu_new",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        pad_token_id=TOKENS["pad"],
        bos_token_id=TOKENS["mmu"],
        eos_token_id=TOKENS["eos"],
        tie_word_embeddings=False,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    model = phi.PhiForCausalLM(config).cuda()
    out = project / "results" / ("tiny-reference-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    out.mkdir(parents=True)
    print("Results:", out, flush=True)
    config.to_json_file(out / "config.json")
    batches = make_batches(torch, prompting, "cuda")
    initial = export_fixture(torch, np, model, batches, out)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with (out / "loss.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["step", "mmu_loss", "t2i_loss", "total_loss"])
        writer.writeheader()
        model.train()
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            losses = {name: forward_loss(torch, model, name, batch)[1] for name, batch in batches.items()}
            total = sum(losses.values())
            assert torch.isfinite(total).item(), "Non-finite training loss"
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            row = dict(step=step, mmu_loss=losses["mmu"].item(), t2i_loss=losses["t2i"].item(), total_loss=total.item())
            writer.writerow(row)
            stream.flush()
            if step == 1 or step % 10 == 0 or step == args.steps:
                print(json.dumps(row), flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_memory = torch.cuda.max_memory_allocated()
    checkpoint = out / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    model.eval()
    with torch.no_grad():
        reference = {name: forward_loss(torch, model, name, batch) for name, batch in batches.items()}
        final = {name: value[1].item() for name, value in reference.items()}
        loaded = phi.PhiForCausalLM(config).cuda().eval()
        loaded.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        reload_error = 0.0
        for name, batch in batches.items():
            logits, _ = forward_loss(torch, loaded, name, batch)
            error = (logits - reference[name][0]).abs().max().item()
            reload_error = max(reload_error, error)
            assert torch.allclose(logits, reference[name][0], atol=1e-6, rtol=1e-6), "Checkpoint mismatch"
    report = dict(
        status="passed" if all(final[name] < initial[name] for name in final) else "loss_not_decreased",
        scope="Synthetic token overfit smoke test; not real-image training or paper benchmark reproduction",
        commit=COMMIT,
        sources={name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in ("models/phi.py", "training/prompting_utils.py")},
        versions={name: importlib.metadata.version(name) for name in ("torch", "numpy", "transformers")},
        seed=args.seed,
        steps=args.steps,
        gpu=torch.cuda.get_device_name(0),
        physical_gpu=args.gpu,
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        tokens=TOKENS,
        initial_loss=initial,
        final_loss=final,
        checkpoint_max_abs_error=reload_error,
        train_seconds=elapsed,
        peak_allocated_bytes=peak_memory,
    )
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    with (out / "environment.txt").open("x") as stream:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=stream, check=True)
    print(json.dumps(report, indent=2), flush=True)
    if report["status"] != "passed":
        raise RuntimeError("Both task losses must decrease; inspect loss.csv")
    print("TINY_REFERENCE_TRAIN_OK", flush=True)
    print("CHECKPOINT_RELOAD_OK", flush=True)
    print("Results:", out, flush=True)


if __name__ == "__main__":
    main()
