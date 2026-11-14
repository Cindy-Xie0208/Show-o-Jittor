"""Server-only Show-o 1 sampler contracts; no model weights are loaded.

The official and native loops receive the same synthetic logits, controlled
categorical choices and Gumbel uniforms. This isolates control flow from RNG
algorithms. Native categorical/Gumbel primitives are smoke-tested separately.
It does not prove full-model generation or cross-framework RNG equality.
"""

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
import types

import numpy as np

COMMIT = "45a5a2de01d1ebd10cd5864d29310a76476cdf23"
CASES = ("cfg-all", "cfg-partial", "no-cfg")


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def extract(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(nodes) != len(names):
        raise RuntimeError("Official function selection is ambiguous")
    for node in nodes:
        node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


def inputs(case):
    ids = np.array([[1, 2, 3, 4] + [18] * 16 + [5]], dtype=np.int32)
    if case == "cfg-partial":
        ids[0, 4:12] = np.arange(8, dtype=np.int32) + 10
    uncond = ids.copy()
    uncond[0, :3] = [6, 2, 7]
    return ids, uncond


def synthetic_logits(ids, step):
    # Small exactly represented values; both branches depend on current tokens.
    b, length = ids.shape
    logits = np.zeros((b, length, 19), dtype=np.float32)
    positions = np.arange(length, dtype=np.int32)[:, None]
    vocab = np.arange(19, dtype=np.int32)[None, :]
    for batch in range(b):
        shift = (int(ids[batch].sum()) + step + 2 * int(ids[batch, 0])) % 7
        logits[batch] = (((positions + 3) * (vocab + 1) + shift * vocab) % 31 - 15) * np.float32(0.125)
    return logits


def reference_worker(root, output):
    import math
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    namespace = dict(torch=torch, math=math, F=torch.nn.functional)
    extract(root / "official/models/sampling.py", ("log", "gumbel_noise", "mask_by_random_topk", "cosine_schedule"), namespace)
    extract(root / "official/models/modeling_showo.py", ("t2i_generate", "mmu_generate"), namespace)
    original_multinomial = torch.multinomial
    original_mask = namespace["mask_by_random_topk"]
    result = {}
    for case in CASES:
        recorded = {}
        calls = [0]
        def forward(ids, attention_mask=None):
            step = calls[0]
            recorded["input_%02d" % step] = ids.cpu().numpy().copy()
            calls[0] += 1
            return torch.from_numpy(synthetic_logits(ids.cpu().numpy(), step)).cuda()
        def controlled_sample(probs, count, generator=None):
            step = calls[0] - 1
            recorded["probs_%02d" % step] = probs.cpu().numpy().copy()
            choices = ((np.arange(probs.shape[0]) * 3 + step * 5) % probs.shape[-1]).astype(np.int64)
            recorded["choice_%02d" % step] = choices
            return torch.from_numpy(choices[:, None]).cuda()
        def controlled_gumbel(probs, generator=None):
            step = calls[0] - 1
            uniform = np.random.default_rng(900 + step).uniform(0.03, 0.97, tuple(probs.shape)).astype(np.float32)
            recorded["uniform_%02d" % step] = uniform
            return -namespace["log"](-namespace["log"](torch.from_numpy(uniform).cuda()))
        def recorded_mask(mask_len, probs, temperature=1.0, generator=None):
            step = calls[0] - 1
            recorded["temperature_%02d" % step] = np.array(temperature, dtype=np.float64)
            value = original_mask(mask_len, probs, temperature, generator=generator)
            recorded["mask_%02d" % step] = value.cpu().numpy().copy()
            return value
        namespace["gumbel_noise"] = controlled_gumbel
        namespace["mask_by_random_topk"] = recorded_mask
        torch.multinomial = controlled_sample
        ids, uncond = inputs(case)
        config = types.SimpleNamespace(model=types.SimpleNamespace(showo=types.SimpleNamespace(
            num_vq_tokens=16, num_new_special_tokens=2, llm_vocab_size=8)),
            dataset=types.SimpleNamespace(preprocessing=types.SimpleNamespace(max_seq_length=2)))
        class Model:
            config = types.SimpleNamespace(mask_token_id=18)
            def __call__(self, ids, attention_mask=None):
                return forward(ids, attention_mask)
        steps = 50 if case == "cfg-all" else 6
        with torch.inference_mode():
            final = namespace["t2i_generate"](Model(), torch.from_numpy(ids).long().cuda(),
                uncond_input_ids=None if case == "no-cfg" else torch.from_numpy(uncond).long().cuda(),
                attention_mask=None, timesteps=steps, guidance_scale=0 if case == "no-cfg" else 5, config=config)
        recorded["final"] = final.cpu().numpy().astype(np.int32)
        np.savez_compressed(output / (case + ".npz"), **recorded)
        result[case] = steps
        print("SAMPLER_REFERENCE_CASE_DONE", case, steps, flush=True)
    torch.multinomial = original_multinomial
    # Actual official MMU loop, unique top-1; record every growing mask.
    records = {}
    calls = [0]
    class MMUModel:
        config = types.SimpleNamespace(w_clip_vit=False)
        def __call__(self, ids, input_embeddings=None, attention_mask=None):
            step = calls[0]
            records["input_%02d" % step] = ids.cpu().numpy()
            records["mask_%02d" % step] = attention_mask.reshape(1, 1, ids.shape[1], ids.shape[1]).cpu().numpy()
            calls[0] += 1
            logits = torch.zeros((1, ids.shape[1], 19), device="cuda")
            logits[:, -1, step + 6] = 4.0
            return logits
    ids = torch.tensor([[1, 2, 3]], device="cuda")
    mask = torch.zeros((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    mask[:, :, 0, 1:] = torch.finfo(torch.float32).min
    mask[:, :, 1, 2:] = torch.finfo(torch.float32).min
    with torch.inference_mode():
        tokens = namespace["mmu_generate"](MMUModel(), ids, attention_mask=mask, max_new_tokens=8, top_k=1, eot_token=9)
    records["tokens"] = np.array([int(value.item()) for value in tokens], dtype=np.int32)
    np.savez_compressed(output / "mmu.npz", **records)
    save_json(output / "reference.json", dict(t2i_cases=result, mmu_steps=calls[0]))


def native_worker(root, output):
    import jittor as jt
    import jittor_sampling as sampler

    jt.flags.use_cuda = 1
    jt.flags.amp_level = 0
    jt.set_global_seed(42)
    original_sample = sampler.categorical_sample
    original_gumbel = sampler.gumbel_noise
    original_mask = sampler.mask_by_random_topk
    # One-hot inputs detect accidental second softmax and exercise multinomial.
    probabilities = np.eye(8, dtype=np.float32)[np.arange(256) % 8]
    chosen = original_sample(jt.array(probabilities)).numpy()
    if not np.array_equal(chosen, np.arange(256) % 8):
        raise RuntimeError("Native categorical primitive failed one-hot contract")
    noise = original_gumbel(jt.zeros((64, 16))).numpy()
    if not np.isfinite(noise).all() or float(noise.std()) == 0:
        raise RuntimeError("Native Gumbel primitive returned invalid noise")
    rows = []
    for case in CASES:
        calls = [0]
        errors = []
        with np.load(output / (case + ".npz"), allow_pickle=False) as archive:
            data = {key: archive[key].copy() for key in archive.files}
        def forward(ids, attention_mask):
            step = calls[0]
            raw = ids.numpy().copy()
            if not np.array_equal(raw, data["input_%02d" % step]):
                errors.append("model_input_%02d" % step)
            calls[0] += 1
            return jt.array(synthetic_logits(raw, step))
        def controlled_sample(probs):
            step = calls[0] - 1
            actual = probs.numpy()
            if not np.allclose(actual, data["probs_%02d" % step], atol=1e-6, rtol=1e-6):
                errors.append("probabilities_%02d" % step)
            return jt.array(data["choice_%02d" % step].astype(np.int32))
        def controlled_gumbel(probs):
            uniform = jt.array(data["uniform_%02d" % (calls[0] - 1)])
            return -sampler._log(-sampler._log(uniform))
        def recorded_mask(mask_len, probs, temperature=1.0):
            step = calls[0] - 1
            if abs(temperature - float(data["temperature_%02d" % step])) > 1e-12:
                errors.append("temperature_%02d" % step)
            value = original_mask(mask_len, probs, temperature)
            if not np.array_equal(value.numpy(), data["mask_%02d" % step]):
                errors.append("remask_%02d" % step)
            return value
        sampler.categorical_sample = controlled_sample
        sampler.gumbel_noise = controlled_gumbel
        sampler.mask_by_random_topk = recorded_mask
        ids, uncond = inputs(case)
        steps = 50 if case == "cfg-all" else 6
        with jt.no_grad():
            final, history, masks = sampler.t2i_generate(forward, jt.array(ids),
                None if case == "no-cfg" else jt.array(uncond), None, num_vq_tokens=16,
                llm_vocab_size=8, num_new_special_tokens=2, mask_token_id=18, max_seq_length=2,
                timesteps=steps, guidance_scale=0 if case == "no-cfg" else 5)
        if not np.array_equal(final, data["final"]) or final.min() < 0 or final.max() >= 8:
            errors.append("final_tokens")
        row = dict(case=case, passed=not errors, forward_calls=calls[0], errors=errors)
        rows.append(row)
        print("SAMPLER_NATIVE_CASE_DONE", json.dumps(row), flush=True)
    sampler.categorical_sample = original_sample
    sampler.gumbel_noise = original_gumbel
    sampler.mask_by_random_topk = original_mask
    with np.load(output / "mmu.npz", allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    calls = [0]
    errors = []
    def forward_mmu(ids, mask):
        step = calls[0]
        if not np.array_equal(ids.numpy(), data["input_%02d" % step]):
            errors.append("input_%02d" % step)
        if not np.array_equal(mask.numpy(), data["mask_%02d" % step]):
            errors.append("mask_%02d" % step)
        calls[0] += 1
        logits = np.zeros((1, ids.shape[1], 19), dtype=np.float32)
        logits[0, -1, step + 6] = 4.0
        return jt.array(logits)
    with jt.no_grad():
        tokens, margins = sampler.mmu_generate(forward_mmu, jt.array(data["input_00"].astype(np.int32)),
            jt.array(data["mask_00"]), max_new_tokens=8, eot_token=9)
    if not np.array_equal(tokens, data["tokens"]):
        errors.append("tokens_or_eos")
    rows.append(dict(case="mmu", passed=not errors, errors=errors, forward_calls=calls[0]))
    report = dict(passed=all(row["passed"] for row in rows), cases=rows,
                  categorical_one_hot=True, gumbel_finite=True,
                  scope="Controlled choices/uniforms on synthetic logits; no weights, full-model validation, cross-framework seed equivalence or image-quality claim")
    save_json(output / "sampler_contract_report.json", report)
    print("NATIVE_SAMPLER_CONTRACT_COMPLETE", output, "passed", report["passed"], flush=True)
    if not report["passed"]:
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=1, choices=(0, 1, 2))
    parser.add_argument("--worker", choices=("reference", "native"), help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path.home() / "Show-o-Jittor"
    if args.worker:
        if args.output is None:
            parser.error("worker requires output")
        {"reference": reference_worker, "native": native_worker}[args.worker](root, args.output)
        return
    commit = subprocess.check_output(["git", "-C", str(root / "official"), "rev-parse", "HEAD"], text=True).strip()
    if commit != COMMIT:
        raise RuntimeError("Official source revision differs")
    subprocess.run(["git", "-C", str(root / "official"), "diff", "--quiet", "HEAD", "--",
                    "models/modeling_showo.py", "models/sampling.py"], check=True)
    used = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=memory.used",
                                    "--format=csv,noheader,nounits"], text=True, timeout=10).strip()
    if int(used) > 1024:
        raise RuntimeError("GPU occupied; stop without terminating any jobs")
    output = root / "results" / ("native-sampler-contract-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).resolve()
    for path in (script, script.with_name("jittor_sampling.py")):
        shutil.copyfile(path, output / path.name)
    source_files = [script, script.with_name("jittor_sampling.py"), root / "official/models/modeling_showo.py", root / "official/models/sampling.py"]
    save_json(output / "run_config.json", dict(official_commit=COMMIT, gpu=args.gpu,
        source_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}))
    print("RUN_DIR", output, flush=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), NVIDIA_TF32_OVERRIDE="0", PYTHONUNBUFFERED="1")
    for worker in ("reference", "native"):
        cmd = [sys.executable, "-u", str(script), "--worker", worker, "--output", str(output)]
        with (output / (worker + ".log")).open("x", encoding="utf-8") as log:
            with subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                status = process.wait()
        if status != 0:
            save_json(output / "run_status.json", dict(status="failed", worker=worker, returncode=status))
            raise SystemExit(status)
    save_json(output / "run_status.json", dict(status="completed", numerical_acceptance="passed"))


if __name__ == "__main__":
    main()
