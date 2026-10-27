"""Server-only multi-sample Show-o 1 PyTorch/Jittor consistency evaluation.

The parent process owns the official PyTorch model and the child process owns
the native Jittor model.  Each process loads the 512 checkpoint once.  The
parent drives identical sampling cases through both models and records final
token agreement, first trajectory divergence, and the reference top-1 margin.
No 83-item numerical alignment is rerun here; the existing 78/83 result is
referenced in the report.
"""

from pathlib import Path
from datetime import datetime
from multiprocessing.connection import Client, Listener
import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import types


DEFAULT_PROMPTS = [
    "A photo of a golden retriever sitting in a field of yellow flowers.",
    "A red bicycle leaning against a brick wall on a sunny street.",
    "A small sailboat on a calm blue lake at sunset.",
    "A cozy cabin in a snowy forest under a starry sky.",
]
DEFAULT_SEEDS = [42, 43]
DEFAULT_QUESTION = "Describe this image in detail."


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def native_worker(args):
    import numpy as np
    import jittor as jt
    from safetensors import safe_open

    root = Path.home() / "Show-o-Jittor"
    jt.flags.use_cuda = 1
    jt.flags.amp_level = 0
    jt.set_global_seed(42)
    reference = root / "results/official512-jittor-align-20261027-004837-924843"
    port = load_module("multi_native_phi", reference / "native_phi.py")
    config = json.loads((reference / "phi_config.json").read_text())
    model = port.model_class(jt, np)(config)
    model.eval()
    state = model.state_dict()
    weight = root / "weights/show-o-512x512/pytorch_model.safetensors"
    with safe_open(str(weight), framework="np") as archive:
        mapping = {key.removeprefix("showo."): key for key in archive.keys()}
        assert set(mapping) == set(state)
        for index, (key, parameter) in enumerate(state.items(), 1):
            array = archive.get_tensor(mapping[key])
            assert array.dtype == np.float32 and tuple(array.shape) == tuple(parameter.shape), key
            parameter.assign(jt.array(array)).stop_grad()
            if index == 1 or index % 48 == 0 or index == len(state):
                print("MULTI_JITTOR_LOAD", index, "/", len(state), key, flush=True)
    candidate_source = Path(args.candidate_run, "operator_script.py").read_text(encoding="utf-8")
    tree = ast.parse(candidate_source)
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "flat_linear"]
    assert len(functions) == 1 and not functions[0].decorator_list
    namespace = {"jt": jt}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "verified_flat_linear", "exec"), namespace)
    type(model.lm_head).execute = namespace["flat_linear"]
    print("MULTI_JITTOR_WEIGHTS_LOADED_EXACTLY", flush=True)

    Path(args.ready).write_text(json.dumps({"status": "ready", "bridge": args.bridge}), encoding="utf-8")
    # Wake the parent if it is blocked reading this worker's stdout while
    # polling for ready.json.
    print("MULTI_NATIVE_READY", flush=True)
    calls = 0
    with Listener(args.bridge, family="AF_UNIX") as listener:
        with listener.accept() as connection:
            while True:
                request = connection.recv()
                command = request.get("command")
                if command == "stop":
                    connection.send({"status": "stopped", "calls": calls})
                    break
                if command == "reset":
                    calls = 0
                    connection.send({"status": "reset"})
                    continue
                assert command == "forward"
                raw = request["input_ids"]
                raw_mask = request["attention_mask"]
                assert isinstance(raw, np.ndarray) and np.issubdtype(raw.dtype, np.integer)
                assert raw.min() >= 0 and raw.max() < 58498
                assert isinstance(raw_mask, np.ndarray) and raw_mask.dtype == np.float32
                assert np.isfinite(raw_mask).all()
                ids = jt.array(raw.astype(np.int32)).stop_grad()
                mask = jt.array(raw_mask).stop_grad()
                with jt.no_grad():
                    logits, trace = model(ids, mask)
                    if request["task"] == "t2i":
                        result = logits[:, -1025:-1, 50305:58497].numpy()
                        assert result.shape == (2, 1024, 8192)
                    else:
                        result = logits[:, -1:, :].numpy()
                        assert result.shape == (1, 1, 58498)
                    assert result.dtype == np.float32 and np.isfinite(result).all()
                    connection.send(result)
                    del ids, mask, logits, trace, result
                calls += 1
                jt.sync_all(True)
                jt.gc()
                if calls == 1 or calls % 10 == 0:
                    print("MULTI_JITTOR_FORWARD", request["task"], calls, flush=True)


def make_official_model(root, torch):
    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    package = types.ModuleType("multi_official")
    package.__path__ = [str(root / "official/models")]
    sys.modules[package.__name__] = package
    modeling = importlib.import_module("multi_official.modeling_showo")
    sampling = importlib.import_module("multi_official.sampling")
    Showo = modeling.Showo
    prompting = load_module("multi_prompting", root / "official/training/prompting_utils.py")
    tokenizer = AutoTokenizer.from_pretrained(str(root / "weights/phi-1_5-original"), local_files_only=True)
    config = OmegaConf.load(root / "official/configs/showo_demo_512x512.yaml")
    weight = root / "weights/show-o-512x512/pytorch_model.safetensors"
    with torch.device("cuda"):
        model = Showo(w_clip_vit=False, vocab_size=58498, llm_vocab_size=50295,
                      num_vq_tokens=1024, llm_model_path=str(root / "weights/phi-1_5-original"))
    model.load_state_dict(load_file(str(weight), device="cuda"), strict=True)
    model.eval().requires_grad_(False)
    return model, config, tokenizer, prompting, sampling


def t2i_inputs(args, torch, tokenizer, prompting):
    prompt = prompting.UniversalPrompting(tokenizer, max_text_len=128, cond_dropout_prob=0.0)
    masked = torch.full((1, 1024), 58497, dtype=torch.long, device="cuda")
    ids, _ = prompt(([args.prompt], masked), "t2i_gen")
    uncond, _ = prompt(([""], masked), "t2i_gen")
    mask = prompting.create_attention_mask_predict_next(
        torch.cat([ids, uncond]), pad_id=tokenizer.pad_token_id,
        soi_id=int(prompt.sptids_dict["<|soi|>"].item()),
        eoi_id=int(prompt.sptids_dict["<|eoi|>"].item()), rm_pad_in_image=True).float()
    return ids.cuda(), uncond.cuda(), mask.cuda()


def top2_margin(logits, torch):
    values = torch.topk(logits, 2, dim=-1).values
    return values[..., 0] - values[..., 1]


def sample_t2i(model, config, ids, uncond, mask, torch, sampling, forward, seed):
    num_tokens = 1024
    mask_id = 58497
    generator = torch.Generator(device="cuda").manual_seed(int(seed))
    working = ids.clone()
    image_ids = torch.where(working[:, -(num_tokens + 1):-1] == mask_id, mask_id,
                            working[:, -(num_tokens + 1):-1] - 50305)
    history = []
    margin_history = []
    with torch.inference_mode():
        for step in range(50):
            if forward is None:
                full = model(torch.cat([working, uncond]), attention_mask=mask)
                cond, unconditional = full.chunk(2)
                logits = (1 + 5.0) * cond - 5.0 * unconditional
                logits = logits[:, -(num_tokens + 1):-1, 50305:58497]
            else:
                # The native bridge returns both conditional and unconditional
                # image logits. Apply the same CFG equation as the official
                # PyTorch sampler before sampling the image codebook.
                full = forward(torch.cat([working, uncond]), mask)
                cond, unconditional = full.chunk(2)
                logits = (1 + 5.0) * cond - 5.0 * unconditional
            probs = logits.softmax(dim=-1)
            sampled = torch.multinomial(probs.reshape(-1, logits.size(-1)), 1, generator=generator)
            sampled = sampled[:, 0].view(*logits.shape[:-1])
            unknown = image_ids == mask_id
            sampled = torch.where(unknown, sampled, image_ids)
            margins = top2_margin(logits, torch)
            history.append(sampled[0].detach().cpu().numpy().astype("int32"))
            margin_history.append(margins[0].detach().cpu().numpy().astype("float32"))
            ratio = 1.0 * (step + 1) / 50
            mask_ratio = sampling.cosine_schedule(torch.tensor(ratio, device="cuda"))
            selected = torch.gather(probs, -1, sampled.long()[..., None]).squeeze(-1)
            selected = torch.where(unknown, selected, torch.finfo(selected.dtype).max)
            mask_len = (num_tokens * mask_ratio).floor().unsqueeze(0).to(logits.device)
            mask_len = torch.max(torch.tensor([1], device="cuda"),
                                 torch.min(unknown.sum(dim=-1, keepdim=True) - 1, mask_len))
            temperature = 1.0 * (1.0 - ratio)
            masking = sampling.mask_by_random_topk(mask_len, selected, temperature, generator=generator)
            working[:, -(num_tokens + 1):-1] = torch.where(
                masking, mask_id, sampled + 50305)
            image_ids = torch.where(masking, mask_id, sampled)
    return image_ids.detach().cpu().numpy().astype("int32"), history, margin_history


def compare_trajectories(reference_history, jittor_history, reference_margins):
    first = None
    for step, (ref, jit) in enumerate(zip(reference_history, jittor_history), 1):
        mismatch = np_nonzero(ref != jit)
        if mismatch is not None:
            token = int(mismatch)
            first = dict(step=step, token_index=token,
                         reference_token=int(ref[token]), jittor_token=int(jit[token]),
                         reference_top1_margin=float(reference_margins[step - 1][token]))
            break
    ref_final = reference_history[-1]
    jit_final = jittor_history[-1]
    all_equal = all(np_equal(a, b) for a, b in zip(reference_history, jittor_history))
    return dict(final_token_agreement=float((ref_final == jit_final).mean()),
                trajectory_token_agreement=float(sum((a == b).sum() for a, b in zip(reference_history, jittor_history)) /
                                               (len(reference_history) * ref_final.size)),
                exact_trajectory_match=bool(all_equal),
                first_divergence=(first or None))


def np_equal(a, b):
    import numpy as np
    return np.array_equal(a, b)


def np_nonzero(condition):
    import numpy as np
    indices = np.flatnonzero(condition)
    return None if len(indices) == 0 else int(indices[0])


def build_mmu_input(codes, question, torch, tokenizer, prompting):
    prompt = prompting.UniversalPrompting(tokenizer, max_text_len=128, cond_dropout_prob=0.0)
    special = lambda name: int(prompt.sptids_dict[name].item())
    prefix = [special("<|mmu|>"), special("<|soi|>")] + (codes[0] + len(tokenizer)).tolist()
    prefix += [special("<|eoi|>"), tokenizer.bos_token_id]
    text_ids = tokenizer("USER: \n" + question + " ASSISTANT:")['input_ids']
    ids = torch.tensor([prefix + text_ids], dtype=torch.long, device="cuda")
    mask = prompting.create_attention_mask_for_mmu(ids, eoi_id=special("<|eoi|>")).float()
    return ids, mask, tokenizer.eos_token_id


def expand_mmu_mask(mask, torch):
    current = mask.squeeze()
    length = current.shape[-1]
    row = torch.hstack([current, torch.zeros((length, 1), device="cuda") + torch.finfo(current.dtype).min])
    expanded = torch.vstack([row, torch.hstack([current[-1, :], torch.tensor([0.0], device="cuda")]).unsqueeze(0)])
    return expanded.unsqueeze(0).unsqueeze(0)


def sample_mmu(model, ids, mask, eot, torch, forward, seed):
    generator = torch.Generator(device="cuda").manual_seed(int(seed))
    current_ids = ids.clone()
    current_mask = mask.clone()
    history, margins = [], []
    with torch.inference_mode():
        for _ in range(64):
            if forward is None:
                logits = model(current_ids, attention_mask=current_mask)
            else:
                logits = forward(current_ids, current_mask)
            next_logits = logits[:, -1, :].clone()
            values, indices = torch.topk(next_logits, 1, dim=-1)
            filtered = torch.full_like(next_logits, float("-inf"))
            filtered.scatter_(1, indices, values)
            probs = filtered.softmax(dim=-1)
            next_token = torch.multinomial(probs, 1, generator=generator)
            history.append(int(next_token[0, 0].item()))
            margins.append(float((torch.topk(next_logits, 2, dim=-1).values[0, 0] -
                                 torch.topk(next_logits, 2, dim=-1).values[0, 1]).item()))
            current_ids = torch.cat([current_ids, next_token], dim=1)
            current_mask = expand_mmu_mask(current_mask, torch)
            if int(next_token[0, 0].item()) == int(eot):
                break
    return history, margins


def encode_mmu_images(root, args, torch, helpers):
    from PIL import Image
    from safetensors.torch import load_file
    from torchvision import transforms

    source = (root / "official/training/utils.py").read_text(encoding="utf-8")
    functions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "image_transform"]
    assert len(functions) == 1 and not functions[0].decorator_list
    namespace = {"transforms": transforms}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "official_image_transform", "exec"), namespace)
    magvit_package = types.ModuleType("multi_magvit")
    magvit_package.__path__ = [str(root / "official/models")]
    sys.modules[magvit_package.__name__] = magvit_package
    Magvit = importlib.import_module("multi_magvit.modeling_magvitv2").MAGVITv2
    model = Magvit()
    weight = root / "weights/magvitv2/pytorch_model.safetensors"
    model.load_state_dict(load_file(str(weight)), strict=True)
    model.eval().requires_grad_(False).cuda()
    flickr_candidates = sorted((root / "data").glob("flickr8k-128-32-*/validation"))
    assert flickr_candidates, "Flickr validation subset not found"
    paths = [root / "official/mmu_validation/dog.png", root / "official/mmu_validation/sofa_under_water.jpg",
             flickr_candidates[-1] / "00000.png", flickr_candidates[-1] / "00001.png"]
    result = []
    with torch.inference_mode():
        for path in paths:
            with Image.open(path) as image:
                pixels = namespace["image_transform"](image.convert("RGB"), resolution=512).unsqueeze(0).cuda()
            codes = model.get_code(pixels).cpu().numpy().astype("int32")
            assert codes.shape == (1, 1024) and codes.min() >= 0 and codes.max() < 8192
            result.append(dict(path=str(path), sha256=sha256(path), codes=codes))
    del model
    torch.cuda.empty_cache()
    return result


def main(args):
    import numpy as np
    import torch

    root = Path.home() / "Show-o-Jittor"
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.gpu != 1:
        raise RuntimeError("This evaluation is pinned to GPU 1")
    used = int(subprocess.check_output([
        "nvidia-smi", "-i", str(args.gpu), "--query-gpu=memory.used",
        "--format=csv,noheader,nounits"], text=True).strip())
    if used > 1024:
        raise RuntimeError(f"GPU {args.gpu} uses {used} MiB; stop and report nvidia-smi; do not stop other jobs")
    candidate = json.loads((args.candidate_run / "candidate_summary.json").read_text())
    assert "flat_linear" in candidate["local_passed_variants"]
    output = root / "results" / ("jittor-multi-512-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    (output / "t2i").mkdir(parents=True)
    (output / "mmu").mkdir()
    bridge_dir = Path(tempfile.mkdtemp(prefix="xxt-showo-multi-", dir="/tmp"))
    bridge = bridge_dir / "native.sock"
    ready = bridge_dir / "ready.json"
    native = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", "native",
                               "--gpu", str(args.gpu), "--candidate-run", str(args.candidate_run),
                               "--bridge", str(bridge), "--ready", str(ready)],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
    log_path = output / "jittor.log"
    with log_path.open("x", encoding="utf-8") as log:
        while not ready.is_file():
            if native.poll() is not None:
                raise RuntimeError("Jittor worker exited before ready")
            line = native.stdout.readline()
            if line:
                print(line, end="")
                log.write(line)
            else:
                time.sleep(0.2)
        while not bridge.exists():
            if native.poll() is not None:
                raise RuntimeError("Jittor socket was not created")
            time.sleep(0.1)
        print("MULTI_LOADING_PYTORCH", flush=True)
        model, config, tokenizer, prompting, sampling = make_official_model(root, torch)
        print("MULTI_PYTORCH_READY", flush=True)
        connection = Client(str(bridge), family="AF_UNIX")
        t2i_rows = []
        for prompt_index, text in enumerate(DEFAULT_PROMPTS):
            for seed in DEFAULT_SEEDS:
                case = f"prompt-{prompt_index + 1:02d}-seed-{seed}"
                print("MULTI_T2I_START", case, flush=True)
                local = argparse.Namespace(prompt=text)
                ids, uncond, mask = t2i_inputs(local, torch, tokenizer, prompting)
                connection.send({"command": "reset"})
                assert connection.recv()["status"] == "reset"
                jittor_forward = lambda a, b: torch.from_numpy(connection_forward(connection, a, b, "t2i")).cuda()
                reference_codes, ref_history, ref_margins = sample_t2i(model, config, ids, uncond, mask, torch, sampling, None, seed)
                connection.send({"command": "reset"})
                assert connection.recv()["status"] == "reset"
                jittor_codes, jit_history, _ = sample_t2i(model, config, ids, uncond, mask, torch, sampling, jittor_forward, seed)
                comparison = compare_trajectories(ref_history, jit_history, ref_margins)
                row = dict(case=case, prompt=text, seed=seed, tokens=1024, **comparison)
                t2i_rows.append(row)
                np.savez_compressed(output / "t2i" / (case + ".npz"), reference=reference_codes, jittor=jittor_codes)
                write_json(output / "t2i" / (case + ".json"), row)
                print("MULTI_T2I_CASE", json.dumps(row, ensure_ascii=False), flush=True)
        mmu_images = encode_mmu_images(root, args, torch, None)
        mmu_rows = []
        for index, item in enumerate(mmu_images, 1):
            question = DEFAULT_QUESTION
            print("MULTI_MMU_START", f"image-{index:02d}", flush=True)
            ids, mask, eot = build_mmu_input(item["codes"], question, torch, tokenizer, prompting)
            case = f"image-{index:02d}"
            connection.send({"command": "reset"})
            assert connection.recv()["status"] == "reset"
            reference_tokens, reference_margins = sample_mmu(model, ids, mask, eot, torch, None, 42)
            connection.send({"command": "reset"})
            assert connection.recv()["status"] == "reset"
            jittor_tokens, _ = sample_mmu(model, ids, mask, eot, torch,
                                          lambda a, b: torch.from_numpy(connection_forward(connection, a, b, "mmu")).cuda(), 42)
            first = None
            for step, (ref, jit) in enumerate(zip(reference_tokens, jittor_tokens), 1):
                if ref != jit:
                    first = dict(step=step, reference_token=int(ref), jittor_token=int(jit),
                                 reference_top1_margin=float(reference_margins[step - 1]))
                    break
            row = dict(case=case, image=item["path"], image_sha256=item["sha256"], question=question,
                       reference_tokens=reference_tokens, jittor_tokens=jittor_tokens,
                       token_agreement=float(sum(a == b for a, b in zip(reference_tokens, jittor_tokens)) /
                                            max(len(reference_tokens), len(jittor_tokens))),
                       first_divergence=first, exact_match=reference_tokens == jittor_tokens)
            mmu_rows.append(row)
            write_json(output / "mmu" / (case + ".json"), row)
            print("MULTI_MMU_CASE", json.dumps({k: v for k, v in row.items() if k not in ("reference_tokens", "jittor_tokens")}, ensure_ascii=False), flush=True)
        connection.send({"command": "stop"})
        connection.recv()
        connection.close()
        if native.wait(timeout=120) != 0:
            raise RuntimeError("Jittor worker failed; inspect jittor.log")
    report = dict(status="completed", output=str(output), resolution=512, image_tokens=1024,
                  t2i=t2i_rows, mmu=mmu_rows,
                  existing_alignment=dict(status="failed", passed=78, checked=83, failed=5,
                                          source="official512-full-operator-candidate-20261028-012034-334582"),
                  scope="Multi-sample token consistency; no new 83-item numerical alignment; official PyTorch reference and native Jittor Transformer each loaded once")
    write_json(output / "multi_sample_report.json", report)
    print("MULTI_SAMPLE_REPORT", output / "multi_sample_report.json", flush=True)
    print("MULTI_SAMPLE_EVAL_COMPLETE", output, flush=True)


def connection_forward(connection, ids, mask, task):
    import numpy as np
    connection.send(dict(command="forward", task=task, input_ids=ids.detach().cpu().numpy(),
                         attention_mask=mask.detach().cpu().numpy().astype(np.float32)))
    return connection.recv()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--worker", choices=("native",))
    parser.add_argument("--bridge", type=str)
    parser.add_argument("--ready", type=str)
    args = parser.parse_args()
    if args.worker == "native":
        if not args.bridge or not args.ready:
            parser.error("native worker requires --bridge and --ready")
        return args
    if Path(sys.prefix).resolve() != (Path.home() / "envs/showo-jittor").resolve():
        raise RuntimeError("Run in the configured showo-jittor environment")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.worker == "native":
        native_worker(parsed)
    else:
        main(parsed)
