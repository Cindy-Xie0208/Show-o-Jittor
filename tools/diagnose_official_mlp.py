"""Server-only, identical-operator-input diagnosis of official Show-o layer 23."""

import argparse
from datetime import datetime
import importlib.util
import inspect
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys


PREFIX = "model.layers.23"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def reference_worker(args, helpers):
    import numpy as np
    import torch
    from safetensors import safe_open
    from transformers import PhiConfig

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    root = Path.home() / "Show-o-Jittor"
    phi = load_module("operator_official_phi", root / "official/models/phi.py")
    config = PhiConfig.from_dict(json.loads((args.reference / "phi_config.json").read_text()))
    config._attn_implementation = "eager"
    with torch.device("cuda"):
        layer = phi.PhiDecoderLayer(config, layer_idx=23)
    with safe_open(str(root / "weights/show-o-512x512/pytorch_model.safetensors"), framework="pt", device="cpu") as archive:
        state = {key.removeprefix("showo." + PREFIX + "."): archive.get_tensor(key)
                 for key in archive.keys() if key.startswith("showo." + PREFIX + ".")}
    layer.load_state_dict(state, strict=True)
    layer.eval().requires_grad_(False)
    del state
    controls = []
    for case in json.loads((args.reference / "cases.json").read_text()):
        name = case["name"]
        saved = args.reference / "reference" / name
        output = args.output / "operator_reference" / name
        output.mkdir(parents=True)
        old = args.diagnostic / "diagnostic_reference" / name
        hidden = torch.from_numpy(np.load(saved / "model.layers.22.npy")).cuda()
        with np.load(saved / "input.npz", allow_pickle=False) as fixture:
            mask = torch.from_numpy(fixture["attention_mask"]).cuda()
        captured = {}

        def capture(key):
            def hook(module, inputs, result):
                value = result[0] if isinstance(result, tuple) else result
                captured[key] = value.detach().cpu().numpy().copy()
            return hook

        def activation_input(module, inputs):
            captured["activation"] = inputs[0].detach().cpu().numpy().copy()

        handles = [layer.input_layernorm.register_forward_hook(capture("norm")),
                   layer.self_attn.register_forward_hook(capture("attention")),
                   layer.mlp.fc1.register_forward_hook(capture("fc1")),
                   layer.mlp.fc2.register_forward_pre_hook(activation_input),
                   layer.mlp.register_forward_hook(capture("mlp"))]
        try:
            with torch.inference_mode():
                positions = torch.arange(hidden.shape[1], device="cuda")[None]
                result = layer(hidden, attention_mask=mask, position_ids=positions, use_cache=False)[0]
                captured["block"] = result.cpu().numpy()
        finally:
            for handle in handles:
                handle.remove()
        for key, path in {
            "norm": old / (PREFIX + ".input_layernorm.npy"),
            "attention": old / (PREFIX + ".self_attn.npy"),
            "mlp": old / (PREFIX + ".mlp.npy"),
            "block": saved / (PREFIX + ".npy"),
        }.items():
            expected = np.load(path, mmap_mode="r", allow_pickle=False)
            row = helpers.diagnostic_comparison(np, captured[key], expected, name, key + ".torch_control")
            row["bitwise_equal"] = bool(np.array_equal(captured[key], expected))
            controls.append(row)
            print("OPERATOR_TORCH_CONTROL", json.dumps(row), flush=True)
            if not row["bitwise_equal"]:
                raise RuntimeError("PyTorch replay changed; preserve results and inspect before native comparison")
        for key in ("fc1", "activation"):
            assert np.isfinite(captured[key]).all() and captured[key].dtype == np.float32
            np.save(output / (key + ".npy"), captured[key], allow_pickle=False)
        del hidden, mask, result, positions, captured
    helpers.write_json(args.output / "operator_reference_status.json", dict(
        status="passed", controls=controls,
        sources=dict(torch_version=torch.__version__, activation=inspect.getsource(type(layer.mlp.activation_fn)))))
    print("OPERATOR_REFERENCE_READY", flush=True)


def native_worker(args, helpers):
    import numpy as np
    import jittor as jt
    from safetensors import safe_open

    helpers.check_torch_cuda("operator dependency imports")
    jt.flags.use_cuda = 1
    jt.flags.amp_level = 0
    jt.set_global_seed(42)
    operator_run = args.reuse_operators or args.output
    assert json.loads((operator_run / "operator_reference_status.json").read_text())["status"] == "passed"
    port = load_module("operator_native_phi", args.diagnostic / "native_phi.py")
    config = json.loads((args.reference / "phi_config.json").read_text())
    # Only the unchanged block is exercised; unused embedding/head stay minimal.
    config.update(num_hidden_layers=1, vocab_size=1, pad_token_id=0)
    scaffold = port.model_class(jt, np)(config)
    layer = scaffold.model.layers[0]
    layer.eval()
    state = layer.state_dict()
    weight = Path.home() / "Show-o-Jittor/weights/show-o-512x512/pytorch_model.safetensors"
    with safe_open(str(weight), framework="np") as archive:
        mapping = {key.removeprefix("showo." + PREFIX + "."): key
                   for key in archive.keys() if key.startswith("showo." + PREFIX + ".")}
        assert set(mapping) == set(state), "Native block parameter keys differ"
        for key, parameter in state.items():
            array = archive.get_tensor(mapping[key])
            assert array.dtype == np.float32 and tuple(array.shape) == tuple(parameter.shape)
            parameter.assign(jt.array(array)).stop_grad()
            assert np.array_equal(parameter.numpy(), array), key
    print("LAST_BLOCK_WEIGHTS_LOADED_EXACTLY", len(state), flush=True)

    def centered_layernorm(module, x):
        dims = [-i for i in range(len(module.normalized_shape), 0, -1)]
        mean = jt.mean(x, dims=dims, keepdims=1)
        centered = x - mean
        variance = jt.mean(centered * centered, dims=dims, keepdims=1)
        normalized = centered / jt.sqrt(variance + module.eps)
        return normalized * module.weight + module.bias

    def flat_linear(module, x):
        shape = tuple(x.shape)
        flat = x.reshape((-1, shape[-1]))
        output = jt.matmul(flat, module.weight.transpose((1, 0)))
        if module.bias is not None:
            output = output + module.bias
        return output.reshape(shape[:-1] + (module.out_features,))

    # Overrides live only in this isolated worker; saved model sources stay intact.
    if args.variant in ("centered_layernorm", "centered_layernorm_flat_linear"):
        type(layer.input_layernorm).execute = centered_layernorm
    if args.variant in ("flat_linear", "centered_layernorm_flat_linear"):
        type(layer.mlp.fc1).execute = flat_linear

    def source_of(value):
        try:
            return inspect.getsource(value)
        except (TypeError, OSError) as error:
            return repr(error)

    helpers.write_json(args.output / "operator_sources.json", dict(
        layernorm=inspect.getsource(type(layer.input_layernorm)),
        linear=inspect.getsource(type(layer.mlp.fc1)),
        native_mlp=inspect.getsource(type(layer.mlp)),
        effective_layernorm=source_of(type(layer.input_layernorm).execute),
        effective_linear=source_of(type(layer.mlp.fc1).execute),
        matmul_transpose=source_of(jt.nn.matmul_transpose), matmul=source_of(jt.matmul),
        variant=args.variant,
        jittor_version=jt.__version__))
    print("OPERATOR_VARIANT", args.variant, "FP32", flush=True)
    rows = []
    with (args.output / "operators.jsonl").open("x", encoding="utf-8") as stream:
        for case in json.loads((args.reference / "cases.json").read_text()):
            name = case["name"]
            old = args.diagnostic / "diagnostic_reference" / name
            saved = args.reference / "reference" / name
            new = operator_run / "operator_reference" / name

            def tensor(path):
                return jt.array(np.load(path, allow_pickle=False)).stop_grad()

            def record(value, expected_path, key, input_source):
                actual = value.numpy()
                assert actual.dtype == np.float32, "Candidate changed activation dtype: " + key
                expected = np.load(expected_path, mmap_mode="r", allow_pickle=False)
                row = helpers.diagnostic_comparison(np, actual, expected, name, key)
                row.update(bitwise_equal=bool(np.array_equal(actual, expected)), input_source=input_source)
                rows.append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print("OPERATOR_COMPARE", json.dumps(row), flush=True)

            norm_path = old / (PREFIX + ".input_layernorm.npy")
            mlp_path = old / (PREFIX + ".mlp.npy")
            attention_path = old / (PREFIX + ".self_attn.npy")
            with jt.no_grad():
                hidden = tensor(saved / "model.layers.22.npy")
                normalized = layer.input_layernorm(hidden)
                record(normalized, norm_path, "layernorm.isolated", "exact Torch block input")
                del normalized
                normalized = tensor(norm_path)
                projected = layer.mlp.fc1(normalized)
                record(projected, new / "fc1.npy", "fc1.isolated", "exact Torch LayerNorm output")
                del projected
                projected = tensor(new / "fc1.npy")
                activated = 0.5 * projected * (1.0 + jt.tanh(
                    math.sqrt(2.0 / math.pi) * (projected + 0.044715 * projected ** 3)))
                record(activated, new / "activation.npy", "gelu_new.isolated", "exact Torch FC1 output")
                del activated, projected
                activated = tensor(new / "activation.npy")
                output = layer.mlp.fc2(activated)
                record(output, mlp_path, "fc2.isolated", "exact Torch activation output")
                del activated, output
                output = layer.mlp(normalized)
                record(output, mlp_path, "mlp.same_norm_input", "exact Torch LayerNorm output")
                del output
                output = layer.mlp(layer.input_layernorm(hidden))
                record(output, mlp_path, "norm_mlp.same_block_input", "exact Torch block input")
                del output
                with np.load(saved / "input.npz", allow_pickle=False) as fixture:
                    mask = jt.array(fixture["attention_mask"].astype(np.float32)).stop_grad()
                trace = {}
                output = layer.self_attn(normalized, mask, trace, "attention")
                record(output, attention_path, "attention.same_norm_input", "exact Torch LayerNorm output")
                del output, trace, normalized
                trace = {}
                output = layer(hidden, mask, trace, PREFIX)
                record(output, saved / (PREFIX + ".npy"), "block.same_input", "exact Torch block input")
                del output, trace, hidden, mask
            jt.sync_all(True)
            jt.gc()
            helpers.check_torch_cuda("operator case " + name)
    summary = dict(status="diagnostic_completed", alignment_status="not_reclassified",
                   variant=args.variant, operator_reference_run=str(operator_run),
                   checked=len(rows), outside_tolerance=sum(not row["passed"] for row in rows),
                   tolerances=dict(atol=helpers.ATOL, rtol=helpers.RTOL), layer=23,
                   diagnostic_run=str(args.diagnostic), reference_run=str(args.reference),
                   comparisons=rows, torch_cuda=helpers.check_torch_cuda("operator completion"),
                   scope="Isolated LN/FC1/GELU/FC2 and composed branches on original fixture shapes; "
                         "native block unchanged, FP32, no full-model pass/decoding/training claim")
    helpers.write_json(args.output / "operators.json", summary)
    print("OPERATOR_DIAGNOSTIC_SUMMARY", json.dumps(dict(
        variant=args.variant,
        checked=summary["checked"], outside_tolerance=summary["outside_tolerance"],
        rows=[dict(case=row["case"], name=row["name"], max_abs=row["max_abs"],
                   passed=row["passed"], bitwise_equal=row["bitwise_equal"], worst=row["worst"]) for row in rows])), flush=True)


def noise_reference_worker(args, helpers):
    """Measure CUDA FP32, CUDA FP64 and CPU FP32 on identical layer-23 inputs."""
    import numpy as np
    import torch
    from safetensors import safe_open
    from transformers import PhiConfig

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    root = Path.home() / "Show-o-Jittor"
    phi = load_module("noise_official_phi", root / "official/models/phi.py")
    config = PhiConfig.from_dict(json.loads((args.reference / "phi_config.json").read_text()))
    config._attn_implementation = "eager"
    with torch.device("cuda"):
        cuda_layer = phi.PhiDecoderLayer(config, layer_idx=23)
    with safe_open(str(root / "weights/show-o-512x512/pytorch_model.safetensors"), framework="pt", device="cpu") as archive:
        state = {key.removeprefix("showo." + PREFIX + "."): archive.get_tensor(key)
                 for key in archive.keys() if key.startswith("showo." + PREFIX + ".")}
    cuda_layer.load_state_dict(state, strict=True)
    cuda_layer.eval().requires_grad_(False)
    cpu_layer = phi.PhiDecoderLayer(config, layer_idx=23)
    cpu_layer.load_state_dict(state, strict=True)
    cpu_layer.eval().requires_grad_(False)
    del state

    rows = []
    for case in json.loads((args.reference / "cases.json").read_text()):
        name = case["name"]
        saved = args.reference / "reference" / name
        target = args.output / "torch" / name
        target.mkdir(parents=True)
        hidden_np = np.load(saved / "model.layers.22.npy", allow_pickle=False)
        with np.load(saved / "input.npz", allow_pickle=False) as fixture:
            mask_np = fixture["attention_mask"]
        positions_cuda = torch.arange(hidden_np.shape[1], device="cuda")[None]

        with torch.inference_mode():
            hidden_cuda = torch.from_numpy(hidden_np).cuda()
            mask_cuda = torch.from_numpy(mask_np).cuda()
            torch.cuda.synchronize()
            cuda_output = cuda_layer(hidden_cuda, attention_mask=mask_cuda,
                                     position_ids=positions_cuda, use_cache=False)[0]
            torch.cuda.synchronize()
            cuda_np = cuda_output.float().cpu().numpy()

            fp64_layer = phi.PhiDecoderLayer(config, layer_idx=23).double().cuda()
            fp64_layer.load_state_dict(cuda_layer.state_dict(), strict=True)
            fp64_output = fp64_layer(hidden_cuda.double(), attention_mask=mask_cuda.double(),
                                     position_ids=positions_cuda, use_cache=False)[0]
            torch.cuda.synchronize()
            fp64_np = fp64_output.double().cpu().numpy()
            del fp64_layer, fp64_output

            hidden_cpu = torch.from_numpy(hidden_np)
            mask_cpu = torch.from_numpy(mask_np)
            cpu_output = cpu_layer(hidden_cpu, attention_mask=mask_cpu,
                                   position_ids=torch.arange(hidden_np.shape[1])[None],
                                   use_cache=False)[0]
            cpu_np = cpu_output.float().numpy()

        np.save(target / "cuda_fp32.npy", cuda_np, allow_pickle=False)
        np.save(target / "cuda_fp64.npy", fp64_np, allow_pickle=False)
        np.save(target / "cpu_fp32.npy", cpu_np, allow_pickle=False)
        for label, actual in (("cuda_fp32_vs_fp64", cuda_np),
                              ("cpu_fp32_vs_fp64", cpu_np)):
            row = helpers.diagnostic_comparison(np, actual, fp64_np, name, label)
            rows.append(row)
            print("NOISE_TORCH_COMPARE", json.dumps(row), flush=True)
        coordinate = (0, 283, 293)
        if all(index < size for index, size in zip(coordinate, cuda_np.shape)):
            rows.append(dict(case=name, name="critical_coordinate", index=list(coordinate),
                             cuda_fp32=float(cuda_np[coordinate]),
                             cuda_fp64=float(fp64_np[coordinate]),
                             cpu_fp32=float(cpu_np[coordinate])))
        del hidden_cuda, mask_cuda, cuda_output, hidden_cpu, mask_cpu, cpu_output
    helpers.write_json(args.output / "torch_noise.json", dict(
        status="passed", cases=rows, tolerances=dict(atol=helpers.ATOL, rtol=helpers.RTOL),
        scope="Same layer-23 inputs and weights; CUDA FP32 versus CUDA FP64 and CPU FP32; no full-model reclassification"))
    print("TORCH_FP32_NOISE_REFERENCE_READY", flush=True)


def noise_native_worker(args, helpers):
    """Run the unchanged native Jittor layer-23 block on the same fixtures."""
    import numpy as np
    import jittor as jt
    from safetensors import safe_open

    helpers.check_torch_cuda("noise native dependency imports")
    jt.flags.use_cuda = 1
    jt.flags.amp_level = 0
    jt.set_global_seed(42)
    port = load_module("noise_native_phi", args.diagnostic / "native_phi.py")
    config = json.loads((args.reference / "phi_config.json").read_text())
    config.update(num_hidden_layers=1, vocab_size=1, pad_token_id=0)
    model = port.model_class(jt, np)(config)
    layer = model.model.layers[0]
    layer.eval()
    state = layer.state_dict()
    weight = Path.home() / "Show-o-Jittor/weights/show-o-512x512/pytorch_model.safetensors"
    with safe_open(str(weight), framework="np") as archive:
        mapping = {key.removeprefix("showo." + PREFIX + "."): key
                   for key in archive.keys() if key.startswith("showo." + PREFIX + ".")}
        assert set(mapping) == set(state)
        for key, parameter in state.items():
            array = archive.get_tensor(mapping[key])
            parameter.assign(jt.array(array)).stop_grad()
            assert np.array_equal(parameter.numpy(), array), key
    rows = []
    for case in json.loads((args.reference / "cases.json").read_text()):
        name = case["name"]
        saved = args.reference / "reference" / name
        target = args.output / "jittor" / name
        target.mkdir(parents=True)
        hidden = jt.array(np.load(saved / "model.layers.22.npy", allow_pickle=False)).stop_grad()
        with np.load(saved / "input.npz", allow_pickle=False) as fixture:
            mask = jt.array(fixture["attention_mask"].astype(np.float32)).stop_grad()
        trace = {}
        with jt.no_grad():
            output = layer(hidden, mask, trace, PREFIX)
        actual = output.numpy()
        np.save(target / "jittor_fp32.npy", actual, allow_pickle=False)
        torch_fp32 = np.load(args.output / "torch" / name / "cuda_fp32.npy", mmap_mode="r", allow_pickle=False)
        torch_fp64 = np.load(args.output / "torch" / name / "cuda_fp64.npy", mmap_mode="r", allow_pickle=False)
        row_a = helpers.diagnostic_comparison(np, actual, torch_fp32, name, "jittor_vs_cuda_fp32")
        row_b = helpers.diagnostic_comparison(np, actual, torch_fp64, name, "jittor_vs_cuda_fp64")
        rows.extend((row_a, row_b))
        print("NOISE_JITTOR_COMPARE", json.dumps(row_a), flush=True)
        print("NOISE_JITTOR_FP64_COMPARE", json.dumps(row_b), flush=True)
        del hidden, mask, trace, output
        jt.sync_all(True)
        jt.gc()
    helpers.write_json(args.output / "jittor_noise.json", dict(
        status="passed", cases=rows, tolerances=dict(atol=helpers.ATOL, rtol=helpers.RTOL),
        scope="Native Jittor layer-23 output against the saved CUDA FP32/FP64 baselines; no full-model reclassification"))
    print("JITTOR_FP32_NOISE_READY", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--reuse-operators", type=Path,
                        help="Reuse completed operator fixtures and test local FP32 implementation candidates")
    parser.add_argument("--noise-run", action="store_true",
                        help="Measure Torch CUDA FP32/FP64/CPU FP32 and native Jittor noise on layer 23")
    parser.add_argument("--variant", choices=("native", "centered_layernorm", "flat_linear",
                                              "centered_layernorm_flat_linear"), default="native",
                        help=argparse.SUPPRESS)
    parser.add_argument("--worker", choices=("reference", "native", "noise_reference", "noise_native"), help=argparse.SUPPRESS)
    parser.add_argument("--reference", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path.home() / "Show-o-Jittor"
    if Path(sys.prefix).resolve() != (Path.home() / "envs/showo-jittor").resolve():
        raise RuntimeError("Run in the configured showo-jittor environment")
    args.diagnostic = args.diagnostic.resolve(strict=True)
    helpers = load_module("operator_helpers", args.diagnostic / "alignment_script.py")
    if args.worker:
        {"reference": reference_worker, "native": native_worker,
         "noise_reference": noise_reference_worker,
         "noise_native": noise_native_worker}[args.worker](args, helpers)
        return
    diagnostic = json.loads((args.diagnostic / "local_replay.json").read_text())
    config = json.loads((args.diagnostic / "run_config.json").read_text())
    assert diagnostic["status"] == "diagnostic_completed"
    assert diagnostic["tolerances"] == dict(atol=helpers.ATOL, rtol=helpers.RTOL)
    assert config["checkpoint_sha256"] == helpers.SHA256 and config["commit"] == helpers.COMMIT
    assert helpers.file_hash(args.diagnostic / "alignment_script.py") == config["runner_sha256"]
    assert helpers.file_hash(args.diagnostic / "native_phi.py") == config["native_source_sha256"]
    args.reference = Path(config["reference_run"]).resolve(strict=True)
    reference = json.loads((args.reference / "reference_status.json").read_text())
    assert reference["status"] == "passed" and reference["checkpoint_sha256"] == helpers.SHA256
    original = json.loads((args.reference / "run_config.json").read_text())
    assert config["native_source_sha256"] == original["native_source_sha256"]
    if args.reuse_operators:
        args.reuse_operators = args.reuse_operators.resolve(strict=True)
        saved = json.loads((args.reuse_operators / "run_config.json").read_text())
        status = json.loads((args.reuse_operators / "operator_reference_status.json").read_text())
        assert status["status"] == "passed" and all(row["bitwise_equal"] for row in status["controls"])
        assert Path(saved["diagnostic"]).resolve(strict=True) == args.diagnostic
        assert Path(saved["reference"]).resolve(strict=True) == args.reference
        assert saved["checkpoint_sha256"] == helpers.SHA256
        assert saved["native_sha256"] == config["native_source_sha256"]
        assert helpers.file_hash(args.reuse_operators / "operator_script.py") == saved["runner_sha256"]
        for case in json.loads((args.reference / "cases.json").read_text()):
            for key in ("fc1", "activation"):
                path = args.reuse_operators / "operator_reference" / case["name"] / (key + ".npy")
                if not path.is_file():
                    raise FileNotFoundError(str(path))
    helpers.check_gpu(args.gpu)
    assert subprocess.check_output(["git", "-C", str(root / "official"), "rev-parse", "HEAD"], text=True).strip() == helpers.COMMIT
    subprocess.run(["git", "-C", str(root / "official"), "diff", "--quiet", "HEAD", "--", "models/phi.py"], check=True)
    print("VERIFYING_EXISTING_WEIGHT_SHA256", flush=True)
    assert helpers.file_hash(root / "weights/show-o-512x512/pytorch_model.safetensors") == helpers.SHA256
    if shutil.disk_usage(root).free < 2 * 1024 ** 3:
        raise RuntimeError("Need 2 GiB free for new operator fixtures")
    prefix = ("official512-noise-" if args.noise_run else
              "official512-operator-candidates-" if args.reuse_operators else
              "official512-operators-")
    args.output = root / "results" / (prefix + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    args.output.mkdir(parents=True, exist_ok=False)
    source = globals().get("showo_source")
    if source is None:
        source = Path(__file__).read_text(encoding="utf-8")
    script = args.output / "operator_script.py"
    compile(source, str(script), "exec")
    with script.open("x", encoding="utf-8") as stream:
        stream.write(source)
    helpers.write_json(args.output / "run_config.json", dict(
        gpu=args.gpu, diagnostic=str(args.diagnostic), reference=str(args.reference),
        reuse_operators=str(args.reuse_operators) if args.reuse_operators else None,
        checkpoint_sha256=helpers.SHA256, commit=helpers.COMMIT,
        runner_sha256=helpers.file_hash(script), helper_sha256=config["runner_sha256"],
        native_sha256=config["native_source_sha256"], atol=helpers.ATOL, rtol=helpers.RTOL))
    print("RUN_DIR", args.output, flush=True)
    base = [sys.executable, "-u", str(script), "--gpu", str(args.gpu),
            "--diagnostic", str(args.diagnostic), "--reference", str(args.reference)]
    try:
        if args.noise_run:
            helpers.check_gpu(args.gpu)
            helpers.run_stage(base + ["--worker", "noise_reference", "--output", str(args.output)],
                              helpers.environment(args.gpu, False), args.output / "noise_reference.log")
            helpers.check_gpu(args.gpu)
            helpers.run_stage(base + ["--worker", "noise_native", "--output", str(args.output)],
                              helpers.environment(args.gpu, True), args.output / "noise_native.log")
            helpers.write_json(args.output / "noise_summary.json", dict(
                status="completed", torch=str(args.output / "torch_noise.json"),
                jittor=str(args.output / "jittor_noise.json"),
                scope="Supplementary FP32 baseline-noise experiment; original 83-item result unchanged"))
            print("FP32_NOISE_EXPERIMENT_COMPLETE", args.output, flush=True)
        elif args.reuse_operators:
            candidates = []
            for variant in ("centered_layernorm", "flat_linear", "centered_layernorm_flat_linear"):
                directory = args.output / variant
                directory.mkdir()
                helpers.check_gpu(args.gpu)
                command = base + ["--worker", "native", "--reuse-operators", str(args.reuse_operators),
                                  "--variant", variant, "--output", str(directory)]
                helpers.run_stage(command, helpers.environment(args.gpu, True), directory / "native.log")
                report = json.loads((directory / "operators.json").read_text())
                candidates.append(dict(variant=variant, checked=report["checked"],
                                       outside_tolerance=report["outside_tolerance"],
                                       block_rows=[row for row in report["comparisons"] if row["name"] == "block.same_input"]))
            summary = dict(status="local_candidates_completed", alignment_status="not_reclassified",
                           variants=candidates, operator_reference_run=str(args.reuse_operators),
                           local_passed_variants=[row["variant"] for row in candidates if row["outside_tolerance"] == 0],
                           tolerances=dict(atol=helpers.ATOL, rtol=helpers.RTOL),
                           scope="Layer 23 only; runtime FP32 candidates, not full-model acceptance")
            helpers.write_json(args.output / "candidate_summary.json", summary)
            print("OPERATOR_CANDIDATE_SUMMARY", json.dumps(summary, indent=2), flush=True)
        else:
            for stage in ("reference", "native"):
                helpers.check_gpu(args.gpu)
                helpers.run_stage(base + ["--worker", stage, "--output", str(args.output)],
                                  helpers.environment(args.gpu, stage == "native"),
                                  args.output / (stage + ".log"))
    except BaseException as error:
        helpers.write_json(args.output / "run_status.json", dict(status="failed", error=repr(error)))
        raise
    helpers.write_json(args.output / "run_status.json", dict(status="diagnostic_completed", alignment_status="not_reclassified"))
    marker = "OPERATOR_CANDIDATES_COMPLETE" if args.reuse_operators else "OPERATOR_DIAGNOSTIC_COMPLETE"
    print(marker, args.output, flush=True)


if __name__ == "__main__":
    main()
