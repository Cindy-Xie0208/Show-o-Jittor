"""Server-side smoke test for the native Jittor MagViT port.

This deliberately runs before changing the full inference driver.  It checks
the official safetensors key set, decodes the already recorded dog codes, and
optionally encodes the same preprocessed dog image for code agreement.
"""

import argparse
import ast
import hashlib
import importlib
import json
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import sys
import time
import types

import numpy as np
from PIL import Image
from safetensors import safe_open

ATOL = 1e-4
RTOL = 1e-4
OFFICIAL_COMMIT = "45a5a2de01d1ebd10cd5864d29310a76476cdf23"
MAGVIT_SHA256 = "79656bf974d99c215465a61df5e93b97e03282b0941f62f6294719527873b778"


def setup_native():
    global jt, MAGVITv2
    import jittor as jt
    from jittor_magvit import MAGVITv2
    jt.flags.use_cuda = 1
    assert jt.flags.use_cuda == 1


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def preview(array):
    return np.clip((array[0].transpose(1, 2, 0) + 1.0) * 127.5, 0, 255).astype(np.uint8)


def load_exact(model, weight):
    state = model.state_dict()
    with safe_open(str(weight), framework="np") as archive:
        expected = {key: archive.get_tensor(key) for key in archive.keys()}
    if set(state) != set(expected):
        raise RuntimeError("MagViT key mismatch: missing=%s extra=%s" %
                           (sorted(set(state) - set(expected)), sorted(set(expected) - set(state))))
    for key, parameter in state.items():
        value = expected[key]
        if tuple(parameter.shape) != tuple(value.shape):
            raise RuntimeError("MagViT shape mismatch at %s: model=%s checkpoint=%s" %
                               (key, tuple(parameter.shape), tuple(value.shape)))
        if key == "quantize.power_vals":
            # Official int64 buffer: [4096, 2048, ..., 1]. All entries fit
            # exactly in the native int32 index arithmetic; never cast weights.
            powers = np.array([2 ** i for i in range(12, -1, -1)], dtype=np.int64)
            if value.dtype != np.int64 or not np.array_equal(value, powers):
                raise RuntimeError("Unexpected official quantize.power_vals: dtype=%s values=%s" %
                                   (value.dtype, value.tolist()))
            print("MAGVIT_INTEGER_BUFFER", key, "checkpoint=int64 native=int32 exact_values=True", flush=True)
            value = value.astype(np.int32)
        elif value.dtype != np.float32:
            raise RuntimeError("MagViT dtype mismatch at %s: checkpoint=%s expected=float32" %
                               (key, value.dtype))
        parameter.assign(jt.array(value)).stop_grad()
    for key, parameter in state.items():
        if not np.array_equal(parameter.numpy(), expected[key]):
            raise RuntimeError("MagViT exact load failed at %s" % key)
    return len(state), sum(int(np.prod(value.shape)) for value in expected.values())


def official_pixels(path):
    # This is the pinned official 256px MMU preprocessing: resize the shorter
    # side to 256, center crop, convert RGB and map [0,255] to [-1,1].
    with Image.open(path) as image:
        image = image.convert("RGB")
        scale = 256.0 / min(image.size)
        size = (round(image.size[0] * scale), round(image.size[1] * scale))
        image = image.resize(size, Image.Resampling.BICUBIC)
        left = (image.width - 256) // 2
        top = (image.height - 256) // 2
        image = image.crop((left, top, left + 256, top + 256))
        array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return jt.array(array.transpose(2, 0, 1)[None]).stop_grad()


def main():
    setup_native()
    root = Path.home() / "Show-o-Jittor"
    weight = root / "weights/magvitv2/pytorch_model.safetensors"
    model = MAGVITv2()
    model.eval()
    tensors, parameters = load_exact(model, weight)
    print("MAGVIT_JITTOR_WEIGHTS_LOADED_EXACTLY", tensors, parameters, flush=True)

    saved = root / "results/magvit-reconstruction-20261026-203006-738585/image_tokens.npz"
    with np.load(saved, allow_pickle=False) as archive:
        codes = archive["dog"].astype(np.int32)
    codes_jt = jt.array(codes.astype(np.int32)).stop_grad()
    with jt.no_grad():
        decoded = model.decode_code(codes_jt)
    decoded_np = decoded.numpy()
    assert decoded_np.shape == (1, 3, 256, 256) and np.isfinite(decoded_np).all()
    preview = np.clip((decoded_np[0].transpose(1, 2, 0) + 1.0) * 127.5, 0, 255).astype(np.uint8)
    output = root / "results" / ("native-magvit-smoke-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    output.mkdir(parents=True, exist_ok=False)
    Image.fromarray(preview).save(output / "dog_jittor.png")
    (output / "decode.json").write_text(json.dumps({
        "status": "completed", "framework": "jittor", "shape": list(decoded_np.shape),
        "finite": True, "codes_sha256": hashlib.sha256(codes.tobytes()).hexdigest(),
    }, indent=2) + "\n", encoding="utf-8")
    print("MAGVIT_JITTOR_DECODE_OK", output / "dog_jittor.png", flush=True)

    source = root / "official/mmu_validation/dog.png"
    pixels = official_pixels(source)
    with jt.no_grad():
        native_codes = model.get_code(pixels).numpy().astype(np.int32)
    agreement = float(np.mean(native_codes == codes))
    np.savez_compressed(output / "dog_codes.npz", codes=native_codes)
    print("MAGVIT_JITTOR_ENCODE_AGREEMENT", agreement, flush=True)
    (output / "encode.json").write_text(json.dumps({
        "status": "completed", "agreement": agreement, "shape": list(native_codes.shape),
        "source": "official/mmu_validation/dog.png", "preprocess": "256 center crop, bicubic, [-1,1]",
    }, indent=2) + "\n", encoding="utf-8")


def reference_worker(root, output):
    import torch
    from safetensors.torch import load_file
    from torchvision import transforms

    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    package = types.ModuleType("paired_magvit_official")
    package.__path__ = [str(root / "official/models")]
    sys.modules[package.__name__] = package
    cls = importlib.import_module(package.__name__ + ".modeling_magvitv2").MAGVITv2
    model = cls()
    print(model.load_state_dict(load_file(str(root / "weights/magvitv2/pytorch_model.safetensors")), strict=True), flush=True)
    model.eval().requires_grad_(False).cuda()
    source = (root / "official/training/utils.py").read_text(encoding="utf-8")
    functions = [node for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef) and node.name == "image_transform"]
    if len(functions) != 1 or functions[0].decorator_list:
        raise RuntimeError("Cannot isolate pinned official image_transform")
    namespace = {"transforms": transforms}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "official_image_transform", "exec"), namespace)
    cases = []
    for resolution in (256, 512):
        for filename in ("dog.png", "sofa_under_water.jpg"):
            name = Path(filename).stem + "-" + str(resolution)
            print("MAGVIT_REFERENCE_CASE_START", name, flush=True)
            started = time.monotonic()
            image_path = root / "official/mmu_validation" / filename
            with Image.open(image_path) as image:
                pixels = namespace["image_transform"](image.convert("RGB"), resolution=resolution).unsqueeze(0)
            with torch.inference_mode():
                gpu_pixels = pixels.cuda()
                codes = model.get_code(gpu_pixels)
                decoded = model.decode_code(codes).cpu().numpy().copy()
            codes_np = codes.cpu().numpy().astype(np.int32)
            if codes_np.shape != (1, (resolution // 16) ** 2) or codes_np.min() < 0 or codes_np.max() >= 8192:
                raise RuntimeError("Invalid reference image codes: " + name)
            if decoded.shape != (1, 3, resolution, resolution) or not np.isfinite(decoded).all():
                raise RuntimeError("Invalid reference decode: " + name)
            np.savez_compressed(output / (name + "-reference.npz"),
                                pixels=pixels.numpy(), codes=codes_np, decoded=decoded)
            Image.fromarray(preview(decoded)).save(output / (name + "-pytorch.png"))
            cases.append(dict(name=name, resolution=resolution, source="official/mmu_validation/" + filename,
                              source_sha256=file_hash(image_path), seconds=time.monotonic() - started))
            print("MAGVIT_REFERENCE_CASE_DONE", name, flush=True)
    write_json(output / "reference_manifest.json", dict(
        cases=cases, torch_version=torch.__version__, preprocessing=ast.get_source_segment(source, functions[0]),
        tf32=False, weight_sha256=MAGVIT_SHA256, official_commit=OFFICIAL_COMMIT))


def compare_decodes(actual, reference):
    if actual.shape != reference.shape or actual.dtype != np.float32 or reference.dtype != np.float32:
        raise RuntimeError("Decoder shape/dtype mismatch")
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise RuntimeError("Nonfinite decoder output")
    ref = reference.astype(np.float64)
    difference = np.abs(actual.astype(np.float64) - ref)
    outside = int(np.count_nonzero(difference > ATOL + RTOL * np.abs(ref)))
    image_a, image_b = preview(actual), preview(reference)
    pixels_diff = np.abs(image_a.astype(np.int16) - image_b.astype(np.int16))
    reference_norm = float(np.linalg.norm(ref.ravel()))
    return dict(atol=ATOL, rtol=RTOL, values=int(ref.size), outside_tolerance=outside,
                passed=outside == 0, max_abs=float(difference.max()), mean_abs=float(difference.mean()),
                relative_l2=float(np.linalg.norm(difference.ravel())) / max(reference_norm, np.finfo(np.float64).tiny),
                uint8_equal=bool(np.array_equal(image_a, image_b)), uint8_max_abs=int(pixels_diff.max()),
                uint8_mean_abs=float(pixels_diff.mean()), uint8_element_agreement=float(np.mean(image_a == image_b)))


def paired_native_worker(root, output):
    setup_native()
    model = MAGVITv2()
    model.eval()
    entries, elements = load_exact(model, root / "weights/magvitv2/pytorch_model.safetensors")
    print("MAGVIT_PAIRED_WEIGHTS_LOADED", entries, elements, flush=True)
    manifest = json.loads((output / "reference_manifest.json").read_text(encoding="utf-8"))
    rows = []
    for case in manifest["cases"]:
        name = case["name"]
        print("MAGVIT_NATIVE_CASE_START", name, flush=True)
        started = time.monotonic()
        with np.load(output / (name + "-reference.npz"), allow_pickle=False) as archive:
            pixels, codes, reference = (archive[key].copy() for key in ("pixels", "codes", "decoded"))
        with jt.no_grad():
            decoded = model.decode_code(jt.array(codes).stop_grad()).numpy()
            native_codes = model.get_code(jt.array(pixels).stop_grad()).numpy()
        if native_codes.shape != codes.shape or not np.isfinite(native_codes).all():
            raise RuntimeError("Invalid native codes: " + name)
        if native_codes.min() < 0 or native_codes.max() >= 8192 or not np.array_equal(native_codes, native_codes.astype(np.int32)):
            raise RuntimeError("Native codes outside integer codebook: " + name)
        native_codes = native_codes.astype(np.int32)
        metrics = compare_decodes(decoded, reference)
        mismatch = np.argwhere(native_codes != codes)
        row = dict(name=name, resolution=case["resolution"], decode=metrics,
                   encode=dict(tokens=int(codes.size), agreement=float(np.mean(native_codes == codes)),
                               mismatches=int(len(mismatch)),
                               first_mismatch=mismatch[0].tolist() if len(mismatch) else None),
                   seconds=time.monotonic() - started)
        Image.fromarray(preview(decoded)).save(output / (name + "-jittor.png"))
        pair = Image.new("RGB", (2 * case["resolution"], case["resolution"]))
        pair.paste(Image.fromarray(preview(reference)), (0, 0))
        pair.paste(Image.fromarray(preview(decoded)), (case["resolution"], 0))
        pair.save(output / (name + "-comparison.png"))
        np.savez_compressed(output / (name + "-jittor.npz"), codes=native_codes, decoded=decoded)
        rows.append(row)
        write_json(output / (name + "-metrics.json"), row)
        print("MAGVIT_NATIVE_CASE_DONE", name, "encode_agreement", row["encode"]["agreement"],
              "decode_passed", metrics["passed"], "max_abs", metrics["max_abs"], flush=True)
    report = dict(status="completed", cases=rows, atol=ATOL, rtol=RTOL,
                  all_encode_exact=all(row["encode"]["mismatches"] == 0 for row in rows),
                  all_decode_within_tolerance=all(row["decode"]["passed"] for row in rows),
                  state_entries=entries, state_elements_including_buffers=elements,
                  scope="Four paired MagViT cases only; same saved pixels for encoding and same reference codes for decoding; no Transformer or sampler validation")
    write_json(output / "magvit_alignment_report.json", report)
    print("MAGVIT_PAIRED_VALIDATION_COMPLETE", output, "encode_exact", report["all_encode_exact"],
          "decode_within_tolerance", report["all_decode_within_tolerance"], flush=True)


def paired_main(args):
    root = Path.home() / "Show-o-Jittor"
    if args.worker:
        if args.output is None:
            raise RuntimeError("Worker output directory required")
        {"reference": reference_worker, "native": paired_native_worker}[args.worker](root, args.output)
        return
    used = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=memory.used",
                                    "--format=csv,noheader,nounits"], text=True, timeout=10).strip()
    if int(used) > 1024:
        raise RuntimeError("GPU occupied; stop without terminating jobs: " + used + " MiB")
    commit = subprocess.check_output(["git", "-C", str(root / "official"), "rev-parse", "HEAD"], text=True).strip()
    if commit != OFFICIAL_COMMIT:
        raise RuntimeError("Official source revision differs")
    print("VERIFYING_MAGVIT_WEIGHT_SHA256", flush=True)
    if file_hash(root / "weights/magvitv2/pytorch_model.safetensors") != MAGVIT_SHA256:
        raise RuntimeError("MagViT weight SHA256 differs")
    output = root / "results" / ("native-magvit-paired-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).resolve()
    companion = script.with_name("jittor_magvit.py")
    shutil.copyfile(script, output / script.name)
    shutil.copyfile(companion, output / companion.name)
    write_json(output / "run_config.json", dict(gpu=args.gpu, atol=ATOL, rtol=RTOL,
               resolutions=[256, 512], images=["dog.png", "sofa_under_water.jpg"],
               script_sha256=file_hash(script), native_sha256=file_hash(companion),
               weight_sha256=MAGVIT_SHA256, official_commit=commit))
    print("RUN_DIR", output, flush=True)
    import os
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), NVIDIA_TF32_OVERRIDE="0", PYTHONUNBUFFERED="1")
    for worker in ("reference", "native"):
        print("MAGVIT_WORKER_START", worker, flush=True)
        cmd = [sys.executable, "-u", str(script), "--paired", "--worker", worker, "--output", str(output)]
        with (output / (worker + ".log")).open("x", encoding="utf-8") as log:
            with subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, bufsize=1) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                status = process.wait()
        if status != 0:
            write_json(output / "run_status.json", dict(status="failed", worker=worker, returncode=status))
            raise RuntimeError("MagViT stage failed: " + worker + "; see " + str(output))
    report = json.loads((output / "magvit_alignment_report.json").read_text(encoding="utf-8"))
    accepted = report["all_encode_exact"] and report["all_decode_within_tolerance"]
    write_json(output / "run_status.json", dict(status="completed", numerical_acceptance="passed" if accepted else "failed"))
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", action="store_true", help="Compare official and Jittor MagViT on four matched cases")
    parser.add_argument("--gpu", type=int, default=1, choices=(0, 1, 2))
    parser.add_argument("--worker", choices=("reference", "native"), help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.paired:
        paired_main(args)
    elif args.worker or args.output:
        parser.error("Worker options require --paired")
    else:
        main()
