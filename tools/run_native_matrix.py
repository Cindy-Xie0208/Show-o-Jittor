"""Sequential native T2I prompt/seed coverage using the validated pipeline.

Each case also repeats the SAME dog MMU input; these are not different MMU
samples. Independent PyTorch verification replays the native inputs and
decodes the same generated codes, not independently sampled T2I outputs.
"""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


PROMPTS = (
    "A red fox standing in a snowy forest, wildlife photography.",
    "A small wooden cabin beside a mountain lake at sunrise.",
    "A watercolor painting of a vase filled with sunflowers.",
    "A yellow city bus driving along a rainy street at night.",
)
SEEDS = (0, 7, 42)
SOURCES = (
    "infer_full_jittor.py", "native_pipeline.py", "jittor_sampling.py",
    "jittor_magvit.py", "validate_native_magvit.py", "validate_native_sampling.py",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cases():
    return [dict(case="prompt-%02d-seed-%d" % (index, seed),
                 prompt=prompt, seed=seed)
            for index, prompt in enumerate(PROMPTS, 1) for seed in SEEDS]


def idle_gpu(gpu):
    raw = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ], text=True, timeout=10)
    used = int(raw.strip())
    if used > 1024:
        raise RuntimeError("GPU %d occupied: %d MiB; queue stopped without terminating other jobs"
            % (gpu, used))


def verify_sources(root, expected):
    for name in SOURCES:
        if sha256(root / "tools" / name) != expected.get(name):
            raise RuntimeError("Source differs from the passed native run: " + name)


def collect_result(case, run_dir, returncode, elapsed):
    result = dict(case, returncode=returncode, seconds=round(elapsed, 2),
                  run_dir=str(run_dir) if run_dir else None)
    if run_dir is None:
        return dict(result, status="failed", error="Pipeline did not emit RUN_DIR")
    status_file = run_dir / "run_status.json"
    if status_file.is_file():
        result["pipeline_status"] = read_json(status_file)
    report_file = run_dir / "native_pipeline_report.json"
    if not report_file.is_file():
        return dict(result, status="failed", error="Verification report missing")
    report = read_json(report_file)
    result.update(
        functional_checks_passed=report["functional_checks_passed"],
        sampled_forward_checks_within_tolerance=report["sampled_forward_checks_within_tolerance"],
        replay=report["replay"],
        same_codes_decode=report["same_codes_decode"],
        mmu=report["mmu"],
    )
    completed = (returncode == 0
                 and result.get("pipeline_status", {}).get("status") == "completed"
                 and report["functional_checks_passed"] is True)
    result["status"] = "completed" if completed else "failed"
    return result


def run_case(root, output, gpu, case):
    started = time.monotonic()
    run_dir = None
    command = [sys.executable, "-u", str(root / "tools/infer_full_jittor.py"),
               "--native-pipeline", "--gpu", str(gpu),
               "--prompt", case["prompt"], "--seed", str(case["seed"]),
               "--max-new-tokens", "64"]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), NVIDIA_TF32_OVERRIDE="0",
               PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    with (output / (case["case"] + ".log")).open("x", encoding="utf-8") as log:
        with subprocess.Popen(command, cwd=str(root), env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                if line.startswith("RUN_DIR "):
                    candidate = Path(line[len("RUN_DIR "):].strip()).resolve()
                    if candidate.parent == (root / "results").resolve() and candidate.name.startswith("full-native-512-"):
                        run_dir = candidate
            code = process.wait()
    return collect_result(case, run_dir, code, time.monotonic() - started)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--root", type=Path, default=Path.home() / "Show-o-Jittor")
    parser.add_argument("--baseline-run", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    plan = cases()
    if args.dry_run:
        print(json.dumps(dict(cases=plan, distinct_t2i_prompts=4, t2i_runs=12,
            distinct_mmu_inputs=1, repeated_dog_mmu_runs=12), ensure_ascii=False, indent=2))
        return 0
    root = args.root.resolve()
    if root != (Path.home() / "Show-o-Jittor").resolve():
        parser.error("The validated child pipeline requires ~/Show-o-Jittor")
    baseline = args.baseline_run or root / "results/full-native-512-20261114-160133-581841"
    state = read_json(baseline / "run_status.json")
    if state.get("status") != "completed" or state.get("functional_checks_passed") is not True:
        raise RuntimeError("A completed functional baseline is required")
    expected = read_json(baseline / "run_config.json")["source_sha256"]
    verify_sources(root, expected)
    idle_gpu(args.gpu)
    if shutil.disk_usage(root).free < 8 * 1024 ** 3:
        raise RuntimeError("At least 8 GiB free disk space required before starting")
    output = root / "results" / ("native-multi-512-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), output / "run_native_matrix.py")
    started = time.monotonic()
    report = dict(status="running", baseline_run=str(baseline), gpu=args.gpu,
        total_cases=len(plan), completed_cases=0, plan=plan, results=[],
        source_sha256=expected, runner_sha256=sha256(__file__),
        distinct_t2i_prompts=4, distinct_mmu_inputs=1,
        original_83_checks=dict(checked=83, passed=78, failed=5, rerun=False),
        limits=("12 T2I prompt/seed cases; the same dog MMU input is repeated. "
                "Reference uses same-input forward replay and same-code decoding, "
                "not independently sampled PyTorch T2I. Strict numerical failures "
                "are recorded separately from functional acceptance."))
    target = output / "native_multi_report.json"
    save_json(target, report)
    print("NATIVE_MATRIX_RUN_DIR", output, flush=True)
    try:
        for index, case in enumerate(plan, 1):
            idle_gpu(args.gpu)
            verify_sources(root, expected)
            if shutil.disk_usage(root).free < 2 * 1024 ** 3:
                raise RuntimeError("Less than 2 GiB free; stopped before next case")
            report["current_case"] = case["case"]
            save_json(target, report)
            print("NATIVE_MATRIX_CASE_START", index, "/", len(plan), case["case"], flush=True)
            result = run_case(root, output, args.gpu, case)
            report["results"].append(result)
            report["completed_cases"] = sum(row["status"] == "completed" for row in report["results"])
            save_json(target, report)
            print("NATIVE_MATRIX_CASE_DONE", case["case"], result["status"], flush=True)
            if result["status"] != "completed":
                raise RuntimeError("Case failed; queue stopped: " + case["case"])
        report["status"] = "completed"
        report["functional_cases_passed"] = sum(row["functional_checks_passed"] is True for row in report["results"])
        report["strict_replay_cases_passed"] = sum(row["sampled_forward_checks_within_tolerance"] is True for row in report["results"])
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        report["seconds"] = round(time.monotonic() - started, 2)
        save_json(target, report)
        print("NATIVE_MATRIX_STATUS", report["status"], "REPORT", target, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
