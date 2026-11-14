"""Package small, sanitized native validation reports and generated PNGs.

No model execution, checkpoint copying, or modification of original runs.
Existing export destinations are never overwritten.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean_string(value, root):
    value = value.replace(str(root), "<PROJECT_ROOT>")
    value = value.replace(str(root.parent), "<HOME>")
    value = re.sub(r"/public/[^/\s\"'<>]+", "<HOME>", value)
    value = re.sub(r"\b[\w.-]+\.leven\.fun\b", "<SSH_HOST>", value)
    return re.sub(r"\b[\w.-]+@dev-[^\s\"'<>]+", "<SSH_ACCOUNT>", value)


def sanitize(value, root):
    if isinstance(value, dict):
        return {clean_string(k, root): sanitize(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v, root) for v in value]
    return clean_string(value, root) if isinstance(value, str) else value


def check_text(text):
    patterns = (r"/public/", r"[\w.-]+\.leven\.fun", r"[\w.-]+@dev-",
                r"\b(?:ghp_|github_pat_|hf_)[A-Za-z0-9_]{16,}")
    if any(re.search(pattern, text) for pattern in patterns):
        raise RuntimeError("Sensitive content remains; export stopped")


def inside(path, parent):
    resolved = Path(path).resolve()
    resolved.relative_to(parent.resolve())
    return resolved


class Export:
    def __init__(self, root, output):
        self.root, self.output = root, output
        self.files = []

    def write(self, relative, value, source=None):
        path = self.output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(sanitize(value, self.root), ensure_ascii=False,
                          indent=2, allow_nan=False) + "\n"
        check_text(text)
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        self.record(path, source)

    def record(self, path, source=None):
        self.files.append(dict(path=path.relative_to(self.output).as_posix(),
            bytes=path.stat().st_size, sha256=digest(path),
            source=source.relative_to(self.root).as_posix() if source else None,
            original_sha256=digest(source) if source else None))

    def report(self, source, relative):
        if source.stat().st_size > 10 * 1024 ** 2:
            raise RuntimeError("Report unexpectedly large: " + source.name)
        self.write(relative, read_json(source), source)

    def image(self, source, relative):
        from PIL import Image
        path = self.output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            if image.width > 2048 or image.height > 2048:
                raise RuntimeError("Unexpected image size")
            # Re-encode pixels without source metadata or internal path text.
            pixels = Image.new("RGB", image.size)
            pixels.paste(image.convert("RGB"))
            pixels.save(path)
        self.record(path, source)


def export(root, matrix_dir, diagnostic_dir, output, archive):
    from PIL import Image, ImageDraw, ImageFont
    root = root.resolve()
    results_root = root / "results"
    matrix_dir = inside(matrix_dir, results_root)
    diagnostic_dir = inside(diagnostic_dir, results_root)
    matrix = read_json(matrix_dir / "native_multi_report.json")
    cases = matrix["results"]
    if len(cases) != 12 or len({row["case"] for row in cases}) != 12:
        raise RuntimeError("Expected 12 unique recorded cases")
    if output.exists() or archive.exists():
        raise RuntimeError("Export already exists; select new --output and --archive paths")
    # Validate required inputs before creating the package.
    runs = []
    for row in cases:
        if not re.fullmatch(r"prompt-\d{2}-seed-\d+", row["case"]):
            raise RuntimeError("Unexpected case name")
        run = inside(row["run_dir"], results_root)
        for name in ("native_pipeline_report.json", "native_result.json", "run_config.json",
                     "run_status.json", "generated.png", "comparison.png"):
            if not (run / name).is_file():
                raise RuntimeError("Missing case asset: " + row["case"] + "/" + name)
        report = read_json(run / "native_pipeline_report.json")
        for key in ("functional_checks_passed", "sampled_forward_checks_within_tolerance", "replay", "same_codes_decode", "mmu"):
            if report[key] != row[key]:
                raise RuntimeError("Matrix/case report mismatch: " + row["case"] + "/" + key)
        runs.append(run)
    diagnostic = read_json(diagnostic_dir / "argmax_diagnostic.json")
    if inside(diagnostic["source_run"], results_root) not in runs:
        raise RuntimeError("Diagnostic does not belong to this matrix")
    output.mkdir(parents=True, exist_ok=False)
    package = Export(root, output)
    package.report(matrix_dir / "native_multi_report.json", "native_multi_report.json")
    package.report(diagnostic_dir / "argmax_diagnostic.json", "argmax_diagnostic.json")
    replay = []
    previews = []
    for row, run in zip(cases, runs):
        prefix = "cases/" + row["case"] + "/"
        for name in ("native_pipeline_report.json", "native_result.json", "run_config.json", "run_status.json"):
            package.report(run / name, prefix + name)
        for name in ("generated.png", "comparison.png"):
            package.image(run / name, prefix + name)
        selected = []
        for stage in ("prepare", "native", "verify"):
            source = run / (stage + ".log")
            if source.is_file():
                for line in source.read_text(encoding="utf-8").splitlines():
                    if line.startswith(("NATIVE_", "MAGVIT_INTEGER_BUFFER", "REPLAY_CHECK ", "FULL_NATIVE_")):
                        selected.append("[" + stage + "] " + clean_string(line, root))
        log = output / prefix / "validation_log.txt"
        text = "\n".join(selected) + "\n"
        check_text(text)
        with log.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        package.record(log)
        replay.extend(dict(case=row["case"], **item) for item in row["replay"])
        previews.append((row["case"], output / prefix / "generated.png"))
    sheet = Image.new("RGB", (3 * 288, 4 * 278), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (name, source) in enumerate(previews):
        x, y = (index % 3) * 288, (index // 3) * 278
        with Image.open(source) as image:
            sheet.paste(image.resize((256, 256), Image.Resampling.LANCZOS), (x + 16, y))
        draw.text((x + 8, y + 258), name, fill="black", font=font)
    sheet_path = output / "native_t2i_contact_sheet.png"
    sheet.save(sheet_path)
    package.record(sheet_path)
    functional = sum(x["functional_checks_passed"] is True for x in cases)
    summary = dict(
        case_count=len(cases), recorded_case_count=len(cases), matrix_status=matrix["status"],
        elapsed_minutes=round(matrix["seconds"] / 60, 2),
        functional_cases_passed=functional,
        strict_cases_passed=sum(x["sampled_forward_checks_within_tolerance"] is True for x in cases),
        repeated_dog_mmu_passed=sum(x["mmu"]["exact_token_match"] is True for x in cases),
        same_codes_decode_passed=sum(x["same_codes_decode"]["passed"] is True for x in cases),
        replay_checks=len(replay),
        strict_replay_checks_passed=sum(x["metrics"]["within_tolerance"] is True for x in replay),
        all_argmax_equal_replay_checks=sum(x["metrics"]["argmax_agreement"] == 1.0 for x in replay),
        nonidentical_argmax_checks=[x for x in replay if x["metrics"]["argmax_agreement"] != 1.0],
        original_83_checks=matrix["original_83_checks"],
        diagnostic=dict(step=diagnostic["step"], difference_reproduced=diagnostic["difference_reproduced"],
                        different_positions=diagnostic["different_positions"]),
        limits="12 T2I prompt/seed cases, one repeated dog MMU input. Same-input replay and same-code decode, no independent PyTorch T2I sampling equivalence. Original failures unchanged.",
        image_review="Pending visual review; contact sheet is a preview, full PNGs are included.")
    package.write("native_release_summary.json", summary)
    support = (("native-magvit-paired-20261114-145717-007496", "magvit_alignment_report.json"),
               ("native-sampler-contract-20261114-151720-646497", "sampler_contract_report.json"))
    for folder, name in support:
        package.report(results_root / folder / name, "components/" + name)
    package.write("manifest.json", dict(schema_version=1, files=package.files.copy(),
        transformation="JSON/log internal paths sanitized; JSON numbers/booleans preserved; PNG pixels re-encoded without metadata. No weights, tensors, or dataset files."))
    total = sum(x.stat().st_size for x in output.rglob("*") if x.is_file())
    if total > 64 * 1024 ** 2:
        raise RuntimeError("Export exceeds 64 MiB; archive not created")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for source in sorted(output.rglob("*")):
            if source.is_file():
                bundle.write(source, output.name + "/" + source.relative_to(output).as_posix())
    print("EXPORT_DIR", output, flush=True)
    print("ARCHIVE", archive, flush=True)
    print("ARCHIVE_BYTES", archive.stat().st_size, flush=True)
    print("ARCHIVE_SHA256", digest(archive), flush=True)
    print("SUMMARY", json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    root = Path.home() / "Show-o-Jittor"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--matrix", type=Path, default=root / "results/native-multi-512-20261114-161708-215097")
    parser.add_argument("--diagnostic", type=Path, default=root / "results/native-argmax-diagnostic-20261114-200935-511823")
    parser.add_argument("--output", type=Path, default=root / "results/summary/native-20261114")
    parser.add_argument("--archive", type=Path, default=Path.home() / "tmp/showo-delivery/native-results-20261114.zip")
    args = parser.parse_args()
    export(args.root, args.matrix, args.diagnostic, args.output, args.archive)


if __name__ == "__main__":
    main()
