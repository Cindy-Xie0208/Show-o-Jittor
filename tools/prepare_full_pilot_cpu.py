"""CPU-only pilot preparation: image audit/preprocessing, captions and schedules.

No Torch/Jittor import, GPU use, model loading, training or evaluation. Image
codes and final supervised fixtures still require the later encoding stage.
"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from native_pipeline import COMMIT, file_hash, image_pixels, read_json, write_json


MANIFEST_SHA = "dce49014745a0a6c2ba9e5c7a75058a2aafa7a72c616a1d94cd124e1d664b47e"


def select_samples(manifest):
    rows = manifest["samples"]
    training = [i for i, x in enumerate(rows) if x["split"] == "train"]
    validation = [i for i, x in enumerate(rows) if x["split"] == "validation"]
    if len(training) != 128 or len(validation) != 32 or len(rows) != 160:
        raise RuntimeError("Expected verified 128/32 source split")
    if len({x["rgb_sha256"] for x in rows}) != len(rows):
        raise RuntimeError("Duplicate image hashes or train/validation leakage")
    return training[:32], validation[:8]


def schedules(seed=42):
    rng = np.random.default_rng(seed)
    # Persist exact permutations/positions; workers consume these arrays,
    # not framework-specific RNG streams.
    order = np.concatenate([rng.permutation(32) for _ in range(4)])[:100].astype(np.int32)
    train_masks = np.zeros((100, 1024), dtype=bool)
    for row in train_masks:
        row[rng.choice(1024, 512, replace=False)] = True
    evaluation_rng = np.random.default_rng(seed + 1)
    evaluation_masks = np.zeros((8, 1024), dtype=bool)
    for row in evaluation_masks:
        row[evaluation_rng.choice(1024, 512, replace=False)] = True
    return dict(train_sample_order=order, train_half_masks=train_masks,
                validation_half_masks=evaluation_masks,
                validation_all_masks=np.ones((8, 1024), dtype=bool))


def prepare(root, data, tokenizer_run, protocol_path, output=None):
    from tokenizers import Tokenizer

    root, data, tokenizer_run = root.resolve(), data.resolve(), tokenizer_run.resolve()
    protocol = read_json(protocol_path)
    source_manifest = data / "manifest.json"
    if file_hash(source_manifest) != MANIFEST_SHA:
        raise RuntimeError("Source Flickr manifest differs")
    if (protocol["status"] != "planned_not_run"
        or protocol["source"]["official_commit"] != COMMIT
        or protocol["source"]["flickr_manifest_sha256"] != MANIFEST_SHA
        or protocol["data"]["train_images"] != 32 or protocol["data"]["validation_images"] != 8
        or protocol["data"]["image_resolution"] != 512 or protocol["data"]["caption_index"] != 0
        or protocol["training"]["target_steps"] != 100 or protocol["training"]["seed"] != 42):
        raise RuntimeError("Protocol differs from the implemented 32/8, 512px, caption0, 100-step schedule")
    source_config = read_json(tokenizer_run / "run_config.json")
    if source_config["official_commit"] != COMMIT:
        raise RuntimeError("Tokenizer source official commit differs")
    if file_hash(Path(__file__).with_name("native_pipeline.py")) != source_config["source_sha256"]["native_pipeline.py"]:
        raise RuntimeError("CPU preprocessor differs from validated native pipeline")
    status = read_json(tokenizer_run / "run_status.json")
    if status.get("status") != "completed" or status.get("functional_checks_passed") is not True:
        raise RuntimeError("A successful native baseline is required for tokenizer provenance")
    special = read_json(tokenizer_run / "prepare.json")
    tokenizer_path = tokenizer_run / "tokenizer/tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size() != 50305:
        raise RuntimeError("Unexpected tokenizer vocabulary")
    manifest = read_json(source_manifest)
    train_indices, val_indices = select_samples(manifest)
    if shutil.disk_usage(root).free < 512 * 1024 ** 2:
        raise RuntimeError("Need at least 512 MiB free for preprocessed pilot images")
    output = output or root / "data" / ("full-pilot-cpu-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    output.mkdir(parents=True, exist_ok=False)
    (output / "pixels").mkdir()
    (output / "tokenizer").mkdir()
    shutil.copyfile(tokenizer_path, output / "tokenizer/tokenizer.json")
    shutil.copyfile(Path(__file__), output / Path(__file__).name)
    shutil.copyfile(Path(__file__).with_name("native_pipeline.py"), output / "native_pipeline.py")
    write_json(output / "protocol.json", protocol)
    selected = []
    for number, source_index in enumerate(train_indices + val_indices):
        row = manifest["samples"][source_index]
        source = (data / row["image"]).resolve()
        source.relative_to(data)
        with Image.open(source) as image:
            rgb = image.convert("RGB")
        if hashlib.sha256(str(rgb.size).encode("ascii") + rgb.tobytes()).hexdigest() != row["rgb_sha256"]:
            raise RuntimeError("Source image changed: " + row["image"])
        caption = row["captions"][0]
        if not isinstance(caption, str) or not caption.strip():
            raise RuntimeError("Caption is empty")
        text_ids = tokenizer.encode(caption).ids
        if not text_ids or min(text_ids) < 0 or max(text_ids) >= 50305:
            raise RuntimeError("Invalid caption tokens")
        pixels = image_pixels(source, resolution=512)
        if pixels.shape != (1, 3, 512, 512) or pixels.dtype != np.float32 or not np.isfinite(pixels).all():
            raise RuntimeError("Invalid CPU preprocessing")
        pixel_name = "pixels/%03d.npy" % number
        np.save(output / pixel_name, pixels, allow_pickle=False)
        selected.append(dict(row_index=number, source_manifest_index=source_index,
            split=row["split"], source_image=row["image"], source_rgb_sha256=row["rgb_sha256"],
            caption_index=0, caption=caption, caption_token_ids=text_ids,
            caption_token_count=len(text_ids), pixel_file=pixel_name,
            pixel_sha256=file_hash(output / pixel_name), pixel_shape=list(pixels.shape)))
        if number == 0 or (number + 1) % 8 == 0:
            print("CPU_PILOT_IMAGE_READY", number + 1, "/ 40", flush=True)
    arrays = schedules(protocol["training"]["seed"])
    np.savez_compressed(output / "schedule.npz", **arrays)
    summary = dict(status="cpu_preparation_completed", train_images=32, validation_images=8,
        source_manifest_sha256=MANIFEST_SHA, source_data=str(data), source_manifest_order_preserved=True,
        train_validation_disjoint=True, all_selected_rgb_hashes_verified=True,
        protocol_sha256=file_hash(protocol_path), script_sha256=file_hash(Path(__file__)),
        preprocessing_sha256=file_hash(Path(__file__).with_name("native_pipeline.py")),
        tokenizer_sha256=file_hash(tokenizer_path), specials=special["specials"],
        bos=special["bos"], eos=special["eos"],
        schedule_sha256=file_hash(output / "schedule.npz"),
        schedule=dict(seed=42, steps=100, calibration_steps=10, mask_count=512,
            sample_order="Local train index 0..31 into the first 32 selected rows; both tasks and both frameworks share each step index",
            masks="Same persisted half-mask positions used by both frameworks; no per-framework RNG regeneration",
            evaluation="Eight held-out images, fixed half masks and all-image masks at steps 0/10/100"),
        rows=selected, pending=["512px MagViT encoding of 40 images", "final input/label/mask fixture construction and official oracle checks", "full Transformer gradient acceptance", "10-step calibration and subsequent fine-tuning", "held-out subset metrics"],
        limits="CPU data preparation only, no training/evaluation/model outputs. Dataset image arrays and captions remain local; do not publish this data directory.")
    write_json(output / "pilot_manifest.json", summary)
    print("CPU_PILOT_PREPARATION_COMPLETE", output, flush=True)
    print("CPU_PILOT_COUNTS train=32 validation=8 steps=100 half_masks=512/1024 GPU_USED=False", flush=True)
    return output


def main():
    root = Path.home() / "Show-o-Jittor"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--data", type=Path, default=root / "data/flickr8k-128-32-20261026-203910-164291")
    parser.add_argument("--tokenizer-run", type=Path, default=root / "results/full-native-512-20261114-160133-581841")
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("full_pilot_protocol.json"))
    args = parser.parse_args()
    prepare(args.root, args.data, args.tokenizer_run, args.protocol)


if __name__ == "__main__":
    main()
