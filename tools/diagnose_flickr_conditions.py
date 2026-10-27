"""Server-only condition sensitivity and image-code baselines; no training."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys


DEFAULT_RUN = "flickr-jittor-20261026-210721-135146"


def worker(args):
    import numpy as np
    import jittor as jt
    import align_tiny_jittor as port
    import train_flickr_jittor_v1 as training

    reference = args.run / "reference"
    data = training.load_data(reference)
    config = json.loads((reference / "config.json").read_text())
    summary = json.loads((args.run / "training_report.json").read_text())
    if summary["status"] != "passed":
        raise ValueError("Expected the completed training run")
    checkpoint = args.run / summary["best_checkpoint"]
    jt.flags.use_cuda = 1
    model = port.model_class(jt, np)(config)
    with np.load(checkpoint, allow_pickle=False) as archive:
        port.strict_load(model, {key: archive[key] for key in archive.files}, jt, np)
    model.eval()
    special = data["special"]
    offset = special["image_offset"]
    validation = data["validation_indices"].tolist()
    image_donors = dict(zip(validation, validation[1:] + validation[:1]))
    caption_donors = {}
    for index in validation:
        length = int(data["text_lengths"][index, 0])
        candidates = [(other, c) for other in validation if other != index for c in range(5)
                      if int(data["text_lengths"][other, c]) == length
                      and not np.array_equal(data["text_ids"][index, 0, :length],
                                             data["text_ids"][other, c, :length])]
        if candidates:
            caption_donors[index] = candidates[0]
    eligible = sorted(caption_donors)
    if len(eligible) < 2:
        raise ValueError("Not enough equal-length different-image caption donors")

    def measure(task, shuffled, indices):
        count = correct = in_codebook = restricted_correct = 0
        loss_sum = restricted_sum = 0.0
        with jt.no_grad():
            for index in indices:
                raw = training.make_batch(data, [index], [0], task, np.random.default_rng(123), 1.0)
                original_labels = raw["labels"].copy()
                if shuffled and task == "mmu":
                    raw["input_ids"][0, 2:258] = data["image_tokens"][image_donors[index]] + offset
                elif shuffled:
                    other, caption = caption_donors[index]
                    length = int(data["text_lengths"][index, 0])
                    raw["input_ids"][0, 1:1 + length] = data["text_ids"][other, caption, :length]
                if not np.array_equal(original_labels, raw["labels"]):
                    raise ValueError("Sensitivity probe changed a prediction target")
                logits, loss, trace = port.loss_for(jt, model, task, training.jt_batch(jt, raw))
                predicted = logits.argmax(dim=-1)[0].numpy()
                labels = raw["labels"]
                if task == "mmu":
                    predicted, labels = predicted[:, :-1], labels[:, 1:]
                valid = labels != -100
                n = int(valid.sum())
                value = float(loss.numpy().item())
                if not np.isfinite(value):
                    raise ValueError("Non-finite diagnostic loss")
                count += n
                correct += int((predicted[valid] == labels[valid]).sum())
                loss_sum += value * n
                if task == "t2i":
                    begin = int(np.flatnonzero(raw["input_ids"][0] == special["soi"])[0]) + 1
                    scores = logits[:, begin:begin + 256, offset:offset + 8192]
                    truth = data["image_tokens"][index].astype(np.int32)
                    restricted = scores.argmax(dim=-1)[0].numpy()[0]
                    restricted_correct += int((restricted == truth).sum())
                    restricted_loss = jt.nn.cross_entropy_loss(scores.reshape((-1, 8192)), jt.array(truth))
                    restricted_sum += float(restricted_loss.numpy().item()) * n
                    chosen = predicted[valid]
                    in_codebook += int(((chosen >= offset) & (chosen < offset + 8192)).sum())
                    del scores, restricted_loss
                del logits, loss, trace
        result = dict(images=len(indices), target_tokens=count, full_vocab_loss=loss_sum / count,
                      full_vocab_token_accuracy=correct / count)
        if task == "t2i":
            result.update(codebook_only_loss=restricted_sum / count,
                          codebook_only_token_accuracy=restricted_correct / count,
                          full_vocab_prediction_in_codebook_fraction=in_codebook / count)
        print(task, "shuffled" if shuffled else "correct", json.dumps(result), flush=True)
        return result

    results = dict(mmu_correct=measure("mmu", False, validation),
                   mmu_shuffled_images=measure("mmu", True, validation),
                   t2i_correct=measure("t2i", False, eligible),
                   t2i_shuffled_captions=measure("t2i", True, eligible))
    frequency = np.bincount(data["image_tokens"][data["train_indices"]].ravel(), minlength=8192).astype(np.float64)
    probability = (frequency + 1) / (frequency.sum() + 8192)
    targets = data["image_tokens"][eligible]
    report = dict(
        scope="Held-out condition sensitivity; no optimization, no free-running generation or grounding claim",
        source_run=str(args.run), checkpoint=str(checkpoint), results=results,
        t2i_image_indices=eligible, equal_length_caption_donors={str(k): list(v) for k, v in caption_donors.items()},
        mmu_wrong_minus_correct_loss=results["mmu_shuffled_images"]["full_vocab_loss"] - results["mmu_correct"]["full_vocab_loss"],
        t2i_wrong_minus_correct_codebook_loss=results["t2i_shuffled_captions"]["codebook_only_loss"] - results["t2i_correct"]["codebook_only_loss"],
        image_code_baselines=dict(scope="Same eligible images, codebook-only space; frequency fitted on training images only",
                                  uniform_loss=float(np.log(8192)), uniform_expected_accuracy=1 / 8192,
                                  train_unigram_add_one_loss=float(-np.log(probability[targets]).mean()),
                                  train_most_frequent_code_accuracy=float((targets == frequency.argmax()).mean())))
    training.write_json(args.output / "diagnostics.json", report)
    print(json.dumps(report, indent=2), flush=True)
    print("FLICKR_CONDITION_DIAGNOSTICS_DONE", args.output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path.home() / "Show-o-Jittor/results" / DEFAULT_RUN)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if Path(sys.prefix).resolve() != (Path.home() / "envs/showo-jittor").resolve():
        raise RuntimeError("Activate the showo-jittor environment")
    if args.worker:
        worker(args)
        return
    import train_flickr_jittor_v1 as training

    args.run = args.run.resolve(strict=True)
    usage = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=memory.used",
                                     "--format=csv,noheader,nounits"], text=True).strip()
    if int(usage) > 1024:
        raise RuntimeError("Selected GPU is occupied; inspect resource allocation")
    args.output = args.run.parent / ("flickr-diagnostics-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    args.output.mkdir(exist_ok=False)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--run", str(args.run),
               "--gpu", str(args.gpu), "--output", str(args.output)]
    print("DIAGNOSTICS_DIR", args.output, flush=True)
    training.run_stage(command, training.worker_environment(args, True), args.output / "diagnostics.log")


if __name__ == "__main__":
    main()
