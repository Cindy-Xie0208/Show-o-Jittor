"""Native Jittor implementation of the Show-o 1 MaskGIT samplers.

The functions in this file intentionally mirror ``vendor/Show-o/models``.
They operate on Jittor variables and do not import PyTorch.  Randomness is
owned by Jittor; callers can set ``jt.set_global_seed`` before sampling.
"""

import math

import jittor as jt


def cosine_schedule(t):
    return jt.cos(t * (math.pi * 0.5))


def _log(x, eps=1e-20):
    return jt.log(jt.maximum(x, eps))


def gumbel_noise(t):
    # The official implementation samples U(0, 1) and applies -log(-log(U)).
    uniform = jt.rand(t.shape, dtype=t.dtype)
    return -_log(-_log(uniform))


def mask_by_random_topk(mask_len, probs, temperature=1.0):
    confidence = _log(probs) + temperature * gumbel_noise(probs)
    sorted_confidence = jt.sort(confidence, dim=-1)[0]
    cut_off = jt.gather(sorted_confidence, 1, mask_len.int())
    return confidence < cut_off


def categorical_sample(probabilities):
    """Sample one index per row from already-normalized probabilities."""
    return jt.multinomial(probabilities, 1)[:, 0]


def _image_slice(input_ids, num_vq_tokens):
    return input_ids[:, -(num_vq_tokens + 1):-1]


def t2i_generate(
    model,
    input_ids,
    uncond_input_ids,
    attention_mask,
    num_vq_tokens=1024,
    llm_vocab_size=50295,
    num_new_special_tokens=10,
    mask_token_id=58497,
    max_seq_length=128,
    timesteps=50,
    guidance_scale=5.0,
    temperature=1.0,
):
    """Run Show-o's classifier-free MaskGIT loop with a Jittor model.

    ``input_ids`` and ``uncond_input_ids`` contain the full conditional and
    unconditional prompts.  The returned values are the final image codes in
    the model's image-codebook range ``[0, 8192)`` and a compact per-step
    history for reproducibility.
    """
    working = input_ids.clone()
    if timesteps < 1:
        raise ValueError("timesteps must be positive")
    uncond_prefix = (uncond_input_ids[:, :max_seq_length + 1]
                     if uncond_input_ids is not None else None)
    image_ids = _image_slice(working, num_vq_tokens)
    image_ids = jt.where(image_ids == mask_token_id, mask_token_id,
                         image_ids - llm_vocab_size - num_new_special_tokens)
    history = []
    mask_history = []

    for step in range(timesteps):
        if uncond_prefix is not None and guidance_scale > 0:
            uncond = jt.concat([uncond_prefix, working[:, max_seq_length + 1:]], dim=1)
            model_input = jt.concat([working, uncond], dim=0)
            logits = model(model_input, attention_mask)
            cond, unconditional = logits.chunk(2, dim=0)
            logits = (1.0 + guidance_scale) * cond - guidance_scale * unconditional
        else:
            logits = model(working, attention_mask)
        logits = logits[:, -(num_vq_tokens + 1):-1,
                        llm_vocab_size + num_new_special_tokens:-1]
        probs = jt.nn.softmax(logits, dim=-1)
        flat_sampled = categorical_sample(probs.reshape((-1, logits.shape[-1])))
        sampled = flat_sampled.reshape(logits.shape[:-1])
        unknown = image_ids == mask_token_id
        sampled = jt.where(unknown, sampled, image_ids)

        ratio = float(step + 1) / float(timesteps)
        mask_ratio = cosine_schedule(jt.array([ratio], dtype="float32"))[0]
        selected = jt.gather(probs, -1, sampled.int().unsqueeze(-1)).squeeze(-1)
        selected = jt.where(unknown, selected, jt.array(3.4028234663852886e38, dtype=selected.dtype))
        mask_len = jt.floor(num_vq_tokens * mask_ratio).reshape((1, 1)).int()
        unknown_count = jt.sum(unknown, dim=-1, keepdims=True).int()
        mask_len = jt.maximum(jt.array(1, dtype="int32"), jt.minimum(unknown_count - 1, mask_len))
        temperature *= 1.0 - ratio
        masking = mask_by_random_topk(mask_len, selected, temperature)
        working = working.clone()
        replacement = sampled + llm_vocab_size + num_new_special_tokens
        working[:, -(num_vq_tokens + 1):-1] = jt.where(masking, mask_token_id, replacement)
        image_ids = jt.where(masking, mask_token_id, sampled)
        history.append(sampled.numpy().copy())
        mask_history.append(masking.numpy().copy())

    return sampled.numpy().astype("int32"), history, mask_history


def expand_mmu_mask(mask):
    """Append one causal row/column using the official MMU convention."""
    current = mask.squeeze()
    length = current.shape[-1]
    min_value = jt.array(-3.4028234663852886e38, dtype=current.dtype)
    right = jt.ones((length, 1), dtype=current.dtype) * min_value
    row = jt.concat([current, right], dim=1)
    last = jt.concat([current[-1, :], jt.zeros((1,), dtype=current.dtype)], dim=0).unsqueeze(0)
    return jt.concat([row, last], dim=0).unsqueeze(0).unsqueeze(0)


def mmu_generate(
    model,
    input_ids,
    attention_mask,
    max_new_tokens=64,
    eot_token=None,
):
    """Official MMU top_k=1 semantics, including sampling among tied maxima."""
    if input_ids.shape[0] != 1:
        raise ValueError("MMU generation currently supports a single sample")
    ids = input_ids.clone()
    mask = attention_mask.clone()
    tokens = []
    margins = []
    for _ in range(max_new_tokens):
        logits = model(ids, mask)
        next_logits = logits[:, -1, :]
        sorted_logits, _ = jt.sort(next_logits, dim=-1)
        filtered = jt.where(next_logits < sorted_logits[:, -1:],
                            jt.array(float("-inf"), dtype=next_logits.dtype), next_logits)
        next_token = categorical_sample(jt.nn.softmax(filtered, dim=-1)).reshape((1, 1)).int()
        tokens.append(int(next_token.numpy()[0, 0]))
        margins.append(float((sorted_logits[:, -1] - sorted_logits[:, -2]).numpy()[0]))
        ids = jt.concat([ids, next_token], dim=1)
        mask = expand_mmu_mask(mask)
        if eot_token is not None and tokens[-1] == int(eot_token):
            break
    return tokens, margins
