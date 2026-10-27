# Show-o 1：PyTorch 参考与 Jittor Transformer 迁移

本项目复现原始 [Show-o](https://github.com/showlab/Show-o)（Show-o 1，ICLR 2025）的统一多模态 Transformer，并将核心 Transformer 前向迁移到 Jittor。项目只覆盖 Show-o 1，不包含 Show-o 2。

项目将官方 PyTorch 实现作为可核验参考：固定官方代码提交、配置、分词器和预训练权重，先建立 PyTorch 基线，再用相同权重在 Jittor 中运行同一套 MMU/T2I 输入。采样循环和 MagViT 视觉编解码器仍使用官方 PyTorch 实现，因此本文档不把结果表述为完整原生 Jittor 流程。

## 项目范围

最低验收范围包括：

- Tiny Show-o 的 PyTorch/Jittor 前向、梯度、两任务训练、checkpoint 保存加载和 token 推理对齐；
- 一个真实 Flickr8k 先导子集上的图文训练流程与验证日志；
- 官方 Show-o 1 512×512 权重的严格加载、Transformer 前向对照和多样例生成一致性；
- 可复现的脚本、结果摘要和限制说明。

完整 1.3B 模型训练、完整模型梯度验证、原生 Jittor 采样器以及 MagViT 迁移属于后续 stretch 工作，不作为本次最低验收条件。

## 环境

服务器验证环境：Python 3.10.22、PyTorch 2.4.1、Jittor 1.3.11.0、CUDA Toolkit 12.4、GCC 11.4、NVIDIA L20。运行完整模型需要约 6 GB 的 Show-o 权重文件和可用的 CUDA GPU。

官方代码固定在提交 `45a5a2de01d1ebd10cd5864d29310a76476cdf23`。512×512 checkpoint 来自 Hugging Face `showlab/show-o-512x512`，revision `dbe757d951099968fb102d249d35a764528df2dd`，文件大小 5,792,758,016 字节，SHA256：

```text
638739e52da4f5cba220b7cd59c04819f852dda180124c5a32c9c48b35a7299f
```

权重、Flickr 图片和运行时缓存不纳入 Git；请按官方仓库和数据集条款自行下载。

## 结果总表

| 项目 | 结果 |
| --- | --- |
| Tiny 初始前向/梯度 | PyTorch 与 Jittor 逐项对齐通过；最大前向绝对误差 `1.67e-6`，最大梯度绝对误差 `2.09e-7` |
| Tiny 训练与重载 | MMU/T2I 各 50 步，checkpoint 重载最大误差为 0；训练日志和 token 推理流程通过 |
| Flickr8k 先导训练 | 128 张训练图、32 张验证图、500 步；最终 MMU loss `5.4042`，T2I 50% mask loss `9.2067`，全 mask loss `9.2103` |
| 完整 512 权重 | 437 个张量、1,448,176,770 个参数，严格载入通过 |
| 83 项完整前向检查 | 78/83 通过；原 `atol=rtol=1e-4` 不变，5 项仍保留为失败 |
| 本底噪声对照 | PyTorch CUDA/CPU FP32 相对 FP64 在阈值内；Jittor 仅在第 23 层半 mask 的少数近零相消点略高于该本底，最坏容差比 1.61，共 6 个元素 |
| 多样例一致性 | 8 个 T2I（4 prompt × 2 seed）和 4 个 MMU 样例的最终图像码/回答 token 与 PyTorch 完全一致；其中 1 个 T2I 第 1 步出现单个采样分歧，最终码仍一致 |

### Tiny 与 Flickr 日志节点

Tiny 50 步终点：

| 任务 | PyTorch loss | Jittor loss |
| --- | ---: | ---: |
| MMU | 0.08059389 | 0.08099939 |
| T2I | 0.13718115 | 0.13765694 |

Flickr8k 先导验证集固定节点（32 张图）：

| step | MMU loss | MMU token accuracy | T2I 50% loss | T2I 50% accuracy | T2I 全 mask loss | T2I 全 mask accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 11.0028 | 0% | 10.9946 | 0% | 10.9977 | 0% |
| 100 | 7.6336 | 22.61% | 9.4107 | 0% | 9.4336 | 0% |
| 500 | 5.4042 | 27.39% | 9.2067 | 0.0244% | 9.2103 | 0.0488% |

这些数值说明训练和 checkpoint 流程完成；它们不是正式数据集指标，也不代表真实图像生成质量达标。

### 多样例说明

12 个样例的结论限定在上述输入范围内。`prompt-02-seed-42` 的一次第 1 步分歧发生在 temperature=1 的随机采样过程中，最可能的原因是随机数落在累积概率边界附近；报告没有保存逐步 mask 状态，因此不能把“后续重新 mask”写成已确认事实。

## 已知限制

- 第 23 层半 mask 输出的 6 个近零元素略超逐元素阈值，误差传播到部分 logits；整体 relative L2 仍约 `8e-7`，argmax 和最终生成 token 保持一致。严格结论仍是 78/83 通过，不放宽阈值、不改判。
- 完整模型的 Transformer 前向使用 Jittor；采样器、随机采样和 MagViT 视觉编解码使用官方 PyTorch 独立进程。
- 尚未完成完整模型梯度/训练、原生 Jittor 采样器、MagViT 原生迁移和正式 FID/CLIP/理解基准。
- Flickr 先导训练只用于验证流程，不能外推为论文规模复现。

## 代码与复现入口

所有可复用代码在 `tools/`：

- `align_tiny_jittor.py`：Tiny 模型构造、前向/梯度对齐、训练和 checkpoint 验证。
- `train_tiny_reference.py`：Tiny PyTorch 参考 fixture 与训练基线。
- `train_flickr_jittor.py`、`prepare_flickr8k_subset.py`、`precompute_flickr_tokens.py`：Flickr8k 先导数据与训练流程。
- `align_official_jittor.py`：完整 512 权重加载和 83 项前向对照。
- `diagnose_official_mlp.py`：第 23 层算子与本底噪声诊断。
- `infer_official_baseline.py`、`infer_official_t2i.py`：官方 PyTorch MMU/T2I 参考。
- `infer_full_jittor.py`：Jittor Transformer 的 512 T2I 和 MMU 对照推理。
- `multi_sample_eval.py`：8 个 T2I 与 4 个 MMU 的一次加载多样例评估。
- `download_showo_ranges.py`、`inspect_official_showo.py`、`verify_official_showo.py`：官方权重元数据、下载和结构核验。

典型服务器运行方式如下。请将环境变量和本地路径替换为自己的配置；不要把密钥、服务器地址、权重或数据提交到 Git。

```bash
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python tools/verify_official_showo.py
python tools/align_official_jittor.py --gpu 0 --reference-run <reference-run>
python tools/infer_full_jittor.py --gpu 0 --candidate-run <candidate-run>
python tools/multi_sample_eval.py --gpu 0 --candidate-run <candidate-run>
```

完整模型脚本需要官方 Show-o 源码、Phi tokenizer、512 checkpoint 和 MagViT 权重；这些文件均由脚本参数或环境路径提供，不随本仓库分发。

## 来源与许可

本项目基于 [showlab/Show-o](https://github.com/showlab/Show-o)，官方仓库为 Apache License 2.0，许可文本见根目录 [`LICENSE`](LICENSE)。请同时遵守 Show-o、Hugging Face checkpoint、Flickr8k 和 MagViT 数据/模型的各自条款。
