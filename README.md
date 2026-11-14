# Show-o 1：Jittor 原生推理迁移与 PyTorch 对齐验证

本项目复现原始 [Show-o](https://github.com/showlab/Show-o)（Show-o 1，ICLR 2025），将 Transformer 前向、T2I/MMU 采样及 MagViT 视觉编解码迁移到 Jittor，并验证 512×512 原生推理流程。项目只覆盖 Show-o 1，不包含 Show-o 2。

项目将官方 PyTorch 实现作为可核验参考：固定官方代码提交、配置、分词器和预训练权重，先建立 PyTorch 基线，再用相同权重验证 Jittor 实现。原生推理入口已在 1 个 512px T2I 和 1 个 dog MMU 样例上通过功能检查；模型前向、采样及视觉编解码使用 Jittor，预处理使用 NumPy/Pillow/tokenizers，PyTorch 在独立进程中准备参考输入和复核结果。原 12 样例评估使用的是 Jittor Transformer 与 PyTorch 采样器/MagViT 的桥接流程，两组证据分别报告。

## 项目范围

最低验收范围包括：

- Tiny Show-o 的 PyTorch/Jittor 前向、梯度、两任务训练、checkpoint 保存加载和 token 推理对齐；
- 一个真实 Flickr8k 先导子集上的图文训练流程与验证日志；
- 官方 Show-o 1 512×512 权重的严格加载、Transformer 前向对照和多样例生成一致性；
- 可复现的脚本、结果摘要和限制说明。

原生 Jittor 采样器和 MagViT 迁移作为扩展工作已完成上述单样例端到端功能验证；完整模型训练、完整模型梯度验证仍未完成，不作为本次最低验收条件。

后续工作已扩展为完整 Transformer 梯度验证、小数据短程微调及固定验证子集评估。目前 32 张训练图和 8 张验证图的 CPU 准备已完成；完整梯度验证等待可用 GPU，10 步校准、100 步微调和相应验证指标尚未运行。

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
| 桥接流程多样例一致性 | Jittor Transformer + PyTorch 采样器/MagViT：8 个 T2I（4 prompt × 2 seed）和 4 个 MMU 样例的最终图像码/回答 token 与 PyTorch 完全一致；其中 1 个 T2I 第 1 步出现单个采样分歧，最终码仍一致 |
| 原生 Jittor MagViT | dog/sofa 各 256/512px 共 4 组配对检查通过；编码图像码一致率均为 1.0，解码均满足 `atol=rtol=1e-4`，四组最大绝对误差分别为 `8.88e-6`、`2.59e-5`、`7.97e-6`、`4.11e-5` |
| 原生 Jittor 采样器契约检查 | 全 mask CFG 50 步、部分 mask CFG 6 步、无 CFG 6 步，以及 MMU 动态 mask/EOS 均通过；使用合成 logits 和受控随机选择，原生 categorical/Gumbel 基础检查通过，尚非全模型生成验收 |
| 原生 512 推理功能验证 | 1 个 T2I 完成 50 步采样与原生 MagViT 解码；1 个 dog MMU 的 64 个回答 token 与既有 PyTorch 基线完全一致；同码解码通过容差检查，6 次前向抽查的 argmax 一致率均为 1.0，`functional_checks_passed=True` |
| 原生流程同输入前向抽查 | T2I 第 1/25/50 步、MMU 第 1/33/64 步共 6 次，4 次满足 `atol=rtol=1e-4`；T2I 第 25/50 步未通过，`sampled_forward_checks_within_tolerance=False`；与原 83 项检查分开记录 |
| 原生多样例队列 | 4 个新 prompt × 3 个 seed 共 12 轮，42.51 分钟完成生成及复核；功能检查 11/12 通过，同一 dog 的 MMU token 和同码解码均 12/12 通过，严格容差整例通过数为 0/12。末例 `prompt-04-seed-42` 的 T2I 第 25 步 argmax 一致率为 `1023/1024`，因此功能检查未通过 |

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

### 原生端到端验证

2026-11-14 的服务器运行 `full-native-512-20261114-160133-581841` 输出 `FULL_NATIVE_PIPELINE_VERIFICATION_COMPLETE`，功能检查通过，抽查逐元素容差检查未全部通过。原生阶段耗时 375.18 秒，包含该轮模型加载和生成，不作为独立吞吐率基准。以下为独立 PyTorch 对同一输入重放的日志结果：

| 抽查位置 | 最大绝对误差 | relative L2 | 超限元素数 | argmax 一致率 | 逐元素容差通过 |
| --- | ---: | ---: | ---: | ---: | --- |
| T2I 第 1 步 | 6.8665e-5 | 1.6476e-6 | 0 | 1.0 | 是 |
| T2I 第 25 步 | 3.9291e-4 | 8.2336e-6 | 1556 | 1.0 | 否 |
| T2I 第 50 步 | 4.5776e-4 | 9.9025e-6 | 9915 | 1.0 | 否 |
| MMU 第 1 步 | 5.7220e-5 | 7.7723e-7 | 0 | 1.0 | 是 |
| MMU 第 33 步 | 4.0054e-5 | 6.0887e-7 | 0 | 1.0 | 是 |
| MMU 第 64 步 | 3.6240e-5 | 4.3024e-7 | 0 | 1.0 | 是 |

该轮 T2I 使用 Jittor 独立随机流；PyTorch 解码复核使用同一组 Jittor 生成的图像码，因此同码解码通过不代表两套框架独立采样会产生相同图像码或图片。功能标志包含 MMU token 全等、同码解码满足容差和 6 次抽查 argmax 全等；它不代表所有前向逐元素对齐。新一轮超限的具体原因尚未单独定位，不能直接归因为 FP32 本底噪声。

后续原生多样例队列 `native-multi-512-20261114-161708-215097` 覆盖 4 个新 prompt × seed 0/7/42。日志显示队列在第 12 轮 `prompt-04-seed-42` 复核后以失败退出：该例 50 步图像生成和 MMU 推理均已完成，MMU token 全等；T2I 第 25 步同输入重放的 argmax 一致率为 `0.9990234375`，即 1024 个图像位置中有 1 个不同，第 1/50 步 argmax 均全等。第 25/50 步超限元素数分别为 1047/9125，最大绝对误差分别为 `3.1662e-4`/`4.3869e-4`。这是已有验收规则捕获的不一致，不能据此断言最终图像码发生分歧。此前“argmax 完全一致”的结论只适用于相应已通过样例，不外推到本队列全部输入。

完整汇总 JSON 已核对：12 轮均保存结果，耗时 42.51 分钟；功能检查 11/12 通过，12 轮的 MMU token 和同码解码均通过。严格前向抽查的整例标志均为 False（0/12），表示每轮至少一项超限，不代表每次抽查均失败。MMU 仍只有同一个 dog 输入，T2I 同码解码仍不等于两个框架独立采样同图。

对末例第 25 步的单步重放复现了同一个 argmax 差异，位置为图像码索引 352（从 0 计数，32×32 网格的第 11 行、第 0 列）。Jittor 的前两名为 6167/7260，margin 为 `1.90735e-6`；PyTorch 的前两名次序相反，margin 为 `1.33514e-5`。该位置的最大 logits 绝对误差为 `6.77109e-5`，前两名的次序在这一微小间隔下发生翻转，不能表述为 argmax 全等。

保存的采样历史确认该位置在第 25 步输入时仍为 mask，实际原生采样码为 7775（并非任一框架的 argmax），该步结束后再次被 mask，最终原生码为 7396。这证实了该位置的重新 mask，但没有对同一步做受控随机数的 PyTorch 采样，也没有独立 PyTorch 最终码对照，因此不据此声称分歧对最终图像无影响。诊断解释了该次前两名翻转的敏感性，未证明所有数值超限都属于 FP32 本底；原功能检查 11/12 和严格检查结论保持不变。

## 已知限制

- 第 23 层半 mask 输出的 6 个近零元素略超逐元素阈值，误差传播到部分 logits；整体 relative L2 仍约 `8e-7`，argmax 和最终生成 token 保持一致。严格结论仍是 78/83 通过，不放宽阈值、不改判。
- 原 12 样例结果来自桥接流程；后续原生队列覆盖 12 个 T2I prompt/seed 组合，但功能检查未全部通过。原生队列的 MMU 始终重复同一张 dog 图，不代表多图理解验证。
- 原生流程 6 次前向抽查中 T2I 第 25/50 步不满足逐元素容差；argmax 一致不保证随机采样结果一致，也不代表其他步或输入均通过。
- 原生 MagViT 的编码一致和解码浮点容差通过，不表述为 PNG 像素逐值相同；生成图尚待视觉审阅。完整模型梯度/训练和正式 FID/CLIP/理解基准尚未完成。
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
- `jittor_magvit.py`、`validate_native_magvit.py`：原生 MagViT 与验证入口；默认运行 256px dog 冒烟测试，`--paired --gpu 0` 模式对 dog/sofa 各执行 256/512px 配对检查，四组已通过。
- `jittor_sampling.py`、`validate_native_sampling.py`：原生 T2I/MMU 采样器及轻量契约检查；三组 T2I（50/6/6 步）和 MMU mask/EOS 检查已通过，原生 categorical/Gumbel 基础检查通过。使用合成 logits、受控采样选择和随机噪声，不代表不同框架相同 seed 会产生相同随机序列。
- `infer_full_jittor.py --native-pipeline`：端到端原生入口，原生 worker 使用 Jittor 完成 Transformer、采样和 MagViT，NumPy/Pillow/tokenizers 完成预处理及提示词；独立 PyTorch 进程检查输入、T2I/MMU 首中末步同输入前向和同图像码解码。已通过上述单样例功能验证，前向抽查逐元素容差为 4/6 通过。
- `run_native_matrix.py`：原生多样例队列，4 个 prompt × 3 个 seed 共 12 个 T2I 样例，逐轮调用已验证入口并汇总到 `native_multi_report.json`。每轮重复相同 dog MMU 输入，不计作不同 MMU 样例；严格容差失败独立记录，功能失败则停止。服务器汇总已核对，功能检查 11/12 通过，末例因前向抽查 argmax 不一致而未通过。
- `diagnose_native_argmax.py`：对保存的某一步 T2I 输入做一次独立 PyTorch 重放，输出 argmax 差异位置、双方 top1/top2 margin 及已保存的 mask/采样状态；不重新生成、不改验收。服务器已复现末例第 25 步的位置 352 差异，确认前两名接近平局及该位置随后重新 mask。
- `export_native_results.py`：将原生队列和单步诊断的小报告、筛选日志及生成/对照 PNG 整理到 `results/summary/native-20261114/`，移除内部路径并生成校验清单、图片总览及 ZIP；不复制权重、数据集或中间张量，不覆盖已有导出目录。已完成本机模拟打包检查，真实结果归档待服务器执行和取回后核验。
- `validate_full_gradients.py`：完整预训练 Transformer 的 MMU/T2I loss 与全部 437 个参数张量的梯度对照；用一张真实 Flickr 训练图片，512px/1024 图像码，独立 PyTorch/Jittor 进程顺序反向，尚未在服务器运行。预先固定梯度 `atol=1e-5, rtol=1e-3`、loss `atol=rtol=1e-4`，所有超限如实记录；不更新原权重、不改变旧前向检查。
- [`configs/full_pilot_protocol.json`](configs/full_pilot_protocol.json)：下一阶段的预设协议，完整 Transformer 小数据微调、32 张训练图/8 张保留验证图、先校准 10 步再评估是否推进到 100 步，两框架使用相同输入顺序和 mask；报告验证子集的 teacher-forced loss/图像码和文字 token 准确率。该协议尚未执行，不计作训练或正式生成质量指标成果。
- `prepare_full_pilot_cpu.py --protocol configs/full_pilot_protocol.json`：不使用 GPU 的数据准备入口，核验上述 32/8 划分及原图哈希，保存 512px 预处理数组、caption token、100 步样例顺序和固定半 mask。服务器已完成 40 张图片的 CPU 准备（`full-pilot-cpu-20261114-204535-041555`）；图像码、最终标签及梯度/训练仍待后续阶段，100 步计划不表示已训练 100 步。产出的数据目录不纳入 Git。
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
python tools/infer_full_jittor.py --native-pipeline --gpu 0
```

完整模型脚本需要官方 Show-o 源码、Phi tokenizer、512 checkpoint 和 MagViT 权重；这些文件均由脚本参数或环境路径提供，不随本仓库分发。

## 来源与许可

本项目基于 [showlab/Show-o](https://github.com/showlab/Show-o)，官方仓库为 Apache License 2.0，许可文本见根目录 [`LICENSE`](LICENSE)。请同时遵守 Show-o、Hugging Face checkpoint、Flickr8k 和 MagViT 数据/模型的各自条款。
