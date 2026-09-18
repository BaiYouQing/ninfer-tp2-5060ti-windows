[English](README.md) | **简体中文**

# NInfer

> 跑在**两张消费级卡**上的 NInfer 张量并行版。在 **2× RTX 5060 Ti（每卡 16 GiB）** 上实测：一份 27B 模型
> 常驻两块卡、**单槽 253,952 token 上下文**、四档 KV cache（`bf16` / `int8` / `fp8` / `k16v8`）、
> MTP3 投机解码且前缀复用真正命中、`/health` 如实反映引擎可用性。本树继承的上游单卡 RTX 5090 工作
> （含 YaRN 的 1,048,576 token 路径）属于上游；正文里哪张表来自哪台机器都有标注。

NInfer 是从零写的 C++/CUDA 推理引擎，只支持**显式注册**的 Qwen 系列 checkpoint。它面向一块
NVIDIA GeForce RTX 5090，通过本地 CLI 或 OpenAI / Anthropic 兼容的 HTTP 接口处理文本、图像与
视频输入。27B 执行包另外支持在**两块** RTX 5090 上做张量并行，并可用 YaRN 位置缩放把上下文
拉到 1,048,576 token。上面这段描述的是**上游**；本 fork 加了什么、实测了什么，见下面的 fork 说明。

> **这是一个 fork。** 上游是 [Neroued/ninfer](https://github.com/Neroued/ninfer)；本树的起点是上游
> 的 `feaf4dd`，经由 TP2 这条线
> （[wamansou/ninfer-tp2-1m](https://github.com/wamansou/ninfer-tp2-1m)、
> [giocom/ninfer-3060X2](https://github.com/giocom/ninfer-3060X2)）继承而来，在其上为 27B 执行包加了
> 两件事：**双卡张量并行**（`--tp 2 --devices A,B`，把每卡权重与 KV 常驻减半，长上下文下约快 40%；
> 一个进程、一份模型、两块卡，不用 NVLink，也不是分布式服务），以及 **YaRN ×4 位置缩放**
> （`--rope yarn`，把注册的原生 262,144 token 上限提到 1,048,576）。
>
> **本 fork 又加了三项**：**KV cache 档位**（`--kv-dtype bf16|int8|fp8|k16v8`）、**在 `--tp 2` 下真正生效
> 的 MTP 前缀复用**、以及**如实反映引擎可用性的 `/health`**（配合 supervisord 自愈）。这三项的实测平台是
> **2× RTX 5060 Ti（每卡 16 GiB）**。`--tp 1` 的贪心输出与 `feaf4dd` 逐字节一致
> （见 [`tests/data/tp1-golden/`](tests/data/tp1-golden/MANIFEST.md)）；单卡行为、支持的 identity、产物格式
> 与协议面均未改变。设计决策与验证证据见
> [Dual-GPU (TP2) execution and YaRN 1M context](docs/maintainer/tp2-yarn-1m.md)，署名见 [NOTICE](NOTICE)。

NInfer 刻意只支持一组封闭的模型产物，而不是做一个通用模型运行时：

| 模型 | 权重档 | NInfer 产物 | 大小 | SHA-256 |
|---|---|---|---:|---|
| [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | `groupwise-int` | `qwen3_6_27b.ninfer` | 17,495,365,888 B（16.29 GiB） | `7b51600ffd10632b9660f56085efdd9b751d79733ad32036a652234b64bebe7b` |
| [Qwen3.6-27B NVFP4](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) | `nvfp4` | `qwen3_6_27b_nvfp4.ninfer` | **18,324,064,000 B（17.07 GiB）** | `bce5f00d066c0f20f1317bf1fdcb458264cf95837c3b1f3fbec163694627893a` |
| [Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | `groupwise-int` | `qwen3_8_27b.ninfer` | 18,210,531,328 B（16.96 GiB） | `eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e` |
| [Qwen3.8-27B NVFP4](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | `nvfp4` | `qwen3_8_27b_nvfp4.ninfer` | **21,492,695,040 B（20.02 GiB）** | `bb3360522a06e136e0367f5703414d26272b7285c8a6ab6194135c17dbd81b32` |
| [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | `groupwise-int` | `qwen3_6_35b_a3b.ninfer` | 22,783,246,080 B（21.22 GiB） | `1fb9ea0b5b8561e49d9604115ec89e5d9f2b6f6434e32c37c57fffd480a325d2` |

Qwen3.6-27B 与 Qwen3.8-27B 各有两个注册权重档。version-2 产物 identity 会自己决定用哪个档，不需要额外的
运行时开关。Qwen3.8-27B 的 `nvfp4` 档是**混合**量化：Text 0–55 层的 MLP 用 NVFP4，token embedding、
attention 输入/输出投影、GDN 的 Q/K/V/Z 与输出投影、output head 以及其余 MLP 权重用行标度 FP8。

## 快速开始

克隆**本仓**（不是上游，也不是本仓所继承的 TP2 fork）：

```bash
git clone https://github.com/lynx-gt/ninfer-tp2-5060ti.git
cd ninfer-tp2-5060ti

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

在双卡上以 K16V8 KV cache 起 27B 的 NVFP4 W4A4 产物（253,952 token 单槽，MTP3 投机解码 + 优化草稿头）。
这个产物**不由本仓分发** —— 先按 [下载模型](#下载模型) 把它准备好并放进 `models/`：

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4w4a4.ninfer \
  --host 0.0.0.0 --port 8815 --model-id qwen3.8-27b-w4a4-mtp3 \
  --tp 2 --devices 0,1 --kv-dtype k16v8 \
  --max-context 253952 --kv-capacity 253952 --prefill-chunk 1024 \
  --spec mtp --draft-tokens 3 --lm-head-draft --max-concurrency 1 --cors
```

环境要求见 [构建要求](#构建要求)，产物转换见
[`docs/maintainer/qwen3.8-27b-w4a4-artifact.md`](docs/maintainer/qwen3.8-27b-w4a4-artifact.md)。

## 下载模型

注册的产物用 Hugging Face CLI 下载（上游已发布的五个 identity 本仓同样支持）：

```bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir models
```

每个 `.ninfer` 文件里就含 NInfer 需要的全部权重与前端资源，它不是 Transformers checkpoint、不是
Safetensors 分发、也不是 GGUF。另外，**投机解码默认关闭**（MTP 状态与优化草稿头不上卡），**视觉默认关闭**
（权重、Vision scratch 与 request-transient 分配都不占）；要接受图像/视频输入就在 CLI 或服务进程上加
`--vision`。

### 本 fork 验证用的 W4A4 产物

本 README 里的 TP2 数据，以及本 fork 新增的 KV 档位、前缀复用与 `/health` 工作，都是在下面这个
**Qwen3.8-27B NVFP4 W4A4** 产物上测的，本仓同样不分发它：

| 产物 | 来源 | 获取方式 |
|---|---|---|
| `qwen3_8_27b_nvfp4w4a4.ninfer`（NVFP4 **W4A4**；权重 4 bit、激活也 4 bit；**16.35 GiB**） | ModelScope 上的合并微调模型 `Merkyor/Qwen3.8-27B-EfficientThink-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-MTP-NVFP4`（W4A4 版），ModelOpt NVFP4 量化、group size 16 | **本仓不分发**：用 `tools/convert/qwen3_8_27b/convert_w4a4.py` 自行转换；源布局、完整命令与验证门槛见 [docs/maintainer/qwen3.8-27b-w4a4-artifact.md](docs/maintainer/qwen3.8-27b-w4a4-artifact.md) |

本 fork 的 TP2 路径有一条专属注意：上游那个 `qwen3.8-27b/nvfp4` 产物**没有**在 `--tp 2` 上验证过
（它的 BF16 例外层过不了列并行的融合权重绑定），所以要用上面这个 W4A4 形态。

## KV cache 档位与长上下文上限

`--kv-dtype` 按 K/V 两侧分别选编码：`bf16`（不量化）、`int8`（每 64 维一个 fp16 scale）、
`fp8`（e4m3，每 256 维一个 fp16 scale）、`k16v8`（BF16 的 key + FP8 的 value）。所有档位的 QK 计算都在
BF16 上做，档位只改变常驻占用与读回路径。

实测（2× RTX 5060 Ti，TP2，单槽，8k token prompt，`--spec mtp --draft-tokens 3 --lm-head-draft
--prefill-chunk 1024`）：

| `--kv-dtype` | prefill | decode | MTP 接受长度 | 单槽上下文上限 |
|---|---|---|---|---|
| `int8` | 4880 tok/s | 106.5 tok/s | 3.04 tok/轮 | 262144 |
| `fp8` | 4170 tok/s | 91.4 tok/s | 2.69 tok/轮 | 262144 |
| `k16v8` | 4480 tok/s | 88.2 tok/s | 2.56 tok/轮 | 253952（chunk 1024）/ 229376（chunk 4096） |
| `bf16` | — | — | 2.89 tok/轮 | — |

各档的**每轮耗时基本相同**（28.5–29.4 ms），token 速率差异来自投机解码达到的接受长度。`--kv-capacity`
必须 ≥ `--max-context`；要把某档顶到它的上限就必须用**显式**容量（`auto` 会预留 512 MiB 的 sizing
headroom，`k16v8` 在 253952 上留不起）。

长上下文（k16v8，单槽，253952）：57.7k token 的 prompt 用 30.4 s prefill 完（1905 tok/s），192.6k token 的
用 287 s（672 tok/s），中段 needle 答对，每卡常驻稳定 15.7 GiB —— prefill 是 O(T²) 所以会变慢，decode 不受影响
（82–96 tok/s）。

前缀复用命中时，日志会报 `cache=7872 reuse=restore_turn_checkpoint`，首 token 时间从 1788 ms 降到 71 ms。
命中的 prefill 是从 checkpoint frontier 往后续算的，不走完整的 prefill chunk 网格，所以**命中与冷启动可能在
最后几位有差异**（贪心文本偶尔也会翻）—— 这与 vLLM/SGLang 前缀缓存是同一类注意事项。命中数会回传到响应的
usage 里：OpenAI 侧是 `usage.prompt_tokens_details.cached_tokens`，Anthropic 侧是
`usage.cache_read_input_tokens`（`cache_creation_input_tokens` 报 0）。

`/health` 如实反映引擎可用性（`200 {"status":"ok"}` / `503 {"status":"unavailable"}`）；引擎进入不可用状态时
`apps/ninfer-serve` 会以非零码退出，让 supervisor（`Restart=on-failure`）在约 16 s 内重新加载模型，而不是留一个
死掉的服务端口在那里。

## 构建要求

- 64 位 Linux；
- 一块 NVIDIA GeForce RTX 5090（`sm_120a`），或两块用于 `--tp 2`（本 fork 另在 2× RTX 5060 Ti 上实测）；
- NVIDIA 驱动支持 CUDA 13.1，且 CUDA Toolkit 为 13.1 或更新；
- CMake 3.28 或更新，以及支持 C++20 的 host 编译器；
- `pkg-config`；
- FFmpeg 开发库：`libavformat >= 60`、`libavcodec >= 60`、`libavutil >= 58`、`libswscale >= 7`；
- `libcurl >= 7.85`；
- Ninja（用下面的命令时需要）。

构建只接受 `120a` 这一种 CUDA 架构，没有 install target，也不发布二进制包 —— NInfer 就在源码构建树里跑。

```bash
git clone https://github.com/lynx-gt/ninfer-tp2-5060ti.git
cd ninfer-tp2-5060ti

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

默认构建产出两个东西：

```text
build/apps/ninfer
build/apps/ninfer-serve
```

测试、benchmark 与维护工具不在默认构建里。

## 与上游的关系

本 fork 从 `Neroued/ninfer` 的 `feaf4dd`（2026-08-20）经 TP2 线继承而来，并在该基底上带着自己的工作。
上游 `master` 从那之后已前进 200 多个提交，且重构了运行时（executor 与 KV sizing 换了文件，KV cache
子系统被重写）。因此两棵树**不可互换**：把 `master` 合进这条线会在数百个文件上冲突，所以本 fork 只跟自己的
基底，按需 cherry-pick 上游修复，而不跟随 `master`。GitHub 会显示本分支同时"领先且落后"`Neroued:master`
—— 那是这条线的常态，不是没人维护。

| | 本 fork | 上游 `master` |
|---|---|---|
| `--kv-dtype` | `bf16`、`int8`、`fp8`、**`k16v8`**（BF16 key + FP8 value） | `bf16`、`int8`、`fp8`、`nvfp4`、`k8v4` |
| 张量并行 | `--tp 2 --devices A,B`，已在 2× RTX 5090 与 2× RTX 5060 Ti 上验证 | 单卡 |
| `/health` | 反映引擎可用性，并在引擎挂掉时由 supervisor 拉起 | 反映引擎可用性 |

想要 `nvfp4` / `k8v4` 档，或上游最新的单卡调度工作，用上游。想要在两张消费级卡上做张量并行服务、
要 `k16v8` 档、要真正命中的 MTP 前缀复用，就用本 fork。

## 能力与限制

**能力。** 三个注册 model ID 都支持：

- 带思考 / 不带思考两种 prompt 模式的文本生成；
- 图像、多图、视频与混合多模态消息；
- 分块 prefill 与 CUDA Graph 解码；
- 启动时固定规模的小并发服务，真批处理解码；
- MTP 投机解码，草稿窗口 1–5；
- KV cache 档位 `bf16`、`int8`（group-64）、`fp8`（e4m3）、`k16v8`（BF16 key + FP8 value）；
- 模型与思考模式感知的官方采样默认值，以及显式的 greedy / temperature / top-k / top-p / min-p /
  presence、frequency penalty 覆盖；
- 兼容前缀复用，含 `--tp 2` 下的 MTP，命中数报在 `usage.prompt_tokens_details.cached_tokens`（OpenAI）/
  `usage.cache_read_input_tokens`（Anthropic）；
- `/health` 反映引擎可用性；引擎不可用时服务进程以非零码退出，交给 supervisor 重启；
- OpenAI Responses Core、OpenAI Chat Completions、Anthropic Messages，含流式与 usage 计量；
- 由 prompt 渲染的函数工具与 tool call 解析。

35B-A3B 另外支持纯文本的 DFlash 投机解码，草稿窗口 1–15。

**限制。**

- 只接受上面列出的五个 `(model_id, weights_id)` 产物 identity；
- 执行专门面向 RTX 5090。默认一个 CUDA 设备；27B 执行包也支持正好两个（`--tp 2 --devices A,B`），
  这是容量特性而不是横向扩展。本 fork 增加了 2× RTX 5060 Ti（16 GiB）上的实测；构建目标两边都是 `sm_120a`；
- 一个 Engine 持有一份常驻模型，启动时固定 1–8 个并发请求容量；decode-ready 的请求在轮边界被压缩进一次
  批处理前向；
- 没有大规模 / 抢占式连续批处理、没有优先级与 QoS 调度、没有 CPU/GPU offload、也不是分布式服务；
- `--vision` 只在 `--tp 1` 下可用（视觉编码器没有分片路径，`--tp 2 --vision` 启动即拒）；`--spec dflash`
  在 `--tp 2` 下同样被拒；
- **`k16v8` 单槽到不了 262144**：它的 BF16 key 每 token 要 26.2 KiB；
- **FP8 两档的 prefill 比 `int8` 慢约 9–14%**（FP8 的 KV 暂存路径还没做成异步），decode 不受影响；
- 前缀复用命中要求**续写同一个前缀**，不是"有公共前缀"：只共享一段更早的公共前缀、但不是引擎保留的那一段，
  不会命中。命中的 prefill 与冷启动可能在最后几位不同（见上一节）；
- C++ 头文件供仓内应用使用，不作为已安装的 SDK 分发。

## 文档

- [文档索引](docs/README.md)
- [CLI](docs/cli.md)
- [HTTP 服务](docs/serving.md)
- [性能](docs/performance.md)
- [Qwen3.8-27B W4A4 产物（本 fork 验证用的形态）](docs/maintainer/qwen3.8-27b-w4a4-artifact.md)
- [双卡 TP2 执行与 YaRN 1M 上下文](docs/maintainer/tp2-yarn-1m.md)
- [CLI 示例](examples/cli/)

## 许可证

NInfer 采用 [Apache License 2.0](LICENSE)。本 fork 的修改沿用同一许可；Apache-2.0 §4(b) 要求的署名见
[NOTICE](NOTICE)。

已发布的产物派生自 [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B)、
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) 与
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)；Qwen3.6-27B NVFP4 产物另外用了
[rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm)
的打包权重，Qwen3.8-27B NVFP4 产物另外用了
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) 的混合 FP8/NVFP4 权重。
这些源仓以 Apache-2.0 分发。vendored 依赖各自保留 `third_party/` 下的许可证文件。
