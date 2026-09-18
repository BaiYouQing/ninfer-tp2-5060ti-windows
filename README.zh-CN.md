[English](README.md) | **简体中文**

# NInfer

> 跑在**两张消费级卡**上的 NInfer 张量并行版。在 **2× RTX 5060 Ti（每卡 16 GiB）** 上实测：一份 27B 模型
> 常驻两块卡、**单槽 253,952 token 上下文**、四档 KV cache（`bf16` / `int8` / `fp8` / `k16v8`）、
> MTP3 投机解码且前缀复用真正命中、`/health` 如实反映引擎可用性。

NInfer 是从零写的 C++/CUDA 推理引擎，只支持**显式注册**的 Qwen 系列 checkpoint。它通过本地 CLI 或
OpenAI / Anthropic 兼容的 HTTP 接口处理文本、图像与视频输入；可以在单卡上跑，27B 执行包也可以在**两块**
卡上做张量并行 —— 见[双卡（TP2）](#双卡tp2)。上面这段描述的是**上游**；本 fork 加了什么、实测了什么，
见下面的 fork 说明。

> **这是一个 fork。** 上游是 [Neroued/ninfer](https://github.com/Neroued/ninfer)；本树的起点是上游
> 的 `feaf4dd`，经由 TP2 这条线
> （[wamansou/ninfer-tp2-1m](https://github.com/wamansou/ninfer-tp2-1m)、
> [giocom/ninfer-3060X2](https://github.com/giocom/ninfer-3060X2)）继承而来，在其上为 27B 执行包加了
> **双卡张量并行**（`--tp 2 --devices A,B`：把每卡权重与 KV 常驻减半；一个进程、一份模型、两块卡，
> 不用 NVLink，也不是分布式服务）。
>
> **本 fork 又加了三项**：**KV cache 档位**（`--kv-dtype bf16|int8|fp8|k16v8`）、**在 `--tp 2` 下真正生效
> 的 MTP 前缀复用**、以及**如实反映引擎可用性的 `/health`**（配合 supervisord 自愈）。这三项的实测平台是
> **2× RTX 5060 Ti（每卡 16 GiB）**。`--tp 1` 的贪心输出与 `feaf4dd` 逐字节一致
> （见 [`tests/data/tp1-golden/`](tests/data/tp1-golden/MANIFEST.md)）；单卡行为、支持的 identity、产物格式
> 与协议面均未改变。设计决策与验证证据见
> [Dual-GPU (TP2) execution and YaRN 1M context](docs/maintainer/tp2-yarn-1m.md)（该文档是上游的，也涵盖本
> fork 未使用的 YaRN 扩展位置路径），署名见 [NOTICE](NOTICE)。

## 权重

本 fork 只跑一个模型：**Qwen3.8-27B NVFP4 W4A4**。`.ninfer` 产物不由本仓分发 —— 用仓内转换器从公开的
源权重自行构建。

| 项 | 值 |
|---|---|
| 源（Hugging Face） | [`nerkyor/Qwen3.8-27B-EfficientThink-Uncensored-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-MTP-NVFP4`](https://huggingface.co/nerkyor/Qwen3.8-27B-EfficientThink-Uncensored-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-MTP-NVFP4/tree/main/W4A4) 的 `W4A4/` 子目录（ModelOpt NVFP4，group size 16，`variant = fast`） |
| 源文件 | `model-nvfp4-fast.safetensors`（18,822,252,240 B，SHA-256 `9b7e1c4d839995ee9ed35ac682ecf31e81ae4bc6ddb4e3aaa7f585b786bb83a0`）与 `vision-mtp-bf16.safetensors`（1,770,897,648 B），加索引和 6 个前端资源 |
| 源校验 | 目录里的 `SHA256SUMS`；`manifest.json` 的 `source_package_sha256 = 2d2eac20ceb1439ab85eda4c5d616150f1c1f4956729333cc6e6e15e49739b21` |
| 转换器 | [`tools/convert/qwen3_8_27b/convert_w4a4.py`](tools/convert/qwen3_8_27b/convert_w4a4.py) |
| 结果 | `qwen3_8_27b_nvfp4w4a4.ninfer`，17,555,334,916 B（16.35 GiB），SHA-256 `63c204d223e73d63d6d4db8a82aa3f4859592cd83b00545bcee38334643341cb` |

### 产物是怎么转出来的

```bash
# 1. 取公开 checkpoint 的 W4A4 子目录
hf download nerkyor/Qwen3.8-27B-EfficientThink-Uncensored-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-MTP-NVFP4 \
  --include "W4A4/*" --local-dir src

# 2. 转换
python3 -m tools.convert.qwen3_8_27b.convert_w4a4 \
  --src src/W4A4 --out models/qwen3_8_27b_nvfp4w4a4.ninfer

# 3. 校验：重算每个对象，与产物逐字节比对
python3 -m tools.convert.qwen3_8_27b.convert_w4a4 \
  --src src/W4A4 --verify models/qwen3_8_27b_nvfp4w4a4.ninfer
```

`huggingface.co` 在部分网络下不可达。`huggingface_hub` 认 `HF_ENDPOINT` 环境变量，所以
`HF_ENDPOINT=https://hf-mirror.com hf download …` 可以经镜像取到同样的文件 —— 本 fork 最近一次复现这三步时
就是这么下的。

转换器从 `src/W4A4` 读 **9 个文件**：两个 safetensors 分片、`model.safetensors.index.json`，以及 6 个前端资源
（`tokenizer.json`、`tokenizer_config.json`、`chat_template.jinja`、`generation_config.json`、
`preprocessor_config.json`、`video_preprocessor_config.json`）。那 6 个会被**逐字节**写进产物，缺任何一个
转换立即中止 —— 所以上面 `W4A4/*` 这条 glob 正是该下的：它把转换器需要的全拉下来，除两个分片外只多约 13 MB。

转换器做了什么、以及刻意不做什么：

- **复用引擎自己的编码器**：NVFP4 对象**逐字节重打包**，BF16/FP32 对象直通，W8/Q4/Q5/Q6 端点、MTP 模块与
  视觉塔都过引擎加载时用的同一个编码器重新量化；
- **融合投影按张量并行分片期望的物理行序输出**：`attention/query_key_gate_value` 为 `[Q | K | Gate | V]`
  （query 与 gate 按 head 从 `q_proj` 里取，每 head 256 + 256 行）、`mlp/gate_up` 为 `[gate | up]`、
  `gdn/query_key_value_z` 为 `[qkv | z]`；
- **`gdn/convolution` 是转置而不是 reshape**：源里存成 `(C, 1, K)`，引擎按 `[K, C]` 读；直接 reshape 会
  保留扁平顺序、静默搞坏全部 GDN 层；
- **九层例外层保持 NVFP4**（Qwen3.6 的 NVFP4 recipe 会把它们留成 BF16）：源本来就是 NVFP4，转换器保留源、
  不降精度；
- **优化草稿头是算出来的**：它由一份频率 ranking 导出（`--draft-ranking`，默认
  `tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64`）。ranking 里每个 token 的值是"模型自己在
  72k 段对话上 teacher-forced argmax 出它的次数"；转换器取频率最高的 **131,072 个** token（全词表 248,320）
  加上全部 special token，把 output head 里对应的行切出来构成 `text/draft_head`，id 列表写成
  `text/draft_head_token_ids`。它是算出来的、不是从源里搬的，换一份语料就会得到不同的草稿头。
  **算法和语料都是上游的**：选取与物化在 `tools/convert/qwen3_6/common/draft_head.py`，ranking fixtures
  随上游仓一起分发在 `tools/freq_corpus/`；这一档只是把它们接到 W4A4 的源上。

校验门槛是逐字节比对，本 fork 的产物在 **1310 个张量对象 + 6 个前端资源**上全部通过。上面这三步在本 fork
的机器上从公开源**重跑过一遍**，产出的文件与随本 fork 发布的产物**逐字节相同**（SHA-256 见上表）。完整页面见
[docs/maintainer/qwen3.8-27b-w4a4-artifact.md](docs/maintainer/qwen3.8-27b-w4a4-artifact.md)。

### 引擎还注册了哪些产物

NInfer 刻意只支持一组封闭的产物、不做通用模型运行时。引擎另外注册并接受上游的 identity —— 官方
[`Qwen3.8-27B NVFP4`](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) 产物
（`qwen3_8_27b_nvfp4.ninfer`，21,492,695,040 B，SHA-256
`bb3360522a06e136e0367f5703414d26272b7285c8a6ab6194135c17dbd81b32`）、Qwen3.6-27B 两个档、
Qwen3.8-27B 的 `groupwise-int`、Qwen3.6-35B-A3B —— 只是它们不在本 fork 的构建与实测范围内。有一条实测
值得记下来：**官方 `nvfp4` 产物也在这里的 `--tp 2` 下跑过**，在 2× RTX 5060 Ti 上加载与生成都正常
（每卡权重 10.08 GiB；未开 MTP 时 decode 38 tok/s，开 `--spec mtp --draft-tokens 3 --lm-head-draft`
时 76–86 tok/s）；用 `int8` KV 时**单槽能吃满它 262,144 token 的原生上限**，
每卡还余 842 MiB —— `ninfer-serve` 与 CLI 的余量**完全一样**，因为视觉关闭时媒体与响应缓冲并不预留。
W4A4 形态每卡权重少 16%，同一 `int8` 档下 decode 是 106 tok/s。⇒ 上面的 W4A4 形态是本 fork 想跑的模型，**不是张量并行的硬要求**。当前构建只接受
version-2 容器，上面这些也都是 version 2。

每个 `.ninfer` 文件里含 NInfer 需要的全部权重与前端资源，它不是 Transformers checkpoint、不是 Safetensors
分发、也不是 GGUF。产物本身完整，而 GPU 常驻在进程启动时就已固定：**投机解码默认关闭**（MTP/DFlash 状态
与优化草稿头不上卡），**视觉默认关闭**（权重、Vision scratch 与 request-transient 固定分配都不占）。要接受
图像/视频输入就在 CLI 或服务进程上加 `--vision`，要投机解码就加 `--spec mtp --draft-tokens 3`。被关掉的
能力不能再由后续请求打开。

## 快速开始

克隆**本仓**（不是上游，也不是本仓所继承的 TP2 fork）：

```bash
git clone https://github.com/lynx-gt/ninfer-tp2-5060ti.git
cd ninfer-tp2-5060ti

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

在双卡上以 K16V8 KV cache 起 27B 的 NVFP4 W4A4 产物（253,952 token 单槽，MTP3 投机解码 + 优化草稿头）。
这个产物不由本仓分发 —— 先按上面的 [权重](#权重) 把它转出来并放进 `models/`：

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4w4a4.ninfer \
  --host 0.0.0.0 --port 8815 --model-id qwen3.8-27b-w4a4-mtp3 \
  --tp 2 --devices 0,1 --kv-dtype k16v8 \
  --max-context 253952 --kv-capacity 253952 --prefill-chunk 1024 \
  --spec mtp --draft-tokens 3 --lm-head-draft --max-concurrency 1 --cors
```

环境要求见 [构建要求](#构建要求)，产物转换见
[`docs/maintainer/qwen3.8-27b-w4a4-artifact.md`](docs/maintainer/qwen3.8-27b-w4a4-artifact.md)。

## 用法

**CLI**（双卡单次提问）：

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4w4a4.ninfer \
  --tp 2 --devices 0,1 \
  --kv-dtype k16v8 --max-context 32768 --kv-capacity 32768 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --prompt "用一句话说明潮汐的成因。" --max-new 256
```

多轮对话或带图/视频时改用 `--messages FILE`（JSON 结构见 [CLI 示例](examples/cli/)）；`--no-thinking`
能让较小的 `--max-new` 预算直接进答案通道 —— 这个 checkpoint 默认的思考模式会把短预算全吃在推理流里。
答案写 stdout，加载进度、计时、吞吐、显存与投机解码统计写 stderr。

**HTTP 服务**（OpenAI / Anthropic 兼容，命令见[快速开始](#快速开始)）：

```bash
curl http://127.0.0.1:8815/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b-w4a4-mtp3",
    "messages": [{"role": "user", "content": "用一句话说明潮汐的成因。"}],
    "max_tokens": 64
  }'
```

服务同时实现 OpenAI Responses Core、OpenAI Chat Completions 与 Anthropic Messages（含流式、token 计数与
usage 计量），以及由 prompt 渲染的函数工具；`/health` 报告引擎可用性。`model` 字段默认等于产物的
`identity.model_id`，只有要发布部署专属别名时才需要 `--model-id`。详见 [CLI](docs/cli.md) 与
[HTTP 服务](docs/serving.md)。

## 双卡（TP2）

`--tp 2` 把一份常驻模型分到两块卡上：一个进程、一份模型、两个 CUDA 设备，不用 NVLink，也不是分布式服务。
它是**容量特性**而不是横向扩展 —— 每卡权重与 KV 常驻减半。它实现在 27B 执行包上（`qwen3.6-27b` 与
`qwen3.8-27b`，两个权重档都行）；`qwen3.6-35b-a3b` 没有张量并行路径，`--tp 2` 启动即拒。

本 README 发布的每一个双卡数字，都是在 **2× RTX 5060 Ti（每卡 16 GiB）** 上对[权重](#权重)里那个 W4A4
产物测的。

**用法**见[快速开始](#快速开始)与[用法](#用法)。几条硬约束：

- `--tp 2` 必须显式给 `--devices A,B`，两块卡要同 compute capability；`--tp 1` 仍是默认；
- `--kv-capacity` 必须 ≥ `--max-context`；要把某档顶到它的上限就得用**显式**容量（`auto` 会多留 512 MiB）；
- 16 GiB 卡上 `--max-concurrency 1` 是算术：下一节的档位表就是单槽的成本，`k16v8` 在 253,952 上放不下第二槽；
- `--spec dflash` 与 `--vision` 在 `--tp 2` 下启动即拒；MTP（`--spec mtp --draft-tokens 1..5`，可加
  `--lm-head-draft`）在 `--tp 2` 下工作，**包括前缀复用**；
- `--rope yarn`（本线从上游继承的扩展位置路径）存在但本 fork 未使用：上限 253,952 低于原生 262,144，
  不需要位置缩放，也没有测过。

**已知限制**（与 TP2 和本定版相关）：

- 视觉只在 `--tp 1` 下可用（编码器没有分片路径，`--tp 2 --vision` 启动即拒）；
- **P2P 取决于板子**：上游在两块 5090 上测得 `cudaDeviceCanAccessPeer` 为 0、集合通信遂走 PCIe 主机中转；
  本 fork 实测的两块 5060 Ti 上该值是**双向 1**。无论哪种，一个 decode token 是 128 次 reduce 加一次 logit
  all-gather，CUDA Graph 下整套集合通信相对每轮约 28 ms 的权重读取很小；
- **前缀复用在 `--tp 2` 下可用，但有两种降级**：本 fork 实现了 TP2 路径上的 retained-state 续跑，续写引擎仍
  持有的前缀会命中（`reuse=restore_turn_checkpoint`）；当保留的 base 已覆盖整个 prompt（没有后缀可算）时
  降级为全量 prefill；命中要求"**续写同一个前缀**"，只共享一段更早的公共前缀不算命中。任何情况下答案不变、
  请求不失败；
- MTP 与纯解码"输出等价、但不逐位相同"：verify 一轮按 `K+1` 列算、普通一轮按 1 列算，GEMM shape 不同 ⇒
  贪心流可能在近似并列的 token 上翻转，但每个落地的 token 仍是目标模型自己的 argmax；
- 命中的 prefill 与冷启动可能在最后几位不同（见上一节）；
- FP8 两档 prefill 比 `int8` 慢约 9–14%（KV 暂存路径未上 `cp.async`），decode 不受影响；
- `k16v8` 单槽到不了 262144（BF16 key 每 token 26.2 KiB ⇒ chunk1024 上限 253952、chunk4096 为 229376）；
- `--ignore-eos` 是诊断 flag，不是产品输出路径。

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

## 性能与评测

本 README 的性能数据就是上一节（2× RTX 5060 Ti、`--tp 2`、单槽）。

**能力评测：本 fork 没有预算去跑 benchmark**，所以这里不列任何分数。想看分数去模型仓库 —— 各产物的
model card（`model-cards/`）以及它们在 Hugging Face 上的页面里有上游的成绩（AIME 2025/2026、
GPQA-Diamond、ERQA、RealWorldQA，EvalScope 1.9.0、单样本）。注意那些是在**上游的产物**上测的，
不是本 fork 转换出来的这一份。

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

- 只接受注册的五个 `(model_id, weights_id)` 产物 identity（上表覆盖其中两个）；
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
