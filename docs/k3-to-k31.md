# K3 → K3.1: what changed and what this repo applies

[Documentation home](../README.md) | [Configuration](configuration.md) | [Compatibility](compatibility.md)

## The model

Kimi K3.1 Pro (released 23 Oct 2026, open weights 4 Nov 2026) is post-trained
from Kimi K3 with large-scale joint RL. The checkpoint shape is K3's: 2.8T
parameters with 104B active, 93 layers, 896 experts (top-16 plus 2 shared),
MXFP4 weights trained with QAT and MXFP8 activations, a 1,048,576-token
context, and the MoonViT-V2 vision tower. It is the same 1561 GB on disk. The
differences that matter for serving, read from `config.json` against K3's
`9f62e4e9`:

| | Kimi K3 | Kimi K3.1 Pro |
|---|---|---|
| Architecture | `KimiK3ForConditionalGeneration` | `KimiK31ForConditionalGeneration` |
| 24 full-attention layers | Gated MLA (dense) | **Gated DSA**: DeepSeek Sparse Attention, with a lightning indexer (32 heads × 128, no RoPE, LayerNorm k) that keeps the **top 2048** keys per query |
| KV cache | engine default | **FP8 forced** (`fp8_ds_mla`), page size 64; MLA + indexer ≈ 18.9 KB/token |
| KDA (69 layers) | unchanged | unchanged; FP32 state ≈ 52 MB/request/rank at TP8 |
| Thinking | parser-driven | **always thinks**; depth via `reasoning_effort` = `low` / `high` / `max` (default) |
| Multi-turn contract | content + tool_calls | the **whole** assistant message, `reasoning_content` included, must be replayed |
| Speculative decoding | DSpark draft (`RadixArk/Kimi-K3-DSpark`) | **DFLASH** draft `RadixArk/Kimi-K3.1-Dflash2-Preview`. There is no NextN layer, so a separate draft is the only option. It is a 5-layer sliding-window MLA draft conditioned on 6 target layers' hidden states, with YaRN ×32 to 1M |

DSA is the reason K3.1 is cheaper at long context. The 24 attention layers stop
attending to every cached token and attend to 2048 selected ones. Agent
sessions resend 100k+ tokens every turn, and that was exactly the regime where
the K3 repo was stuck (decode cost at 100k).

## The deployment: K3 repo baseline vs this repo

| Area | K3 repo (13 Oct 2026 baseline) | This repo | Source / reason |
|---|---|---|---|
| Image | `sglang-rocm:v0.5.21-rocm724-mi35x-20261011` | `sglang:dev-kimi-k31-pro-mi35x` **@sha256:5901db14** | The only image with the K3.1 ROCm port. ROCm 10, which matches Bunya's ROCm 10.1 nodes. Pinned by digest because the tag moves |
| Attention | `--attention-backend triton`, `MLA_DECODE_TUNE=1` | none passed; SGLang **pins DSA prefill/decode to `triton` on ROCm** | `_KIMI_K31_DSA_BACKEND_HIP`. The K3 MLA-decode tune does not apply to DSA |
| KV cache | auto | FP8 (forced by K3.1) | `KV_CACHE_DTYPE` accepts only empty/`fp8_e4m3` |
| Speculative | **off**: DSpark lost on real text past ~60k | **DFLASH on** with ReplaySSM and block verification | Upstream verified every MI355X cell *with* DFLASH. See the caveat below |
| Decode kernel | standard | **AMD monokernel** (`--enable-amd-monokernel`, `SGLANG_AMD_MONO_K31_WIDE=1`) for 1–8 agents | Persistent decode/verify kernels from `sgl-amd-mono` |
| Context parallel | DCP8 experimental only | **DCP8 (a2a)** from 5 agents | Splits the KV cache across 8 GPUs; more agents and more context per GPU |
| KV offload | none | **HiCache** host tier (write-back) from 9 agents, **sized to the node's GTT** | Upstream ratios overflow Bunya's 1133 GiB GTT (below) |
| CUDA graphs | decode bs 256 | decode bs = agent count; **prefill graphs** (breakable, 256…4096) | Captures exactly the shapes agent turns use |
| KDA state pool | defaults | 150 slots (1–4 agents) or memory ratio 0.65; `extra_buffer_lazy`; decode-lock skip | Keeps agent prefixes resident |
| Scheduling | default | `lpm` + `--prefill-decode-interval` 16 / 8 | Prefix-aware ordering; caps how often prefill preempts decode |
| Memory | `--mem-fraction-static 0.85` | 0.85 + `SGLANG_ROCM_RUNTIME_RESERVE_GB` 6/10 + `HSA_USERPTR_FOR_PAGED_MEM=0` + HIP allocator GC 0.95 | Leaves HBM for HIP-queue scratch; pins host pools in GTT. Upstream saw 0.87+ overflow |
| aiter | `USE_AITER`, `K3_OPT`, `FLYDSL_FORCE`, `SITUV2_A8W4` | same four | Unchanged K3 FP4 MoE path |
| Loader | 32 threads + presharded (local config) | upstream's `{"enable_multithread_load": true}` + prefetch | K3.1 shards are about 17 GB (K3's were 5 GB), so 32 threads would hold about 560 GB in RAM. Presharded remains available as an A/B |

### Bunya-specific additions in this port

1. **GTT-aware HiCache sizing.** The pinned host pools are GTT memory, and the
   8 ranks share half of node RAM. Upstream sized them against 1511 GiB, which
   would pin about 1.3 TiB. bun159 reports 1133 GiB, so upstream's
   `--hicache-ratio 2.5` / `SGLANG_HICACHE_MAMBA_HOST_RATIO=10` would fail
   `hipHostMalloc` at startup. The launcher reads
   `/sys/class/drm/card*/device/mem_info_gtt_total` and scales both ratios
   (`HICACHE_GTT_BUDGET`, default 0.85). It also warns when the pinned estimate
   exceeds the Slurm `--mem` allowance.
2. **`flagcheck` preflight.** Before a 1.5 TB load, every `--flag` of the
   resolved recipe is checked against the image's `--help`, and every
   `SGLANG_*` / `AITER_*` variable against the image's source tree. This
   generalises the K3 repo's per-flag probes. Day-0 images rename flags; the
   K3 repo hit this with `--enable-gdn-replayssm-spec`.
3. **Shard-completeness check.** Serve counts resolved `.safetensors` against
   the index, which handles `huggingface_hub` 1.x's hub-wide xet blob store.
   Serving uses `HF_HUB_OFFLINE=1` and a resolved snapshot path.
4. **`toolcheck` and `agentstress.py` replay `reasoning_content`** and use
   K3.1's recommended sampling (temperature 1.0, top_p 0.95). An agent client
   that drops replayed reasoning is the likeliest K3.1-specific failure.
5. **TTM shim for HiCache.** SGLang budgets HiCache from
   `/sys/module/ttm/parameters/pages_limit`. That reads 0 on Bunya, so the
   budget goes negative and every HiCache launch would fail after the weight
   load. The launcher binds the node's real GTT page count over it.
6. **Monokernel guard.** SGLang's monokernel only runs speculative verify steps
   (DFLASH or DSpark, ReplaySSM, 6 or 8 draft tokens). With `SPECULATIVE=none`
   or another block size the launcher turns it off instead of letting SGLang
   refuse to start.
7. **`reasoning_effort: "medium"` remap.** The pinned image treats medium as a
   native K3.1 tier, but the pinned checkpoint's tokenizer only accepts
   low/high/max and asserts, so clients that default to medium got HTTP 500
   on every turn. `shims/sitecustomize.py` (bound in by the launcher) maps
   medium to high, as SGLang does for K3 (`MEDIUM_EFFORT`).
8. **`status` reports** whether the monokernel engaged (`[k31 wide] first wide
   step`) and the recent DFLASH accept length.
9. Caches, logs, PID and API key use `k31` names, so K3 and K3.1 can share one
   `MODEL_CACHE_DIR`. The sbatch wrapper uses `SLURM_SUBMIT_DIR` (the K3
   wrapper's spool-directory caveat is fixed).

### Caveat carried over from K3: speculative decoding on real text

The K3 repo found DSpark accepted ~7 tokens on bench prompts but ~2 on real
agent text, which made it a net loss past ~60k context. Upstream's MI355X
AgentX numbers for K3.1 use a **simulated** acceptance of 3.399
(`SGLANG_SIMULATE_ACC_LEN`), not measured acceptance. DFLASH is on by default
because it is upstream's verified configuration. This repo then measured it on Bunya.
On a real-text agent session to 160k, DFLASH decoded at 238–257 tok/s against
74–75 without speculation, with accept length 3.3–3.8 (see
[measurements](compatibility.md#bunya-measurements)). Unlike DSpark on K3, it wins
at every context length. Re-run the
[A/B](validation.md#speculative-decoding-ab) if you change the draft or image.

## Not applied (and why)

- **PD disaggregation (1P1D, >56 agents).** It needs two MI355X nodes and
  Mooncake over RDMA. It is out of scope for a single-node launcher.
- **HiCache L3 (Mooncake)** has not run on MI355X upstream.
- **FlashMLA / TRT-LLM / Prims-TS DSA kernels and MegaMoE** are NVIDIA-only.
- **HiSparse** is rejected on ROCm by SGLang.
