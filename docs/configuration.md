# Configuration reference

[Documentation home](../README.md) | [Runbook](runbook.md) | [K3 → K3.1](k3-to-k31.md)

Values come from `kimik31.env` (or `$KIMIK31_ENV`). Exported variables win.
**An empty engine knob means "the selected profile's upstream value".** Run
`./serve-kimik31.sh plan` to see the resolved environment and argv. It needs
no GPU and no Apptainer.

## Recipe selection

| Variable | Default | Meaning |
|---|---|---|
| `AGENTS` | 4 | Concurrent agent sessions; selects the profile when `PROFILE=auto` |
| `PROFILE` | auto | `lowlatency`, `mono-dcp8`, `throughput`, `throughput-wide` |
| `SPECULATIVE` | dflash | `dflash` or `none` (non-spec is not verified upstream on MI355X) |
| `DFLASH_BLOCK_SIZE` | profile (6, or 5 wide) | Draft block (verifies block−1 tokens) |
| `BLOCK_VERIFY` | 1 | `--speculative-use-block-verification` (sampled drafts; upstream's single-node acceptance used it) |
| `MONOKERNEL` | auto | Forces the AMD decode monokernel on/off. It needs DFLASH with block 6 or 8 and no HiCache; `auto` turns it off otherwise |
| `HICACHE` | auto | Forces the host KV tier on/off |

### What each profile sets

| | lowlatency (1 / 2–4) | mono-dcp8 | throughput | throughput-wide |
|---|---|---|---|---|
| DCP | – | 8, a2a | 8, a2a | 8, a2a |
| monokernel | yes | yes | – | – |
| HiCache | – | – | yes | yes |
| `--max-running-requests` | 1 / 4 | 8 | max(16, AGENTS) | 48 / 64 (>48) |
| `--chunked-prefill-size` | 16384 | 16384 | 16384 | 32768 |
| `--prefill-decode-interval` | – / 16 | 16 | 16 | 8 |
| KDA pool | `--max-mamba-cache-size` 60 / 150 | ratio 0.65 | ratio 0.65 | ratio 0.65 |
| DFLASH block | 6 | 6 | 6 | 5 |
| runtime reserve | 6 GB | 10 GB | 10 GB | 10 GB |
| extra env | `SGLANG_AMD_MONO_K31_WIDE` | same | – | row-sharded prefill, fused ring locs |

Common to all profiles: `--mem-fraction-static 0.85`, `--schedule-policy lpm`,
`extra_buffer_lazy`, breakable prefill graphs at 256–4096, DFLASH with the
Triton draft attention, FP8 draft KV and ReplaySSM, `kimi_k3` parsers,
`--enable-metrics`, `--stream-interval 1`, prefetch and multithreaded loading.
The environment always includes the four aiter K3 variables,
`SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1`, `HSA_USERPTR_FOR_PAGED_MEM=0` and
`PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.95`.

## HiCache

The host pools are pinned GTT memory, and the 8 ranks share the node's GTT
limit (half of RAM). Upstream's ratios (`--hicache-ratio 2.5`,
`SGLANG_HICACHE_MAMBA_HOST_RATIO=10`) pinned 170–185 GB per GPU against a
1511 GiB GTT. Bunya's MI355X nodes have 1133 GiB, so those ratios would fail at
startup.

| Variable | Default | Meaning |
|---|---|---|
| `HICACHE_GTT_BUDGET` | 0.85 | Ratios scale by `GTT/1511 × budget` (capped at 1) |
| `HICACHE_RATIO` | scaled (1.5 on bun159) | Explicit `--hicache-ratio` |
| `HICACHE_MAMBA_HOST_RATIO` | scaled (6 on bun159) | Explicit KDA host-pool ratio |
| `TTM_SHIM` | auto | Binds the real GTT page limit over `/sys/module/ttm/parameters/pages_limit` when the host reports 0. SGLang reads that file for its HiCache budget; without the shim every HiCache launch fails on Bunya |
| `MEDIUM_EFFORT` | auto | Remaps `reasoning_effort: "medium"` to `high`. The image forwards medium to the tokenizer as a K3.1 tier, but the pinned checkpoint only accepts `low`/`high`/`max`, so clients that default to medium (most agent clients) get HTTP 500. `auto` turns it off when the checkpoint gains a medium tier; `low`/`high`/`max` choose the target; `off` passes medium through |

The launcher prints an estimate of pinned GiB and warns when it approaches the
GTT limit or the job's `--mem`. If startup fails in `hipHostMalloc`, lower the
budget.

## Engine overrides (empty = profile value)

`CONTEXT_LEN`, `MEM_FRACTION`, `MAX_RUNNING_REQUESTS`, `CUDA_GRAPH_MAX_BS_DECODE`
(defaults to `MAX_RUNNING_REQUESTS`), `CHUNKED_PREFILL_SIZE`,
`PREFILL_DECODE_INTERVAL`, `SCHEDULE_POLICY`, `MAX_MAMBA_CACHE_SIZE`,
`MAMBA_FULL_MEMORY_RATIO`, `MAMBA_RADIX_STRATEGY`, `ROCM_RUNTIME_RESERVE_GB`.

- `KV_CACHE_DTYPE` must be empty or `fp8_e4m3`. K3.1 rejects everything else.
- `DISABLE_RADIX_CACHE=1` turns prefix caching off. Agent clients lose most of
  their throughput without it.
- `HSA_NO_SCRATCH_RECLAIM`: the pinned image already sets it to `1` in its own
  environment. That is the setting Bunya's MEC firmware 0x2c (<177) wants
  against RCCL reclaim races (an open question in the K3 repo). Set it only to
  override the image.
- `EXTRA_ENGINE_ARGS` is appended to argv. `EXTRA_ENV="A=1 B=2"` is passed into
  the container. Both are checked by `flagcheck`.

## Pins

| Variable | Default |
|---|---|
| `MODEL_ID` / `MODEL_REVISION` | `moonshotai/Kimi-K3.1-Pro` @ `d9e99b0792a5e93b78ea9bea62a68167e36ddd9e` |
| `DFLASH_MODEL` / `DFLASH_REVISION` | `RadixArk/Kimi-K3.1-Dflash2-Preview` @ `ea257b38a8ee755b65287bb8ccd0938b519ce26b` |
| `SGLANG_IMAGE` | `docker://lmsysorg/sglang@sha256:5901db1401ad7b8bc62ac67175f6f013b1b868e2a2dab063ee8848e7290fa9a7` |
| `SERVED_MODEL_NAME` | `kimi-k3.1-pro` |

## Plumbing (unchanged from the K3 repo)

`ROCM_MODE` (auto/rocm/devices), `ROCMINFO_SHIM`, `AITER_GPU_ARCHS`,
`ENABLE_AITER`, `FLYDSL_FORCE`, `SET_CPU_AFFINITY=0`, `READY_TIMEOUT`,
`AITER_JIT_DIR` / `AITER_JIT_TARGET`, `AITER_CONFIG_DIR`, `SGLANG_CACHE_DIR`,
`APPTAINER_CACHEDIR` / `APPTAINER_TMPDIR`, `LAUNCH_CMD`.

## Weight loading

`WEIGHT_LOAD_THREADS` (empty = upstream's multithread default; K3.1 shards are
~17 GB each, so every thread holds that much host RAM), `LOAD_FORMAT`
(`presharded` is an opt-in A/B that costs one slow load plus up to 1.5 TB),
`PRESHARDED_PATH`, `PREFETCH_CHECKPOINTS`, `PREFETCH_BLOCK_SIZE_MB`.
