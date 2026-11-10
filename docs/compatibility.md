# Compatibility and validation status

[Documentation home](../README.md) | [Validation](validation.md) | [K3 → K3.1](k3-to-k31.md)

## Pins

| Component | Value | Notes |
|---|---|---|
| Model | `moonshotai/Kimi-K3.1-Pro` @ `d9e99b07` (5 Nov 2026) | 96 shards, 1561 GB, MXFP4 (compressed-tensors) |
| Draft | `RadixArk/Kimi-K3.1-Dflash2-Preview` @ `ea257b38` (6 Nov 2026) | 5.6 GB, `DFlash2DraftModel`, trained block 5 |
| Image | `lmsysorg/sglang@sha256:5901db14…` (= `dev-kimi-k31-pro-mi35x`, 9 Nov 2026) | sglang `de0f4918`, aiter `5405ee00`, sgl-amd-mono `219f40b`, ROCm 10.0.0, Python 3.12 |
| Node | bun159: ROCm 10.1.0, amdgpu 7.1.9, 8 × MI355X (287 GiB), GTT 1133 GiB, MEC fw 0x2c | |

The image tag `dev-kimi-k31-pro-mi35x` moves. Upstream's MI355X AgentX numbers
were measured on digest `5901db14`, which is the one pinned here.

## Checked on Bunya (bun159, 10 Nov 2026)

| Check | Result |
|---|---|
| `gpucheck` | `rocm` mode: 8 devices, gfx950. `devices` mode: aiter cannot read rocminfo (use `rocm`) |
| `check` | registry knows `KimiK31ForConditionalGeneration`; DSA backend pinned to `triton` on ROCm; levers present: monokernel, DFLASH, block verification, ReplaySSM, DCP a2a, runtime reserve, row-sharded prefill |
| `flagcheck` | every flag of all four profiles accepted. `SGLANG_AMD_MONO_K31_WIDE` is inert on this digest (sgl-amd-mono 9cc3c30 made wide the default) |
| Image env | sets `HSA_NO_SCRATCH_RECLAIM=1` and `SGLANG_SET_CPU_AFFINITY=1` itself; the launcher overrides affinity to 0 for the Slurm cgroup |
| CPU tests | `python3 -m unittest discover -s tests`: all pass |

Serving measurements are in [Bunya measurements](#bunya-measurements).

## Upstream verification status (cookbook, Nov 2026)

| Profile | Upstream status |
|---|---|
| lowlatency (TP8 + monokernel) | verified with DFLASH (AgentX c=1, c=4) |
| mono-dcp8 | verified with DFLASH (AgentX c=8) |
| throughput (DCP8 + HiCache) | verified with DFLASH + HiCache L2 (AgentX c=16, c=32) |
| throughput-wide | verified with DFLASH + HiCache L2 (AgentX c=48, c=56) |
| any profile, `SPECULATIVE=none` | in progress (not verified) |

Upstream's AgentX DFLASH acceptance was **simulated** at 3.399. Upstream's
HiCache ratios assume a 1511 GiB GTT. This repo scales them for Bunya, and
adds a TTM shim without which SGLang refuses any HiCache pool on these nodes.

## Bunya measurements

bun159, 10 Nov 2026, the pinned image, weights and draft. Logs are in `logs/`
(`stress-*.log`, `kimik31-server-20261110-*.log`). Bench files are in
`$MODEL_CACHE_DIR/bench-k31/`. Each figure is a single run with no error bars.

### Startup

| Start | Time to ready | Notes |
|---|---|---|
| Cold (first ever) | 5159 s | ~62 min GPFS weight read (I/O-bound: every rank reads all 96 shards), ~20 min one-time FlyDSL/aiter JIT, ~3 min graph capture |
| Warm restart, same node | 311–371 s | JIT cache on scratch and weights in page cache |
| Restart after a HiCache run | ~20 min | HiCache pins ~0.86 TB of host RAM, which evicts part of the weights from page cache |

### Default profile (`lowlatency`, AGENTS=4, DFLASH block 6, monokernel)

KV pool 1,407,104 FP8 tokens at 1M context. The monokernel runs target-verify
for batch 1–4 × 6 tokens. `toolcheck` passes (2.8 s).

**Real agentic text** (`agentstress.py`, one growing session, 20k-token turns,
effort `low`, temperature 1.0):

| Context | DFLASH + monokernel | No speculation | K3 repo baseline (13 Oct, no spec) |
|---|---|---|---|
| 0–60k | **238** tok/s | 75 | 59 |
| 60–120k | **257** tok/s | 75 | 55 |
| 120–160k | **255** tok/s | 74 | 50 |

Real-text DFLASH accept length was 3.3–3.8 (upstream simulated 3.399). TTFT
for each new 20k-token turn was ~1.2 s with prefix-cache hits (cached 143,872
of 164,347 at turn 8). All needles were found and nothing was corrupted, in
both runs. **DFLASH is a ~3.3× win at every context length on real text,** so
it stays the default.

**Random-prompt bench** (`bench-kimik31.sh`, 1024 in / 512 out, n=64).
Acceptance is inflated to 5.6–5.7, so TPOT here overstates real decode:

| Concurrency | Output tok/s | Median TTFT | Median TPOT |
|---|---|---|---|
| 1 | 410 | 114 ms | 2.15 ms |
| 2 | 549 | 134 ms | 3.12 ms |
| 4 | 838 | 158 ms | 4.24 ms |
| 1 @ 100k in (`longcontext`, n=4) | 101 | 5.1 s (cold prefill) | 2.23 ms |

### `throughput` profile (AGENTS=16, DCP8 + HiCache + DFLASH)

It starts only with the TTM shim (see [troubleshooting](troubleshooting.md)).
The device KV pool is 11.15M tokens. HiCache pinned 32.9 GB KV + 74.9 GB KDA
state per rank (~862 GB total) with ratio 1.5 / mamba-host 6. `toolcheck`
passes. Random bench at concurrency 16, n=96: **1536 output tok/s**, median
TTFT 1.17 s, median TPOT 7.39 ms.

### Not yet measured on Bunya

`mono-dcp8` (AGENTS 5–8) and `throughput-wide` (AGENTS 33–56) pass
`flagcheck` but have not been served. The concurrent multi-agent soak at 16+
sessions with real text has not been run. Neither has `LOAD_FORMAT=presharded`;
since the cold start is I/O-bound, presharding is the lever for faster cold
restarts.
