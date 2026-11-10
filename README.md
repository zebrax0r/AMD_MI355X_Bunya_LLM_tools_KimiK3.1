# Kimi K3.1 Pro on Bunya MI355X

Serve [moonshotai/Kimi-K3.1-Pro](https://huggingface.co/moonshotai/Kimi-K3.1-Pro)
with SGLang in Apptainer on one Bunya MI355X node (8x gfx950), and connect
opencode or Kimi Code to its OpenAI-compatible endpoint.

This repo ports [AMD_MI355X_Bunya_LLM_tools_KimiK3](https://github.com/zebrax0r/AMD_MI355X_Bunya_LLM_tools_KimiK3)
to Kimi K3.1 Pro. It keeps all of that repo's Bunya plumbing (GPU passthrough
probing, the rocminfo shim, the aiter JIT and `/tmp/aiter_configs` binds,
scratch-only caches, the API key, detach). The engine recipe is replaced with
upstream's **verified MI355X K3.1 recipes**, plus Bunya-specific sizing.

## Start here

| Task | Guide |
|---|---|
| Install, allocate, download, launch, verify, stop | [Operating runbook](docs/runbook.md) |
| Pick a recipe, understand a knob | [Configuration](docs/configuration.md) |
| What changed from K3, and why | [K3 → K3.1](docs/k3-to-k31.md) |
| Pins, verification status, Bunya measurements | [Compatibility](docs/compatibility.md) |
| Connect opencode or Kimi Code | [Clients](docs/clients.md) |
| A launch or request failed | [Troubleshooting](docs/troubleshooting.md) |
| CPU tests, benchmarks, soak | [Validation](docs/validation.md) |

## Quick start (on an allocated MI355X node)

```bash
cp -n kimik31-env.example kimik31.env && chmod 600 kimik31.env   # set MODEL_CACHE_DIR, HF_TOKEN_FILE
./serve-kimik31.sh pull        # ~14 GB image, pinned by digest
./serve-kimik31.sh gpucheck    # container reaches the GPUs?
./serve-kimik31.sh check       # image knows KimiK31ForConditionalGeneration?
./serve-kimik31.sh flagcheck   # image accepts every flag of the selected recipe?
./serve-kimik31.sh download    # ~1.56 TB weights + 5.6 GB DFLASH draft
./serve-kimik31.sh serve --detach
./serve-kimik31.sh toolcheck
```

## Recipes

Choose the recipe by **how many agent sessions you will run at once**
(`AGENTS`, default 4). Every row is an upstream-verified MI355X cell, with DFLASH
speculative decoding on.

| `AGENTS` | Profile | Layout | Key levers |
|---|---|---|---|
| 1–4 | `lowlatency` | TP8 | AMD decode **monokernel**, DFLASH block 6, 150 KDA state slots |
| 5–8 | `mono-dcp8` | TP8·DCP8 | monokernel + decode context parallel |
| 9–32 | `throughput` | TP8·DCP8 | **HiCache** host tier (sized to Bunya's GTT), DFLASH block 6 |
| 33–56 | `throughput-wide` | TP8·DCP8 | HiCache, 32k prefill chunks, DFLASH block 5, row-sharded prefill |

`./serve-kimik31.sh plan` prints the exact environment and argv without
touching a GPU. See [configuration](docs/configuration.md) for overrides.

## Before starting

You need `gpu_rocm` access with an account/QoS that grants 8 MI355X GPUs,
Apptainer (compute nodes only), and about **1.7 TB of scratch**: 1561 GB of
weights, a ~30 GB image and a 5.6 GB draft. HiCache profiles also pin up to
about 0.8 TB of **host** RAM. Check your quota with `rquota`.

## Layout

```
serve-kimik31.sh       launcher (serve, plan, pull, check, gpucheck, flagcheck,
                       toolcheck, parsers, loadstat, download, stop, status)
serve-kimik31.sbatch   batch wrapper
bench-kimik31.sh       sglang bench client (sweep/latency/throughput/longcontext)
agentstress.py         growing-conversation agentic stress test
opencode-setup.sh, kimicode-setup.sh  client configuration
tests/                 CPU-only checks: python3 -m unittest discover -s tests -v
```
