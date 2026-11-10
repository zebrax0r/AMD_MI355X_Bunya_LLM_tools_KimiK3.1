#!/usr/bin/env bash
#
# serve-kimik31.sh — one-click Kimi K3.1 Pro serving on Bunya's AMD MI355X nodes
# (bun159/160/161, 8x gfx950) via SGLang inside an Apptainer container.
#
# Port of serve-kimik3.sh (github.com/zebrax0r/AMD_MI355X_Bunya_LLM_tools_KimiK3)
# to moonshotai/Kimi-K3.1-Pro. The Bunya plumbing (GPU passthrough probing,
# rocminfo shim, aiter JIT / config binds, scratch caches, API key, detach) is
# carried over unchanged in behaviour. What is new is the engine recipe: it is
# SGLang's verified MI355X Kimi-K3.1-Pro AgentX recipe (cookbook, Nov 2026),
# selected by how many concurrent agents you intend to run:
#
#   AGENTS   profile          layout                 notes
#   1-4      lowlatency       TP8      + monokernel  DFLASH block 6
#   5-8      mono-dcp8        TP8·DCP8 + monokernel  DFLASH block 6
#   9-32     throughput       TP8·DCP8 + HiCache L2  DFLASH block 6
#   33-56    throughput-wide  TP8·DCP8 + HiCache L2  DFLASH block 5, row-sharded prefill
#
# Deliberate deviations from upstream, all Bunya-specific:
#   1. binds 0.0.0.0 with --api-key and --served-model-name (tunnels/off-node use)
#   2. SGLANG_SET_CPU_AFFINITY=0 (SLURM cgroup owns a subset of cores)
#   3. HiCache host pools are sized from the node's REAL GTT limit (1133 GiB on
#      bun159) instead of upstream's 1511 GiB — upstream's ratios would pin
#      ~1.4 TB and fail hipHostMalloc at startup here.
#   4. image pinned by DIGEST, weights and draft pinned by commit.
#
# Usage:
#   ./serve-kimik31.sh [serve]    start the server (default; runs until killed)
#   ./serve-kimik31.sh serve --detach
#                                 start, wait until healthy, return the shell
#   ./serve-kimik31.sh plan       print the resolved recipe + argv (no GPU/apptainer)
#   ./serve-kimik31.sh pull       build the .sif from the container image (once)
#   ./serve-kimik31.sh check      can this image load this model? + K3.1 levers
#   ./serve-kimik31.sh gpucheck   can this image reach this node's GPUs? (~1 min)
#   ./serve-kimik31.sh flagcheck  does the image accept every flag of this recipe?
#   ./serve-kimik31.sh toolcheck  two-turn tool-call round trip vs a running server
#   ./serve-kimik31.sh parsers    list tool-call/reasoning parsers this image has
#   ./serve-kimik31.sh loadstat   why the last cold start was slow (reads the log)
#   ./serve-kimik31.sh download   prefetch model + draft weights (no GPU needed)
#   ./serve-kimik31.sh stop       stop a running server
#   ./serve-kimik31.sh status     show server state + health endpoint
#
# Configuration comes from kimik31.env next to this script (or $KIMIK31_ENV),
# see kimik31-env.example. Environment variables you export beforehand win.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="serve"
DETACH=0
for arg in "$@"; do
    case "$arg" in
        --detach|-d) DETACH=1 ;;
        *)           MODE="$arg" ;;
    esac
done

log()  { printf '\033[1;34m[kimik31]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[kimik31 WARN]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[kimik31 ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

# ── Load config ─────────────────────────────────────────────────────────────

ENV_FILE="${KIMIK31_ENV:-$SCRIPT_DIR/kimik31.env}"
if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    [[ "$MODE" == "plan" ]] || log "Loaded config from $ENV_FILE"
else
    warn "No config file at $ENV_FILE (copy kimik31-env.example to kimik31.env); using environment only."
fi

# ── Pins ────────────────────────────────────────────────────────────────────
#
# moonshotai/Kimi-K3.1-Pro, released 4 Nov 2026. d9e99b07 (5 Nov, "Update
# encoding_k3.py") is the tip as of 10 Nov 2026 and what this repo was built
# against. Like K3, the repo keeps moving after release; an unpinned MODEL_ID
# makes the next `download` silently take main into a 1.5 TB cache.
MODEL_ID="${MODEL_ID:-moonshotai/Kimi-K3.1-Pro}"
MODEL_REVISION="${MODEL_REVISION:-d9e99b0792a5e93b78ea9bea62a68167e36ddd9e}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-kimi-k3.1-pro}"

# lmsysorg/sglang:dev-kimi-k31-pro-mi35x is a MOVING day-0 tag (ROCm 10, built
# from the kimi-k3.1-day-0 ROCm port). The cookbook's MI355X AgentX numbers were
# measured on sha256:5901db14 — pin that digest so a re-pull cannot change the
# engine under you. ROCm 10 only: upstream never ran K3.1 on ROCm 7 images, and
# Bunya's MI355X nodes run ROCm 10.1 (amdgpu 7.1.9) since Nov 2026.
SGLANG_IMAGE="${SGLANG_IMAGE:-docker://lmsysorg/sglang@sha256:5901db1401ad7b8bc62ac67175f6f013b1b868e2a2dab063ee8848e7290fa9a7}"

PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-8}"

# ── Recipe selection ────────────────────────────────────────────────────────
AGENTS="${AGENTS:-4}"
PROFILE="${PROFILE:-auto}"
SPECULATIVE="${SPECULATIVE-dflash}"
DFLASH_MODEL="${DFLASH_MODEL:-RadixArk/Kimi-K3.1-Dflash2-Preview}"
# ea257b38 (6 Nov 2026) is the tip; weights unchanged since d0d8bc2e (4 Nov),
# later commits are card/licence only. The draft's own config says block_size 5;
# the recipe passes the measured block explicitly, so a draft rewrite that
# changes its config cannot silently change what is served.
DFLASH_REVISION="${DFLASH_REVISION:-ea257b38a8ee755b65287bb8ccd0938b519ce26b}"
DFLASH_BLOCK_SIZE="${DFLASH_BLOCK_SIZE:-}"
BLOCK_VERIFY="${BLOCK_VERIFY:-1}"
MONOKERNEL="${MONOKERNEL:-auto}"
HICACHE="${HICACHE:-auto}"
HICACHE_RATIO="${HICACHE_RATIO:-}"
HICACHE_MAMBA_HOST_RATIO="${HICACHE_MAMBA_HOST_RATIO:-}"
HICACHE_GTT_BUDGET="${HICACHE_GTT_BUDGET:-0.85}"
TTM_SHIM="${TTM_SHIM:-auto}"

# ── Engine knobs: EMPTY = the profile's value from the verified recipe ──────
CONTEXT_LEN="${CONTEXT_LEN:-}"
MEM_FRACTION="${MEM_FRACTION:-}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-}"
CUDA_GRAPH_MAX_BS_DECODE="${CUDA_GRAPH_MAX_BS_DECODE:-}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-}"
PREFILL_DECODE_INTERVAL="${PREFILL_DECODE_INTERVAL:-}"
SCHEDULE_POLICY="${SCHEDULE_POLICY:-}"
MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-}"
MAMBA_FULL_MEMORY_RATIO="${MAMBA_FULL_MEMORY_RATIO:-}"
MAMBA_RADIX_STRATEGY="${MAMBA_RADIX_STRATEGY:-}"
ROCM_RUNTIME_RESERVE_GB="${ROCM_RUNTIME_RESERVE_GB:-}"
DISABLE_RADIX_CACHE="${DISABLE_RADIX_CACHE:-0}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
TOOL_PARSER="${TOOL_PARSER:-kimi_k3}"
REASONING_PARSER="${REASONING_PARSER:-kimi_k3}"
ENABLE_METRICS="${ENABLE_METRICS:-1}"
STREAM_INTERVAL="${STREAM_INTERVAL:-1}"
HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-}"
ENABLE_AITER="${ENABLE_AITER:-1}"
FLYDSL_FORCE="${FLYDSL_FORCE:-1}"
ROCM_MODE="${ROCM_MODE:-auto}"
AITER_GPU_ARCHS="${AITER_GPU_ARCHS:-}"
ROCMINFO_SHIM="${ROCMINFO_SHIM:-auto}"
SET_CPU_AFFINITY="${SET_CPU_AFFINITY:-0}"
READY_TIMEOUT="${READY_TIMEOUT:-14400}"
LAUNCH_CMD="${LAUNCH_CMD:-sglang serve}"
EXTRA_ENGINE_ARGS="${EXTRA_ENGINE_ARGS:-}"
EXTRA_ENV="${EXTRA_ENV:-}"

MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-}"
SIF_PATH="${SIF_PATH:-}"
WEIGHT_LOAD_THREADS="${WEIGHT_LOAD_THREADS:-}"
LOAD_FORMAT="${LOAD_FORMAT:-}"
PRESHARDED_PATH="${PRESHARDED_PATH:-}"
PREFETCH_BLOCK_SIZE_MB="${PREFETCH_BLOCK_SIZE_MB:-}"
PREFETCH_CHECKPOINTS="${PREFETCH_CHECKPOINTS:-1}"
LOADER_CFG=""

# Measured from the K3.1-Pro safetensors index (96 shards).
WEIGHTS_GB=1561
NEED_GB=1700

# Probe containers need writable caches that are not the real ones, and /tmp is
# shared between users on these nodes, so carry the uid (K3 lesson, 12 Aug 2026:
# an EMPTY AITER_JIT_DIR makes aiter call os.makedirs("") and every sglang model
# module fails to import — which 'check' then misreports as "unsupported").
PROBE_TMP="/tmp/kimik31-$(id -u 2>/dev/null || echo user)"
PROBE_ENV=(--env "AITER_JIT_DIR=$PROBE_TMP/aiter-jit"
           --env "SGLANG_CACHE_DIR=$PROBE_TMP/sglang-cache")

# ── Validation ──────────────────────────────────────────────────────────────

is_uint()  { [[ "$1" =~ ^[0-9]+$ ]]; }
is_num()   { [[ "$1" =~ ^[0-9]+(\.[0-9]+)?$ ]]; }
opt_uint() { [[ -z "${!1}" ]] || is_uint "${!1}" || die "$1 must be a non-negative integer or empty, got '${!1}'."; }
opt_num()  { [[ -z "${!1}" ]] || is_num  "${!1}" || die "$1 must be a number or empty, got '${!1}'."; }

is_uint "$AGENTS" && (( AGENTS >= 1 )) || die "AGENTS must be a positive integer, got '$AGENTS'."
case "$PROFILE" in
    auto|lowlatency|mono-dcp8|throughput|throughput-wide) ;;
    *) die "PROFILE must be auto, lowlatency, mono-dcp8, throughput or throughput-wide, got '$PROFILE'." ;;
esac
case "$SPECULATIVE" in
    dflash) ;;
    ""|none) SPECULATIVE="none" ;;
    *) die "SPECULATIVE must be 'dflash' or 'none'/empty, got '$SPECULATIVE'.
  K3.1 has no NextN layer, so DFLASH (a separate draft) is the only speculative
  algorithm upstream ships for it. The K3 DSpark draft does not fit K3.1." ;;
esac
case "$MONOKERNEL" in auto|0|1) ;; *) die "MONOKERNEL must be auto, 0 or 1, got '$MONOKERNEL'." ;; esac
case "$HICACHE"    in auto|on|off) ;; *) die "HICACHE must be auto, on or off, got '$HICACHE'." ;; esac
case "$TTM_SHIM"   in auto|off) ;; *) die "TTM_SHIM must be auto or off, got '$TTM_SHIM'." ;; esac
case "$BLOCK_VERIFY" in 0|1) ;; *) die "BLOCK_VERIFY must be 0 or 1, got '$BLOCK_VERIFY'." ;; esac
case "$ROCM_MODE"  in auto|rocm|devices) ;; *) die "ROCM_MODE must be auto, rocm or devices, got '$ROCM_MODE'." ;; esac
case "$ROCMINFO_SHIM" in auto|off|force) ;; *) die "ROCMINFO_SHIM must be auto, off or force, got '$ROCMINFO_SHIM'." ;; esac
case "$MAMBA_RADIX_STRATEGY" in ""|auto|extra_buffer|extra_buffer_lazy|no_buffer) ;;
    *) die "MAMBA_RADIX_STRATEGY must be empty, auto, extra_buffer, extra_buffer_lazy or no_buffer, got '$MAMBA_RADIX_STRATEGY'." ;;
esac
# K3.1's server-arg hook forces FP8 KV (the reference fp8_ds_mla format) and
# raises on anything else, but only after the container is up. Fail here.
case "$KV_CACHE_DTYPE" in ""|auto|fp8_e4m3) ;;
    *) die "KV_CACHE_DTYPE must be empty or fp8_e4m3 — Kimi-K3.1 rejects '$KV_CACHE_DTYPE' (it requires the fp8_ds_mla KV format)." ;;
esac
for v in MAX_RUNNING_REQUESTS CUDA_GRAPH_MAX_BS_DECODE CHUNKED_PREFILL_SIZE PREFILL_DECODE_INTERVAL \
         MAX_MAMBA_CACHE_SIZE ROCM_RUNTIME_RESERVE_GB DFLASH_BLOCK_SIZE CONTEXT_LEN \
         WEIGHT_LOAD_THREADS PREFETCH_BLOCK_SIZE_MB STREAM_INTERVAL; do opt_uint "$v"; done
for v in MEM_FRACTION MAMBA_FULL_MEMORY_RATIO HICACHE_RATIO HICACHE_MAMBA_HOST_RATIO HICACHE_GTT_BUDGET; do opt_num "$v"; done
[[ -n "$DFLASH_BLOCK_SIZE" && "$SPECULATIVE" != "dflash" ]] \
    && warn "DFLASH_BLOCK_SIZE=$DFLASH_BLOCK_SIZE is ignored while SPECULATIVE=none."
[[ "$AITER_GPU_ARCHS" == *";"* || "$AITER_GPU_ARCHS" == *","* ]] \
    && die "AITER_GPU_ARCHS must name exactly one architecture (ROCm/aiter#3807), got '$AITER_GPU_ARCHS'."
[[ -n "$PRESHARDED_PATH" && "$LOAD_FORMAT" != "presharded" ]] \
    && warn "PRESHARDED_PATH is set but LOAD_FORMAT is '${LOAD_FORMAT:-auto}' — the path is ignored."
if [[ -n "$WEIGHT_LOAD_THREADS" ]] && (( WEIGHT_LOAD_THREADS > 16 )); then
    # K3.1's 96 shards average ~16 GB (K3's were ~5 GB), so the buffered loader's
    # (threads + 2) resident shards cost three times what they did on K3.
    warn "WEIGHT_LOAD_THREADS=$WEIGHT_LOAD_THREADS holds roughly $(( (WEIGHT_LOAD_THREADS + 2) * 16 )) GB of shards in host RAM
  (K3.1 shards are ~16 GB). Check that against your --mem."
fi

# ── Resolve the recipe ──────────────────────────────────────────────────────

if [[ "$PROFILE" == "auto" ]]; then
    if   (( AGENTS <= 4 ));  then PROFILE=lowlatency
    elif (( AGENTS <= 8 ));  then PROFILE=mono-dcp8
    elif (( AGENTS <= 32 )); then PROFILE=throughput
    elif (( AGENTS <= 56 )); then PROFILE=throughput-wide
    else die "AGENTS=$AGENTS: upstream's single-node MI355X recipe is verified to 56 agents.
  Past that the cookbook runs a 1P1D (prefill + decode on two nodes, Mooncake over
  RDMA), which this single-node launcher does not do. Set AGENTS<=56."
    fi
fi

# Every value below is transcribed from the cookbook's MI355X cells (Deploy panel
# + AgentX launches of 2026-11-02..04). Empty user knobs take these.
R_ENV=()
case "$PROFILE" in
    lowlatency)
        p_dcp=1; p_mono=1; p_hicache=off; p_reserve=6; p_block=6
        p_chunk=16384; p_mfmr=""; p_radix=extra_buffer_lazy
        if (( AGENTS <= 1 )); then
            p_maxrun=1; p_mamba_cache=60; p_pdi=""      # AgentX c=1 point
        else
            p_maxrun=4; p_mamba_cache=150; p_pdi=16     # Deploy cell / AgentX c=4
        fi ;;
    mono-dcp8)
        p_dcp=8; p_mono=1; p_hicache=off; p_reserve=10; p_block=6
        p_chunk=16384; p_mfmr=0.65; p_radix=extra_buffer_lazy
        p_maxrun=8; p_mamba_cache=""; p_pdi=16 ;;       # AgentX c=8 point
    throughput)
        p_dcp=8; p_mono=0; p_hicache=on; p_reserve=10; p_block=6
        p_chunk=16384; p_mfmr=0.65; p_radix=extra_buffer_lazy
        p_maxrun=$(( AGENTS < 16 ? 16 : AGENTS )); p_mamba_cache=""; p_pdi=16 ;;
    throughput-wide)
        p_dcp=8; p_mono=0; p_hicache=on; p_reserve=10; p_block=5
        p_chunk=32768; p_mfmr=0.65; p_radix=extra_buffer_lazy
        p_maxrun=$(( AGENTS > 48 ? 64 : 48 )); p_mamba_cache=""; p_pdi=8
        # From 48 agents: MoE front + residual stream on row shards in prefill (no
        # extra memory) and a draft block's ring locations in one launch.
        R_ENV+=(SGLANG_ENABLE_ROCM_K3_ROW_SHARDED_ROUTED_FRONT=1
                SGLANG_ENABLE_ROCM_K3_ROW_SHARDED_STREAM=1
                SGLANG_OPT_DFLASH_RING_LOCS_FUSED=1) ;;
esac

[[ "$MONOKERNEL" == "auto" ]] || p_mono="$MONOKERNEL"
[[ "$HICACHE" == "auto" ]]    || p_hicache="$HICACHE"
: "${DFLASH_BLOCK_SIZE:=$p_block}"
# The monokernel only runs decode/verify for a speculative target: SGLang's
# handle_monokernel() requires speculative_algorithm DFLASH|DSPARK, ReplaySSM,
# and 6 or 8 draft tokens per request — refused at startup otherwise (measured
# on the pinned image, 10 Nov 2026).
if [[ "$p_mono" == "1" ]] && [[ "$SPECULATIVE" != "dflash" || ( "$DFLASH_BLOCK_SIZE" != 6 && "$DFLASH_BLOCK_SIZE" != 8 ) ]]; then
    if [[ "$MONOKERNEL" == "1" ]]; then
        die "MONOKERNEL=1 needs SPECULATIVE=dflash with DFLASH_BLOCK_SIZE 6 or 8 (SGLang's monokernel
  runs speculative verify steps only). Got SPECULATIVE=$SPECULATIVE, block $DFLASH_BLOCK_SIZE."
    fi
    [[ "$MODE" == "plan" ]] || warn "Monokernel off: it requires DFLASH with block 6 or 8 (SPECULATIVE=$SPECULATIVE, block $DFLASH_BLOCK_SIZE)."
    p_mono=0
fi
if [[ "$p_mono" == "1" && "$p_hicache" == "on" ]]; then
    die "MONOKERNEL=1 with HICACHE=on: the AMD decode monokernel does not accept KV
  offload, and SGLang refuses the combination at startup. Pick one."
fi
if (( p_dcp > 1 )) && (( TP_SIZE % p_dcp )); then
    die "PROFILE=$PROFILE uses DCP$p_dcp, which must divide TP_SIZE=$TP_SIZE."
fi

: "${MEM_FRACTION:=0.85}"
: "${MAX_RUNNING_REQUESTS:=$p_maxrun}"
: "${CUDA_GRAPH_MAX_BS_DECODE:=$MAX_RUNNING_REQUESTS}"
: "${CHUNKED_PREFILL_SIZE:=$p_chunk}"
: "${PREFILL_DECODE_INTERVAL:=$p_pdi}"
: "${SCHEDULE_POLICY:=lpm}"
: "${MAX_MAMBA_CACHE_SIZE:=$p_mamba_cache}"
: "${MAMBA_FULL_MEMORY_RATIO:=$p_mfmr}"
: "${MAMBA_RADIX_STRATEGY:=$p_radix}"
: "${ROCM_RUNTIME_RESERVE_GB:=$p_reserve}"
DCP_SIZE="$p_dcp"
USE_MONO="$p_mono"
USE_HICACHE="$p_hicache"

if (( MAX_RUNNING_REQUESTS < AGENTS )); then
    warn "MAX_RUNNING_REQUESTS=$MAX_RUNNING_REQUESTS is below AGENTS=$AGENTS — extra agents will queue."
fi
# Upstream: "--mem-fraction-static 0.87 or 0.88 overflowed GTT at startup or ran
# slower on the single-node recipe."
if awk -v m="$MEM_FRACTION" 'BEGIN{exit !(m > 0.86)}'; then
    warn "MEM_FRACTION=$MEM_FRACTION: upstream saw 0.87/0.88 overflow GTT or run slower on MI355X."
fi

# ── HiCache: size the host pools from THIS node's GTT ───────────────────────
#
# The pinned host pools are hipHostMalloc'd, i.e. GTT memory, and all 8 ranks
# share the node's GTT limit (half of RAM). Upstream's ratios (2.5 / mamba 10)
# pinned 170-185 GB per GPU against a 1511 GiB GTT, and ratio 12 overflowed it.
# Bunya's MI355X nodes have 2.2 TB RAM -> 1133 GiB GTT, so the same ratios would
# fail at startup. Scale both by (GTT / 1511) * HICACHE_GTT_BUDGET. The pooled
# size is linear in both ratios, so this keeps the same headroom fraction.
node_gtt_gib() {
    local f
    for f in /sys/class/drm/card*/device/mem_info_gtt_total; do
        [[ -r "$f" ]] || continue
        local b; b="$(<"$f")"
        is_uint "$b" && (( b > 0 )) && { echo $(( b / 1073741824 )); return 0; }
    done
    echo ""
}
GTT_GIB="$(node_gtt_gib)"
GTT_SRC="this node"
if [[ -z "$GTT_GIB" ]]; then GTT_GIB=1133; GTT_SRC="assumed (Bunya MI355X; no GPU visible here)"; fi

HICACHE_FLAGS=()
HICACHE_NOTE=""
if [[ "$USE_HICACHE" == "on" ]]; then
    scale="$(awk -v g="$GTT_GIB" -v b="$HICACHE_GTT_BUDGET" 'BEGIN{s=g/1511*b; if(s>1)s=1; printf "%.4f", s}')"
    [[ -n "$HICACHE_RATIO" ]] \
        || HICACHE_RATIO="$(awk -v s="$scale" 'BEGIN{printf "%.1f", int(2.5*s*10)/10}')"
    [[ -n "$HICACHE_MAMBA_HOST_RATIO" ]] \
        || HICACHE_MAMBA_HOST_RATIO="$(awk -v s="$scale" 'BEGIN{v=int(10*s); if(v<1)v=1; print v}')"
    if awk -v r="$HICACHE_RATIO" 'BEGIN{exit !(r <= 1.0)}'; then
        warn "HiCache ratio would be $HICACHE_RATIO (<= 1: host tier no larger than HBM) — disabling HiCache."
        USE_HICACHE=off
    else
        HICACHE_FLAGS=(--enable-hierarchical-cache --hicache-write-policy write_back
                       --hicache-ratio "$HICACHE_RATIO")
        R_ENV+=("SGLANG_HICACHE_MAMBA_HOST_RATIO=$HICACHE_MAMBA_HOST_RATIO"
                # keeps every rank on the Rust TreeCore; write-back asserts on a
                # mix with the Python one
                RUSTUP_TOOLCHAIN=stable)
        # ~177 GB/GPU at 2.5/10 upstream, linear in the ratios.
        est_gib="$(awk -v r="$HICACHE_RATIO" -v m="$HICACHE_MAMBA_HOST_RATIO" \
            'BEGIN{printf "%d", 8*177*0.931*((r/2.5)+(m/10))/2}')"
        HICACHE_NOTE="ratio $HICACHE_RATIO, mamba-host $HICACHE_MAMBA_HOST_RATIO, ~${est_gib} GiB pinned of ${GTT_GIB} GiB GTT ($GTT_SRC)"
        if (( est_gib > GTT_GIB * 95 / 100 )); then
            warn "HiCache would pin ~${est_gib} GiB against a ${GTT_GIB} GiB GTT — hipHostMalloc will likely fail at startup."
        fi
        if [[ -n "${SLURM_MEM_PER_NODE:-}" ]] && is_uint "$SLURM_MEM_PER_NODE" \
           && (( est_gib * 1024 > SLURM_MEM_PER_NODE - 200 * 1024 )); then
            warn "HiCache pins ~${est_gib} GiB of host RAM; the job has $(( SLURM_MEM_PER_NODE / 1024 )) GiB (--mem). Leave >=200 GiB headroom."
        fi
    fi
fi

# ── Profile env (the cookbook's MI355X cells) ───────────────────────────────

if [[ "$ENABLE_AITER" == "1" ]]; then
    # AITER ROCm path, its K3 fused kernels, FlyDSL MoE kernels, SiTU v2 MoE with
    # A8W4 activation quantisation. Same four as K3.
    R_ENV+=(SGLANG_USE_AITER=1 SGLANG_AITER_K3_OPT=1 AITER_SITUV2_A8W4=1)
    if [[ "$FLYDSL_FORCE" == "1" ]]; then R_ENV+=(AITER_FLYDSL_FORCE=1)
    else warn "FLYDSL_FORCE=0 — aiter's prebuilt gemm path, below upstream's numbers."; fi
fi
R_ENV+=(PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.95
        # keeps GPU memory free for HIP queue scratch, allocated outside torch
        "SGLANG_ROCM_RUNTIME_RESERVE_GB=$ROCM_RUNTIME_RESERVE_GB"
        SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1
        # back pinned host pools with hipHostMalloc (GTT, never migrates);
        # page migration in registered host memory stalls every GPU queue
        HSA_USERPTR_FOR_PAGED_MEM=0)
# The cookbook's switch for the K3.1 wide monokernel. sgl-amd-mono 9cc3c30 (8 Nov
# 2026) made wide the only path under --enable-amd-monokernel, so the pinned
# digest never reads it (flagcheck reports it inert); earlier digests of the
# moving dev-kimi-k31-pro-mi35x tag do. Harmless either way.
[[ "$USE_MONO" == "1" ]] && R_ENV+=(SGLANG_AMD_MONO_K31_WIDE=1)
[[ -n "$HSA_NO_SCRATCH_RECLAIM" ]] && R_ENV+=("HSA_NO_SCRATCH_RECLAIM=$HSA_NO_SCRATCH_RECLAIM")
[[ -n "$AITER_GPU_ARCHS" ]] && R_ENV+=("GPU_ARCHS=$AITER_GPU_ARCHS")
[[ -n "$PREFETCH_BLOCK_SIZE_MB" ]] && R_ENV+=("SGLANG_PREFETCH_BLOCK_SIZE_MB=$PREFETCH_BLOCK_SIZE_MB")
# shellcheck disable=SC2206
extra_env=($EXTRA_ENV)
for kv in ${extra_env[@]+"${extra_env[@]}"}; do
    [[ "$kv" == [A-Za-z_]*=* ]] || die "EXTRA_ENV entries must be NAME=VALUE, got '$kv'."
    R_ENV+=("$kv")
done

# ── Build the engine argv ───────────────────────────────────────────────────
# Uses globals resolved later in serve mode (MODEL_PATH, DRAFT_PATH, API key,
# loader config); 'plan' calls it with placeholders.

build_cmd() {
    # shellcheck disable=SC2206
    local launcher=($LAUNCH_CMD)
    # shellcheck disable=SC2206
    local extra=($EXTRA_ENGINE_ARGS)
    cmd=("${launcher[@]}"
         --model-path "$MODEL_PATH"
         --served-model-name "$SERVED_MODEL_NAME"
         --trust-remote-code
         --tp-size "$TP_SIZE")
    (( DCP_SIZE > 1 )) && cmd+=(--dcp-size "$DCP_SIZE" --dcp-comm-backend a2a)
    cmd+=(--mem-fraction-static "$MEM_FRACTION"
          --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
          --max-running-requests "$MAX_RUNNING_REQUESTS"
          --cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS_DECODE"
          --schedule-policy "$SCHEDULE_POLICY")
    [[ -n "$PREFILL_DECODE_INTERVAL" ]] && cmd+=(--prefill-decode-interval "$PREFILL_DECODE_INTERVAL")
    [[ -n "$MAX_MAMBA_CACHE_SIZE" ]]    && cmd+=(--max-mamba-cache-size "$MAX_MAMBA_CACHE_SIZE")
    [[ -n "$MAMBA_FULL_MEMORY_RATIO" ]] && cmd+=(--mamba-full-memory-ratio "$MAMBA_FULL_MEMORY_RATIO")
    [[ -n "$MAMBA_RADIX_STRATEGY" ]]    && cmd+=(--mamba-radix-cache-strategy "$MAMBA_RADIX_STRATEGY")
    # Prefill CUDA graphs (breakable) over the chunk sizes agent turns produce.
    cmd+=(--cuda-graph-backend-prefill breakable
          --cuda-graph-bs-prefill 256 512 768 1024 1536 2048 3072 4096)
    [[ "$USE_MONO" == "1" ]] && cmd+=(--enable-amd-monokernel)
    cmd+=(${HICACHE_FLAGS[@]+"${HICACHE_FLAGS[@]}"})
    if [[ "$SPECULATIVE" == "dflash" ]]; then
        cmd+=(--speculative-algorithm DFLASH
              --speculative-draft-model-path "$DRAFT_PATH"
              --speculative-dflash-block-size "$DFLASH_BLOCK_SIZE"
              --speculative-draft-kv-cache-dtype fp8_e4m3
              --speculative-draft-attention-backend triton
              # Without ReplaySSM, DFLASH keeps one KDA state per draft token
              # and the KV pool collapses (5.13M -> 0.77M tokens upstream).
              --enable-linear-replayssm-spec)
        [[ "$BLOCK_VERIFY" == "1" ]] && cmd+=(--speculative-use-block-verification)
    fi
    [[ -n "$KV_CACHE_DTYPE" && "$KV_CACHE_DTYPE" != "auto" ]] && cmd+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
    [[ -n "$CONTEXT_LEN" ]] && cmd+=(--context-length "$CONTEXT_LEN")
    [[ "$DISABLE_RADIX_CACHE" == "1" ]] && cmd+=(--disable-radix-cache)
    [[ "$TOOL_PARSER" != "none" && -n "$TOOL_PARSER" ]] && cmd+=(--tool-call-parser "$TOOL_PARSER")
    [[ "$REASONING_PARSER" != "none" && -n "$REASONING_PARSER" ]] && cmd+=(--reasoning-parser "$REASONING_PARSER")
    [[ "$ENABLE_METRICS" == "1" ]] && cmd+=(--enable-metrics)
    [[ -n "$STREAM_INTERVAL" ]] && cmd+=(--stream-interval "$STREAM_INTERVAL")
    [[ "$PREFETCH_CHECKPOINTS" == "1" ]] && cmd+=(--weight-loader-prefetch-checkpoints)
    [[ -n "$LOAD_FORMAT" ]] && cmd+=(--load-format "$LOAD_FORMAT")
    [[ -n "$LOADER_CFG" ]] && cmd+=(--model-loader-extra-config "$LOADER_CFG")
    cmd+=(--host 0.0.0.0 --port "$PORT" --api-key "$API_KEY_ARG")
    cmd+=(${extra[@]+"${extra[@]}"})
}

# Loader config. --weight-loader-prefetch-checkpoints alone makes SGLang fall
# back to SINGLE-THREADED loading (the K3 "sawtooth"); naming
# enable_multithread_load keeps it multi-threaded. Upstream's recipe passes
# exactly {"enable_multithread_load": true}; num_threads only if you set it.
make_loader_cfg() {
    local c='"enable_multithread_load": true'
    [[ -n "$WEIGHT_LOAD_THREADS" && "$WEIGHT_LOAD_THREADS" != "0" ]] && c+=", \"num_threads\": $WEIGHT_LOAD_THREADS"
    if [[ "$LOAD_FORMAT" == "presharded" && -n "$PRESHARDED_PATH" ]]; then
        c+=", \"presharded_path\": \"$PRESHARDED_PATH\""
        [[ "$SPECULATIVE" == "dflash" ]] && c+=", \"draft_presharded_path\": \"$PRESHARDED_PATH/dflash\""
    fi
    LOADER_CFG="{$c}"
}

recipe_summary() {
    local s="profile=$PROFILE agents=$AGENTS tp=$TP_SIZE"
    (( DCP_SIZE > 1 )) && s+=" dcp=$DCP_SIZE"
    s+=" mono=$USE_MONO hicache=$USE_HICACHE spec=$SPECULATIVE"
    if [[ "$SPECULATIVE" == "dflash" ]]; then
        s+="(block $DFLASH_BLOCK_SIZE"; [[ "$BLOCK_VERIFY" == "1" ]] && s+=", block-verify"; s+=")"
    fi
    s+=" max-running=$MAX_RUNNING_REQUESTS ctx=${CONTEXT_LEN:-model max (1M)}"
    printf '%s' "$s"
}

# ── Runtime state paths ─────────────────────────────────────────────────────

if [[ -n "$MODEL_CACHE_DIR" ]]; then
    PID_FILE="${PID_FILE:-$MODEL_CACHE_DIR/kimik31-server.pid}"
    LOG_FILE="${LOG_FILE:-$MODEL_CACHE_DIR/kimik31-server.log}"
    LOADTIMES_FILE="${LOADTIMES_FILE:-$MODEL_CACHE_DIR/kimik31-loadtimes.log}"
else
    PID_FILE="${PID_FILE:-$SCRIPT_DIR/kimik31-server.pid}"
    LOG_FILE="${LOG_FILE:-$SCRIPT_DIR/kimik31-server.log}"
    LOADTIMES_FILE="${LOADTIMES_FILE:-$SCRIPT_DIR/kimik31-loadtimes.log}"
fi
API_KEY_FILE="${MODEL_CACHE_DIR:-$SCRIPT_DIR}/kimik31-api-key"

server_running() { [[ -f "$PID_FILE" ]] && kill -0 "$(<"$PID_FILE")" 2>/dev/null; }

stop_server() {
    if server_running; then
        local pid; pid="$(<"$PID_FILE")"
        log "Stopping server (pid $pid) ..."
        kill "$pid" 2>/dev/null || true
        for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    else
        warn "No running server recorded in $PID_FILE."
    fi
    # Single-user 8-GPU node: clean up strays of either invocation form.
    pkill -u "$(id -u)" -f 'sglang.launch_server' 2>/dev/null || true
    pkill -u "$(id -u)" -f 'sglang serve'         2>/dev/null || true
    rm -f "$PID_FILE"
}

resolve_key() {
    local key=""
    [[ -r "$API_KEY_FILE" ]] && key="$(<"$API_KEY_FILE")"
    printf '%s' "${KIMIK31_API_KEY:-$key}"
}

# ── Simple modes (no apptainer) ─────────────────────────────────────────────

case "$MODE" in
    plan)
        MODEL_PATH="<cache>/models--${MODEL_ID//\//--}/snapshots/${MODEL_REVISION:-main}"
        DRAFT_PATH="<cache>/models--${DFLASH_MODEL//\//--}/snapshots/${DFLASH_REVISION:-main}"
        API_KEY_ARG="<redacted>"
        make_loader_cfg
        build_cmd
        echo "# $(recipe_summary)"
        [[ -n "$HICACHE_NOTE" ]] && echo "# hicache: $HICACHE_NOTE"
        echo "# env:"
        for e in "${R_ENV[@]}" "SGLANG_SET_CPU_AFFINITY=$SET_CPU_AFFINITY"; do echo "  $e"; done
        echo "# argv:"
        printf '  %q' "${cmd[@]}"; echo
        exit 0
        ;;
    stop)
        stop_server; log "Stopped."; exit 0 ;;
    status)
        if server_running; then log "Server process alive (pid $(<"$PID_FILE"))."
        else log "No server process recorded (pidfile $PID_FILE)."; fi
        if curl -fsS -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            log "Health check on port $PORT: OK"
            curl -fsS -m 5 "http://127.0.0.1:${PORT}/v1/models" \
                 -H "Authorization: Bearer $(resolve_key)" 2>/dev/null | head -c 400 || true
            echo
        else
            warn "http://127.0.0.1:${PORT}/health not responding (not started, or still loading)."
        fi
        if [[ -r "$LOG_FILE" ]]; then
            head -n 3 "$LOG_FILE" | grep '^### recipe' || true
            if grep -q 'k31 wide\] first wide step' "$LOG_FILE"; then
                log "AMD monokernel engaged ('[k31 wide] first wide step' in the log)."
            elif grep -q -- '--enable-amd-monokernel' "$LOG_FILE"; then
                log "Monokernel requested; not yet engaged (it logs on the first decode step)."
            fi
            acc="$(grep -oE ', accept len: [0-9.]+' "$LOG_FILE" | tail -n 20 \
                   | awk '{s+=$NF; n++} END {if (n) printf "%.2f over the last %d log lines", s/n, n}')"
            [[ -n "$acc" ]] && log "DFLASH accept length: $acc"
            log "Log: $LOG_FILE"
        fi
        exit 0
        ;;
    toolcheck)
        # Pure HTTP; also works through a tunnel. Turn 2 is the half that matters:
        # the tool returns a value the model cannot guess, and it must come back.
        # K3.1 REQUIRES the full assistant message — reasoning_content included —
        # to be passed back on the next turn; that is exactly what agent clients
        # get wrong, so this replays it the way the model card says.
        BASE="${TOOLCHECK_URL:-http://127.0.0.1:$PORT}" \
        KEY="$(resolve_key)" \
        MODEL="$SERVED_MODEL_NAME" \
        EFFORT="${TOOLCHECK_EFFORT:-low}" \
        python3 - <<'PY'
import json, os, sys, urllib.request, urllib.error

BASE, KEY, MODEL, EFFORT = (os.environ[k] for k in ("BASE", "KEY", "MODEL", "EFFORT"))
SECRET = 61.4
NODE   = "bun159"
TOOLS = [{"type": "function", "function": {
    "name": "get_gpu_temperature",
    "description": "Return the current GPU temperature in Celsius for a named Bunya compute node.",
    "parameters": {"type": "object",
                   "properties": {"node": {"type": "string", "description": "Node hostname, e.g. bun159"}},
                   "required": ["node"]}}}]

def post(payload):
    if EFFORT:
        payload["reasoning_effort"] = EFFORT
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code}: {e.read().decode()[:400]}"); sys.exit(1)
    except Exception as e:
        print(f"  request failed: {e}\n  Is the server up? ./serve-kimik31.sh status"); sys.exit(1)

fails = []
def check(ok, label, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  — {detail}" if detail else ""))
    if not ok: fails.append(label)

msgs = [{"role": "user", "content": f"What is the current GPU temperature on {NODE}? Use the tool, then state the number."}]
print(f"Turn 1 — does the model emit a tool call? (reasoning_effort={EFFORT or 'server default'})")
r1 = post({"model": MODEL, "messages": msgs, "tools": TOOLS, "tool_choice": "auto",
           "max_tokens": 8192, "temperature": 1.0, "top_p": 0.95})
m1 = r1["choices"][0]["message"]; fin = r1["choices"][0]["finish_reason"]
calls = m1.get("tool_calls") or []
reasoning = m1.get("reasoning_content") or m1.get("reasoning") or ""
check(bool(reasoning), "reasoning parser split out reasoning_content",
      f"{len(reasoning)} chars" if reasoning else "K3.1 always thinks; empty means REASONING_PARSER is wrong")
check(bool(calls), "model returned tool_calls",
      f"finish_reason={fin}" + ("" if calls else f", content={(m1.get('content') or '')[:120]!r}"))
if not calls:
    if fin == "length": print("\n  finish_reason=length: it never finished thinking. Raise max_tokens.")
    print("  A raw tool call in content means the tool parser does not match. See ./serve-kimik31.sh parsers.")
    sys.exit(1)
fn = calls[0].get("function", {})
check(fn.get("name") == "get_gpu_temperature", "correct function name", repr(fn.get("name")))
raw = fn.get("arguments")
try:
    args = json.loads(raw) if isinstance(raw, str) else raw; ok_json = isinstance(args, dict)
except Exception as e:
    args, ok_json = None, False; print(f"        arguments did not parse: {e}")
check(ok_json, "arguments are valid JSON", repr(raw)[:160])
check(bool(args) and NODE in str(args.get("node", "")), "argument value carried through",
      repr(args.get("node") if args else None))
check(bool(calls[0].get("id")), "tool_call has an id", repr(calls[0].get("id")))

print("\nTurn 2 — does the model use the tool result (full assistant message replayed)?")
asst = {"role": "assistant", "content": m1.get("content") or "", "tool_calls": calls}
if reasoning: asst["reasoning_content"] = reasoning
msgs += [asst, {"role": "tool", "tool_call_id": calls[0].get("id"), "name": "get_gpu_temperature",
                "content": json.dumps({"node": NODE, "celsius": SECRET})}]
r2 = post({"model": MODEL, "messages": msgs, "tools": TOOLS, "max_tokens": 8192,
           "temperature": 1.0, "top_p": 0.95})
m2 = r2["choices"][0]["message"]; final = (m2.get("content") or "").strip()
check(bool(final), "final answer has content", f"finish_reason={r2['choices'][0]['finish_reason']}")
check(str(SECRET) in final, f"final answer contains the tool's value ({SECRET})")
print(f"\n  final answer: {final[:300]}")
usage = r2.get("usage") or {}
if usage: print(f"  usage turn 2: {usage}")
print()
if fails:
    print(f"TOOL-CALL ROUND TRIP FAILED: {len(fails)} check(s) — {', '.join(fails)}"); sys.exit(1)
print("TOOL-CALL ROUND TRIP OK — opencode / kimicode agentic loops should work.")
PY
        exit $?
        ;;
    serve|pull|download|check|gpucheck|flagcheck|parsers|loadstat) ;;
    *) die "Unknown mode '$MODE'. Use: serve | plan | pull | download | check | gpucheck | flagcheck | toolcheck | parsers | loadstat | stop | status" ;;
esac

# ── Preflight (shared) ──────────────────────────────────────────────────────

command -v apptainer >/dev/null 2>&1 \
    || die "apptainer not found — run this INSIDE a compute-node allocation (salloc/sbatch). Apptainer is not installed on Bunya login nodes."
command -v curl >/dev/null 2>&1 || die "curl not found on PATH."
[[ -n "$MODEL_CACHE_DIR" ]] \
    || die "MODEL_CACHE_DIR is not set. Point it at scratch (e.g. /scratch/user/\$USER/models). See kimik31-env.example."
mkdir -p "$MODEL_CACHE_DIR" 2>/dev/null || true
[[ -d "$MODEL_CACHE_DIR" && -w "$MODEL_CACHE_DIR" ]] \
    || die "MODEL_CACHE_DIR '$MODEL_CACHE_DIR' does not exist or is not writable."

SIF_PATH="${SIF_PATH:-$MODEL_CACHE_DIR/kimik31-mi355x.sif}"
if [[ "$SIF_PATH" == */ || -d "$SIF_PATH" ]]; then
    SIF_PATH="${SIF_PATH%/}/kimik31-mi355x.sif"
    log "SIF_PATH was a directory — using $SIF_PATH"
fi

: "${APPTAINER_CACHEDIR:=$MODEL_CACHE_DIR/../apptainer-cache}"
: "${APPTAINER_TMPDIR:=$MODEL_CACHE_DIR/../apptainer-tmp}"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" 2>/dev/null || true
export APPTAINER_CACHEDIR APPTAINER_TMPDIR
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR" SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"

# sglang resolves TRITON/TORCHINDUCTOR/JIT caches from SGLANG_CACHE_DIR, default
# ~/.cache/sglang — on /home, which has a quota. Keep it on scratch.
SGLANG_CACHE_DIR="${SGLANG_CACHE_DIR:-$MODEL_CACHE_DIR/sglang-cache-k31}"
mkdir -p "$SGLANG_CACHE_DIR" 2>/dev/null || true
[[ -w "$SGLANG_CACHE_DIR" ]] || warn "SGLANG_CACHE_DIR='$SGLANG_CACHE_DIR' is not writable — sglang will fall back to ~/.cache on /home."

# ── /tmp/aiter_configs: aiter's HARDCODED merged-config path ────────────────
# aiter merges per-model tuned CSVs into /tmp/aiter_configs/<op>.csv. The host
# /tmp is bound in and shared between users, so the first user to run aiter owns
# it and everyone else dies at import (PermissionError ... .csv.lock). Skipping
# the merge is NOT survivable either (K3, 19 Oct 2026: aiter then picks a CK MoE
# kernel that was never built). So bind a per-user, per-job directory over it,
# creating the mount point (mode 1777) when the node has none.
aiter_config_base="$MODEL_CACHE_DIR/aiter-configs"
AITER_CONFIG_DIR="${AITER_CONFIG_DIR:-$aiter_config_base/k31-${SLURM_JOB_ID:-$$}}"
AITER_CONFIG_ARGS=()

aiter_config_setup() {
    local owner
    [[ -n "$AITER_CONFIG_DIR" ]] || return 0
    if [[ -e /tmp/aiter_configs && ! -w /tmp/aiter_configs ]]; then
        owner="$(stat -c '%U' /tmp/aiter_configs 2>/dev/null || echo '?')"
        log "/tmp/aiter_configs on this node belongs to '$owner', not you — binding over it."
    fi
    if [[ "$AITER_CONFIG_DIR" == "$aiter_config_base/"* && -d "$aiter_config_base" ]]; then
        find "$aiter_config_base" -mindepth 1 -maxdepth 1 -type d -mtime +7 -exec rm -rf {} + 2>/dev/null || true
    fi
    [[ -e /tmp/aiter_configs ]] || { mkdir -m 1777 /tmp/aiter_configs 2>/dev/null \
        && log "Created /tmp/aiter_configs on this node (mode 1777) as a mount point."; } || true
    if mkdir -p "$AITER_CONFIG_DIR" 2>/dev/null && [[ -w "$AITER_CONFIG_DIR" && -f "$SIF_PATH" ]] \
       && apptainer exec --bind "$AITER_CONFIG_DIR":/tmp/aiter_configs "$SIF_PATH" \
            sh -c 'touch /tmp/aiter_configs/.write-test && rm -f /tmp/aiter_configs/.write-test' >/dev/null 2>&1; then
        AITER_CONFIG_ARGS=(--bind "$AITER_CONFIG_DIR":/tmp/aiter_configs)
    else
        warn "Could not bind $AITER_CONFIG_DIR over /tmp/aiter_configs. If startup dies with
  'Permission denied: /tmp/aiter_configs/*.lock', another user owns it on this node:
  run 'mkdir -m 1777 /tmp/aiter_configs' as root/RCC, or move nodes."
    fi
}

# ── Building the .sif ───────────────────────────────────────────────────────
# Apptainer 1.5.0-1.5.2 wrapped mksquashfs in proot, which dies on hosts that
# refuse ptrace; `build --ignore-proot` is the escape hatch (it is not on pull).
pull_image() {
    local logf rc
    logf="$(mktemp "${APPTAINER_TMPDIR:-/tmp}/kimik31-pull.XXXXXX.log")"
    log "Pulling $SGLANG_IMAGE -> $SIF_PATH (~14 GB compressed; one-time) ..."
    if apptainer pull "$SIF_PATH" "$SGLANG_IMAGE" 2>&1 | tee "$logf"; then rc=0; else rc=${PIPESTATUS[0]}; fi
    if (( rc == 0 )); then rm -f "$logf"; log "Image ready: $SIF_PATH"; return 0; fi
    rm -f "$SIF_PATH"
    if grep -qi 'proot\|ptrace' "$logf"; then
        warn "Apptainer could not use proot here; retrying with build --ignore-proot."
        if PROOT_NO_SECCOMP=1 apptainer build --ignore-proot "$SIF_PATH" "$SGLANG_IMAGE" 2>&1 | tee "$logf"
        then rc=0; else rc=${PIPESTATUS[0]}; fi
        (( rc == 0 )) || rm -f "$SIF_PATH"
    fi
    rm -f "$logf"
    return "$rc"
}

if [[ "$MODE" == "pull" ]]; then
    [[ -f "$SIF_PATH" ]] && { log "Image already present: $SIF_PATH (delete it to re-pull)."; exit 0; }
    pull_image || die "apptainer pull failed. Outbound internet from this node? Space in $APPTAINER_CACHEDIR?
  If the pinned digest has been garbage-collected upstream, pick a current
  lmsysorg/sglang:dev-kimi-k31-pro-mi35x digest (docs/compatibility.md)."
    exit 0
fi

if [[ ! -f "$SIF_PATH" ]]; then
    log "No .sif at $SIF_PATH yet — pulling it now (one-time)."
    pull_image || die "apptainer pull failed. Run './serve-kimik31.sh pull' explicitly to debug."
fi

case "$MODE" in
    serve|check|parsers|loadstat|gpucheck|flagcheck)
        aiter_config_setup
        PROBE_ENV+=(${AITER_CONFIG_ARGS[@]+"${AITER_CONFIG_ARGS[@]}"}) ;;
esac

if [[ -z "${HF_TOKEN:-}" && -n "${HF_TOKEN_FILE:-}" ]]; then
    [[ -r "$HF_TOKEN_FILE" ]] || die "HF_TOKEN_FILE '$HF_TOKEN_FILE' is not readable."
    HF_TOKEN="$(<"$HF_TOKEN_FILE")"
fi

# ── GPU passthrough ─────────────────────────────────────────────────────────
# '--rocm' binds the NODE's ROCm libs into the container ahead of its own; that
# broke K3 when the node and image ROCm diverged. 'devices' binds /dev/kfd and
# /dev/dri only. Probe both; cache the answer per node+image.

ROCM_MODES=(rocm devices)
visible_devices_ok() { [[ "$1" =~ ^[0-9]+(,[0-9]+)*$ ]]; }
sif_stamp() { stat -c '%s:%Y' "$SIF_PATH" 2>/dev/null || echo '?'; }
ROCM_MODE_CACHE="$MODEL_CACHE_DIR/.rocm-mode-k31"
rocm_cache_key() { printf '%s|%s' "$(hostname -s 2>/dev/null || echo node)" "$(sif_stamp)"; }

make_rocminfo_shim() {   # -> shim_args
    shim_args=()
    [[ "$ROCMINFO_SHIM" == "off" ]] && return 0
    local target snap shim
    target="$(apptainer exec "$SIF_PATH" sh -c 'ls -d /opt/rocm*/bin/rocminfo 2>/dev/null | head -n1' 2>/dev/null || true)"
    [[ -n "$target" ]] || return 0
    if [[ "$ROCMINFO_SHIM" != "force" ]] && apptainer exec --rocm "$SIF_PATH" "$target" >/dev/null 2>&1; then
        return 0
    fi
    command -v rocminfo >/dev/null 2>&1 || { warn "Container rocminfo fails and the host has none to copy."; return 0; }
    snap="$MODEL_CACHE_DIR/rocminfo-host-k31.txt"
    if ! rocminfo > "$snap" 2>/dev/null || ! grep -q 'gfx' "$snap"; then
        rm -f "$snap"; warn "The host's rocminfo did not produce usable output — cannot shim."; return 0
    fi
    shim="$MODEL_CACHE_DIR/rocminfo-shim-k31.sh"
    { printf '#!/bin/sh\ncat <<'\''KIMIK31_ROCMINFO_EOF'\''\n'; cat "$snap"; printf 'KIMIK31_ROCMINFO_EOF\n'; } > "$shim"
    chmod +x "$shim"
    shim_args=(--bind "$shim":"$target")
    log "rocminfo shim: the image's rocminfo fails on this node; binding the host's output over $target"
}

set_rocm_mode_args() {
    case "$1" in
        rocm)    rocm_args=(--rocm) ;;
        devices) rocm_args=(--bind /dev/kfd:/dev/kfd --bind /dev/dri:/dev/dri) ;;
        *)       die "internal: unknown ROCm mode '$1'" ;;
    esac
}

ROCM_VER_CMD='cat /opt/rocm/.info/version 2>/dev/null || ls -d /opt/rocm-* 2>/dev/null | sed -n "1s|.*/opt/rocm-||p"'
sif_rocm_ver()  { apptainer exec "$SIF_PATH" sh -c "$ROCM_VER_CMD" 2>/dev/null | head -n1; }
host_rocm_ver() {
    local v
    v="$(sh -c "$ROCM_VER_CMD" 2>/dev/null | head -n1 || true)"
    [[ -z "$v" ]] && command -v amd-smi >/dev/null 2>&1 \
        && v="$(amd-smi version 2>/dev/null | grep -o 'ROCm version: [0-9.]*' | awk '{print $3}' | head -n1 || true)"
    [[ -z "$v" ]] && v="$(command -v rocminfo 2>/dev/null | grep -o '[0-9][0-9]*\.[0-9][0-9]*\(\.[0-9][0-9]*\)\?' | tail -n1 || true)"
    printf '%s' "$v"
}

GPU_PROBE_PY='
n, gfx, err = -1, "", ""
try:
    import torch
    n = torch.cuda.device_count()
except BaseException as e:
    err = "torch: %s: %s" % (type(e).__name__, e)
if not err:
    try:
        from aiter.jit.utils.chip_info import get_gfx
        gfx = str(get_gfx() or "")
    except BaseException as e:
        err = "aiter: %s: %s" % (type(e).__name__, e)
print("KIMIK31_DEVICES %d" % n)
print("KIMIK31_GFX %s" % (gfx or "-"))
print("KIMIK31_ERR %s" % (" ".join(err.split()) or "-"))
'

gpu_probe() {   # <mode> [extra apptainer args] -> PROBE_DEVICES/GFX/ERR
    local mode="$1"; shift
    local out
    set_rocm_mode_args "$mode"
    out="$(apptainer exec "${rocm_args[@]}" "$@" "$SIF_PATH" python3 -c "$GPU_PROBE_PY" 2>&1 || true)"
    PROBE_DEVICES="$(sed -n 's/^KIMIK31_DEVICES //p' <<<"$out" | tail -n1)"
    PROBE_GFX="$(sed -n 's/^KIMIK31_GFX //p' <<<"$out" | tail -n1)"
    PROBE_ERR="$(sed -n 's/^KIMIK31_ERR //p' <<<"$out" | tail -n1)"
    if [[ -z "$PROBE_DEVICES" ]]; then
        PROBE_DEVICES="-1"; PROBE_GFX="-"
        PROBE_ERR="$(tail -n 3 <<<"$out" | tr '\n' ' ')"; PROBE_ERR="${PROBE_ERR:-container produced no output}"
    fi
    [[ "$PROBE_DEVICES" =~ ^[0-9]+$ && "$PROBE_DEVICES" -gt 0 && "$PROBE_GFX" == gfx* ]]
}

if [[ "$MODE" == "gpucheck" ]]; then
    log "Image:  $SIF_PATH"
    printf '\n  host ROCm       : %s\n' "$(host_rocm_ver || true)"
    printf '  container ROCm  : %s\n' "$(sif_rocm_ver || true)"
    printf '  amdgpu driver   : %s\n' "$(cat /sys/module/amdgpu/version 2>/dev/null || echo '?')"
    printf '  GTT per GPU     : %s GiB (%s)\n' "$GTT_GIB" "$GTT_SRC"
    printf '  MEC firmware    : %s\n' "$(cat /sys/class/drm/card*/device/fw_version/mec_fw_version 2>/dev/null | sort -u | tr '\n' ' ')"
    for v in ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES; do
        [[ -n "${!v:-}" ]] && printf '  %-16s: %s%s\n' "$v" "${!v}" \
            "$(visible_devices_ok "${!v}" || echo '   <- NOT index-form; ROCr may fail to enumerate')"
    done
    echo
    make_rocminfo_shim
    probe_extra=(${AITER_CONFIG_ARGS[@]+"${AITER_CONFIG_ARGS[@]}"} ${shim_args[@]+"${shim_args[@]}"})
    [[ -n "$AITER_GPU_ARCHS" ]] && probe_extra+=(--env "GPU_ARCHS=$AITER_GPU_ARCHS")
    gpucheck_ok=""; best_devices=0
    for m in "${ROCM_MODES[@]}"; do
        if gpu_probe "$m" ${probe_extra[@]+"${probe_extra[@]}"} "${PROBE_ENV[@]}"; then
            printf '  %-8s : OK    devices=%s gfx=%s\n' "$m" "$PROBE_DEVICES" "$PROBE_GFX"
            [[ -z "$gpucheck_ok" ]] && gpucheck_ok="$m"
        else
            printf '  %-8s : FAIL  devices=%s gfx=%s\n             %s\n' "$m" "$PROBE_DEVICES" "$PROBE_GFX" "$PROBE_ERR"
        fi
        [[ "$PROBE_DEVICES" =~ ^[0-9]+$ && "$PROBE_DEVICES" -gt "$best_devices" ]] && best_devices="$PROBE_DEVICES"
    done
    echo
    if [[ -n "$gpucheck_ok" ]]; then
        log "Verdict: use ROCM_MODE=$gpucheck_ok on this node."
        printf '%s %s\n' "$(rocm_cache_key)" "$gpucheck_ok" > "$ROCM_MODE_CACHE" 2>/dev/null || true
        exit 0
    fi
    rm -f "$ROCM_MODE_CACHE" 2>/dev/null || true
    if (( best_devices > 0 )); then
        warn "Verdict: GPUs reachable ($best_devices) but aiter cannot name the architecture — a broken
  in-image rocminfo, not a broken container. Try ROCMINFO_SHIM=force."
    else
        warn "Verdict: the container cannot reach the GPUs in any mode. The node's driver and the
  image's ROCm are incompatible; you need an image built for this node's ROCm."
    fi
    exit 1
fi

# ── Image introspection (cached, textual, no imports) ───────────────────────

IMAGE_HELP=""; IMAGE_HELP_PROBED=0
image_help() {
    if (( ! IMAGE_HELP_PROBED )); then
        IMAGE_HELP_PROBED=1
        IMAGE_HELP="$(apptainer exec "${PROBE_ENV[@]}" "$SIF_PATH" \
            bash -c "sglang serve --help 2>&1 || python3 -m sglang.launch_server --help 2>&1" 2>/dev/null || true)"
    fi
    printf '%s' "$IMAGE_HELP"
}

# Every SGLANG_*/AITER_* name the image's python trees mention. An env var the
# image never reads is silently inert, which is the safe direction but makes a
# typo or an old image cost a benchmark before anyone notices.
IMAGE_ENVS=""; IMAGE_ENVS_PROBED=0
image_envs() {
    if (( ! IMAGE_ENVS_PROBED )); then
        IMAGE_ENVS_PROBED=1
        IMAGE_ENVS="$(apptainer exec "$SIF_PATH" bash -c '
            roots=""
            for d in /sgl-workspace/sglang/python /sgl-workspace/aiter/aiter /sgl-workspace/sgl-amd-mono; do
                [[ -d "$d" ]] && roots="$roots $d"
            done
            [[ -n "$roots" ]] || exit 0
            grep -rhoE "(SGLANG|AITER|HSA|PYTORCH_HIP)_[A-Z0-9_]+" --include=*.py $roots 2>/dev/null | sort -u
            echo __probed__
        ' 2>/dev/null || true)"
    fi
    printf '%s' "$IMAGE_ENVS"
}

# flags_missing <argv...> -> prints every --flag absent from the image's --help
flags_missing() {
    local help a; help="$(image_help)"
    [[ -n "$help" ]] || { echo "__noprobe__"; return 0; }
    for a in "$@"; do
        [[ "$a" == --* ]] || continue
        grep -q -- "${a%%=*}\b" <<<"$help" || echo "$a"
    done
}

# env_inert <NAME=V...> -> prints names the image never references
env_inert() {
    local envs kv name; envs="$(image_envs)"
    grep -qx __probed__ <<<"$envs" || return 0
    for kv in "$@"; do
        name="${kv%%=*}"
        case "$name" in SGLANG_*|AITER_*) ;; *) continue ;; esac
        grep -qxF "$name" <<<"$envs" || echo "$name"
    done
}

# ── loadstat / parsers ──────────────────────────────────────────────────────

if [[ "$MODE" == "loadstat" ]]; then
    log "Weight loading report from $LOG_FILE"
    if [[ -r "$LOG_FILE" ]]; then
        if grep -qi "falling back to single-threaded" "$LOG_FILE"; then
            warn "SINGLE-THREADED weight loading was used:"
            grep -i -m1 "falling back to single-threaded" "$LOG_FILE" | sed 's/^/    /'
        else
            log "No single-threaded fallback warning found."
        fi
        grep -m1 -oE '\-\-load-format [^ ]+|\-\-model-loader-extra-config .*\}' "$LOG_FILE" | sed 's/^/    /' || true
    else
        warn "No server log at $LOG_FILE yet."
    fi
    log "Time-to-ready history ($LOADTIMES_FILE):"
    [[ -r "$LOADTIMES_FILE" ]] && tail -10 "$LOADTIMES_FILE" | sed 's/^/    /' || echo "    (nothing recorded yet)"
    log "Compare COLD runs only: 2.2 TB of host RAM can hold the 1.56 TB of weights in page cache."
    image_help | grep -A 12 -- '--load-format' | head -16 | sed 's/^/    /' || true
    exit 0
fi

if [[ "$MODE" == "parsers" ]]; then
    log "Parsers available in $SIF_PATH:"
    image_help | grep -A 6 -iE '\-\-(tool-call|reasoning)-parser' || warn "Could not read parser choices from --help."
    log "K3.1 uses the K3 parsers: kimi_k3 for both (the cookbook's choice)."
    exit 0
fi

# ── check: can this image load this model? ──────────────────────────────────

if [[ "$MODE" == "check" ]]; then
    log "Image:   $SIF_PATH"
    log "Model:   $MODEL_ID @ ${MODEL_REVISION:-main}"
    echo
    rc=0
    apptainer exec --bind "$MODEL_CACHE_DIR":"$MODEL_CACHE_DIR" \
        --env HF_HOME="$MODEL_CACHE_DIR" --env HF_TOKEN="${HF_TOKEN:-}" \
        --env MODEL_ID="$MODEL_ID" --env MODEL_REVISION="$MODEL_REVISION" \
        "${PROBE_ENV[@]}" "$SIF_PATH" python3 - <<'PY' || rc=$?
import os, sys
model_id, rev = os.environ["MODEL_ID"], os.environ.get("MODEL_REVISION") or None
try:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id, revision=rev, trust_remote_code=True)
    archs = getattr(cfg, "architectures", None) or []
    print(f"  config.json architectures : {archs}")
    src = getattr(cfg, "text_config", None) or cfg
    def g(o, k):
        return o.get(k) if isinstance(o, dict) else getattr(o, k, None)
    for k in ("num_hidden_layers", "num_experts", "num_experts_per_token", "max_position_embeddings"):
        print(f"  {k:26}: {g(src, k)}")
    dsa = g(src, "dsa_config") or {}
    print(f"  {'dsa (sparse MLA) layers':26}: {len(g(dsa,'dsa_layers') or [])}, index_topk={g(dsa,'index_topk')}")
except Exception as e:
    print(f"  !! could not load config for {model_id}: {e}\n     (gated, or no network from this node?)")
    sys.exit(2)
print()
from sglang.srt.models.registry import ModelRegistry  # type: ignore
known = set(getattr(ModelRegistry, "models", {}) or {})
missing = [a for a in archs if a not in known]
for a in archs:
    print(f"  {'OK  ' if a in known else 'MISS'} SGLang registry {'knows' if a in known else 'does NOT know'} '{a}'")
print(f"\n  ({len(known)} architectures registered; Kimi-family: {sorted(a for a in known if 'kimi' in a.lower())})")
if not known:
    print("\n  => INCONCLUSIVE: the registry came back EMPTY — every model module failed to import.")
    print("     Look for 'Ignore import error when loading sglang.srt.models.*' above.")
    sys.exit(3)
if missing:
    print("\n  => This image CANNOT serve this model. Use the dev-kimi-k31-pro-mi35x image.")
    sys.exit(1)
print("\n  => This image can load this model.")
PY
    echo
    log "K3.1 / MI355X levers in this image:"
    envs="$(image_envs)"
    lever() {  # <label> <present?>
        if [[ "$2" == 1 ]]; then log "  OK   $1"; else warn "  MISS $1"; fi
    }
    h="$(image_help)"
    lever "--enable-amd-monokernel (sgl-amd-mono persistent decode/verify kernels)" \
        "$(grep -q -- '--enable-amd-monokernel' <<<"$h" && echo 1 || echo 0)"
    lever "SGLANG_AMD_MONO_K31_WIDE (K3.1 wide monokernel)" "$(grep -qx SGLANG_AMD_MONO_K31_WIDE <<<"$envs" && echo 1 || echo 0)"
    lever "DFLASH speculative algorithm" "$(grep -q 'DFLASH' <<<"$h" && echo 1 || echo 0)"
    lever "--speculative-use-block-verification" "$(grep -q -- '--speculative-use-block-verification' <<<"$h" && echo 1 || echo 0)"
    lever "--enable-linear-replayssm-spec (ReplaySSM ring for KDA verify)" "$(grep -q -- '--enable-linear-replayssm-spec' <<<"$h" && echo 1 || echo 0)"
    lever "--dcp-size / a2a (decode context parallel)" "$(grep -q -- '--dcp-comm-backend' <<<"$h" && echo 1 || echo 0)"
    lever "SGLANG_ROCM_RUNTIME_RESERVE_GB" "$(grep -qx SGLANG_ROCM_RUNTIME_RESERVE_GB <<<"$envs" && echo 1 || echo 0)"
    lever "row-sharded prefill (SGLANG_ENABLE_ROCM_K3_ROW_SHARDED_*)" \
        "$(grep -qx SGLANG_ENABLE_ROCM_K3_ROW_SHARDED_STREAM <<<"$envs" && echo 1 || echo 0)"
    hip_dsa="$(apptainer exec "$SIF_PATH" bash -c 'grep -rhoE "_KIMI_K31_DSA_BACKEND_HIP *= *\"[a-z_]+\"" /sgl-workspace/sglang/python/sglang/srt/arg_groups 2>/dev/null | head -1' 2>/dev/null || true)"
    [[ -n "$hip_dsa" ]] && log "  DSA backend pinned on ROCm: ${hip_dsa##*=}"
    log "  container ROCm: $(sif_rocm_ver || echo '?')   node ROCm: $(host_rocm_ver || echo '?')"
    echo
    case "$rc" in
        0) log "check passed." ;;
        1) warn "check FAILED: this image cannot serve $MODEL_ID." ;;
        2) warn "check INCONCLUSIVE: could not read the model config." ;;
        3) warn "check INCONCLUSIVE: the image's model registry would not import." ;;
        *) warn "check exited with status $rc." ;;
    esac
    exit "$rc"
fi

# ── flagcheck: every recipe flag + env against the image ────────────────────

recipe_preflight() {   # dies on a flag the image does not have
    local missing inert
    MODEL_PATH="${MODEL_PATH:-x}"; DRAFT_PATH="${DRAFT_PATH:-x}"; API_KEY_ARG="${API_KEY_ARG:-x}"
    [[ -n "${LOADER_CFG:-}" ]] || make_loader_cfg
    build_cmd
    # shellcheck disable=SC2207
    missing=($(flags_missing "${cmd[@]}"))
    if [[ "${missing[*]:-}" == "__noprobe__" ]]; then
        warn "Could not read --help from the image; launching with the recipe flags unverified."
    elif (( ${#missing[@]} )); then
        die "This image does not accept: ${missing[*]}
  These come from the $PROFILE recipe (or EXTRA_ENGINE_ARGS). SGLang would reject
  them only AFTER the container is up. Use the pinned dev-kimi-k31-pro-mi35x image,
  or override the recipe (see docs/configuration.md)."
    fi
    # shellcheck disable=SC2207
    inert=($(env_inert "${R_ENV[@]}"))
    (( ${#inert[@]} )) && warn "These recipe env vars are never read by this image (inert): ${inert[*]}"
    return 0
}

if [[ "$MODE" == "flagcheck" ]]; then
    log "Recipe: $(recipe_summary)"
    recipe_preflight
    log "flagcheck passed: the image accepts every flag of this recipe."
    exit 0
fi

# ── Weights ─────────────────────────────────────────────────────────────────

repo_dir() { printf '%s/hub/models--%s' "$MODEL_CACHE_DIR" "${1//\//--}"; }

# resolve_snapshot <repo> <revision> -> path on stdout, or nothing
resolve_snapshot() {
    local dir rev="$2" ref
    dir="$(repo_dir "$1")"
    if [[ "$rev" =~ ^[0-9a-f]{40}$ ]]; then
        [[ -d "$dir/snapshots/$rev" ]] && printf '%s' "$dir/snapshots/$rev"
        return 0
    fi
    ref="$dir/refs/${rev:-main}"
    [[ -r "$ref" && -d "$dir/snapshots/$(<"$ref")" ]] && printf '%s' "$dir/snapshots/$(<"$ref")"
    return 0
}

if [[ "$MODE" == "download" ]]; then
    free_gb="$(df -Pk "$MODEL_CACHE_DIR" | awk 'NR==2 {print int($4/1024/1024)}')"
    (( ${free_gb:-0} >= NEED_GB )) || warn "Only ${free_gb} GB free in $MODEL_CACHE_DIR; $MODEL_ID is ~${WEIGHTS_GB} GB. Check 'rquota'."
    dl() {   # <repo> <rev>
        local rev_arg=""; [[ -n "$2" ]] && rev_arg=" --revision '$2'"
        log "Prefetching $1${2:+ @ ${2:0:8}} into $MODEL_CACHE_DIR ..."
        apptainer exec --bind "$MODEL_CACHE_DIR":"$MODEL_CACHE_DIR" \
            --env HF_HOME="$MODEL_CACHE_DIR" --env HF_TOKEN="${HF_TOKEN:-}" \
            --env HF_XET_HIGH_PERFORMANCE=1 \
            "$SIF_PATH" bash -c "hf download '$1'$rev_arg || huggingface-cli download '$1'$rev_arg"
    }
    dl "$MODEL_ID" "$MODEL_REVISION" || die "Weights download failed. Re-run to resume."
    if [[ "$SPECULATIVE" == "dflash" ]]; then
        dl "$DFLASH_MODEL" "$DFLASH_REVISION" || die "DFLASH draft download failed."
    fi
    log "Download complete."
    exit 0
fi

MODEL_PATH="$(resolve_snapshot "$MODEL_ID" "$MODEL_REVISION")"
if [[ -z "$MODEL_PATH" ]]; then
    die "$MODEL_ID${MODEL_REVISION:+ @ $MODEL_REVISION} is not in $MODEL_CACHE_DIR/hub.
  Fetch it first (no GPU needed, ~${WEIGHTS_GB} GB):   ./serve-kimik31.sh download
  Another revision of the same repo only transfers the files that differ."
fi
# A present directory is not a complete checkpoint: a half-finished download
# leaves dangling symlinks. Count shards against the index before a 1.5 TB load.
idx="$MODEL_PATH/model.safetensors.index.json"
[[ -e "$idx" ]] || die "No model.safetensors.index.json in $MODEL_PATH — the download is incomplete. Re-run './serve-kimik31.sh download'."
shards_want="$(python3 -c 'import json,sys; print(len(set(json.load(open(sys.argv[1]))["weight_map"].values())))' "$idx" 2>/dev/null || echo 0)"
shards_have="$(find -L "$MODEL_PATH" -maxdepth 1 -name '*.safetensors' -type f 2>/dev/null | wc -l)"
if (( shards_want > 0 && shards_have < shards_want )); then
    die "Only $shards_have of $shards_want weight shards are present in $MODEL_PATH.
  The download is incomplete (or still running). Re-run './serve-kimik31.sh download'."
fi
log "Weights: $MODEL_ID -> $(basename "$MODEL_PATH") ($shards_have shards)"

DRAFT_PATH=""
if [[ "$SPECULATIVE" == "dflash" ]]; then
    DRAFT_PATH="$(resolve_snapshot "$DFLASH_MODEL" "$DFLASH_REVISION")"
    # The draft loads only AFTER the 1.5 TB target; a missing draft found then
    # wastes the whole load (K3, 1 Aug 2026).
    [[ -n "$DRAFT_PATH" && -e "$DRAFT_PATH/config.json" && -e "$DRAFT_PATH/model.safetensors" ]] \
        || die "SPECULATIVE=dflash but the draft $DFLASH_MODEL${DFLASH_REVISION:+ @ ${DFLASH_REVISION:0:8}} is not (fully) cached.
  Fetch it:   ./serve-kimik31.sh download      or serve without it:   SPECULATIVE=none"
    draft_info="$(python3 -c '
import json,sys; c=json.load(open(sys.argv[1]))
print(c.get("model_type",""), (c.get("dflash_config") or {}).get("block_size", c.get("block_size","")), (c.get("rope_parameters") or {}).get("factor",""))' "$DRAFT_PATH/config.json" 2>/dev/null || true)"
    read -r d_type d_block d_rope <<<"$draft_info"
    [[ -n "${d_type:-}" ]] || die "The draft config at $DRAFT_PATH/config.json is unparseable — re-fetch it."
    log "Draft:   $DFLASH_MODEL -> $(basename "$DRAFT_PATH") (trained block ${d_block:-?}, serving block $DFLASH_BLOCK_SIZE, YaRN x${d_rope:-?})"
fi

# ── Serve-mode preflight ────────────────────────────────────────────────────

[[ -e /dev/kfd ]] || die "/dev/kfd not found — is this a ROCm GPU node?"
[[ -e /dev/dri ]] || die "/dev/dri not found — is this a ROCm GPU node?"
if command -v rocminfo >/dev/null 2>&1; then
    gfx="$(rocminfo 2>/dev/null | grep -o 'gfx[0-9a-f]*' | head -n1 || true)"
    case "$gfx" in
        gfx950) log "Detected gfx950 (MI350X/MI355X)." ;;
        "")     warn "Could not detect GPU arch from rocminfo." ;;
        *)      warn "Detected $gfx — K3.1 needs ~1.5 TB of MXFP4 weights and gfx950; this recipe is MI355X-only." ;;
    esac
fi
host_rocm="$(host_rocm_ver || true)"; sif_rocm="$(sif_rocm_ver || true)"
[[ -n "$host_rocm$sif_rocm" ]] && log "ROCm: node ${host_rocm:-?} / container ${sif_rocm:-?}"
if [[ -n "$sif_rocm" && "${sif_rocm%%.*}" != "10" ]]; then
    warn "Container ROCm is $sif_rocm. Upstream: 'Requires ROCm 10; the recipes have not run on ROCm 7 images.'"
fi

GPU_VIS="${ROCR_VISIBLE_DEVICES:-${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}}"
alloc_count=""
[[ -n "$GPU_VIS" ]] && alloc_count="$(awk -F, '{print NF}' <<<"$GPU_VIS")"
[[ -z "$alloc_count" && -n "${SLURM_GPUS_ON_NODE:-}" ]] && alloc_count="$SLURM_GPUS_ON_NODE"
if [[ -n "$alloc_count" ]] && is_uint "$alloc_count" && (( alloc_count > 0 && TP_SIZE != alloc_count )); then
    warn "TP_SIZE=$TP_SIZE but $alloc_count GPU(s) are allocated. K3.1 needs all 8."
fi
if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
    die "Port $PORT is already in use on this node (another server? './serve-kimik31.sh status')."
fi
server_running && die "A server is already recorded running (pid $(<"$PID_FILE")). Use 'status' or 'stop' first."

# ── API key ─────────────────────────────────────────────────────────────────

if [[ -z "${KIMIK31_API_KEY:-}" ]]; then
    if [[ -r "$API_KEY_FILE" ]]; then
        KIMIK31_API_KEY="$(<"$API_KEY_FILE")"; log "Using API key from $API_KEY_FILE"
    else
        KIMIK31_API_KEY="$(openssl rand -hex 24 2>/dev/null || head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')"
        (umask 077 && printf '%s' "$KIMIK31_API_KEY" > "$API_KEY_FILE")
        log "Generated new API key and saved it to $API_KEY_FILE"
    fi
fi
API_KEY_ARG="$KIMIK31_API_KEY"

# ── aiter JIT dir: writable copy of the image's aiter/jit ───────────────────
# FlyDSL kernels JIT-compile into aiter/jit inside the read-only .sif at graph
# capture — AFTER the weight load. Seed a scratch copy from the image (keeping
# its prebuilt module_*.so), bind it back at the same path, re-seed when the
# image changes (stale copies give undefined-symbol crashes).
AITER_JIT_DIR="${AITER_JIT_DIR:-$MODEL_CACHE_DIR/aiter-jit-k31}"
detected_jit="$(apptainer exec "$SIF_PATH" python3 - <<'PY' 2>/dev/null | tail -n 1
import importlib.util, os
spec = importlib.util.find_spec("aiter")
origin = getattr(spec, "origin", None) if spec is not None else None
print(os.path.join(os.path.dirname(origin), "jit") if origin else "")
PY
)"
if [[ -n "${AITER_JIT_TARGET:-}" ]]; then
    apptainer exec "$SIF_PATH" test -d "$AITER_JIT_TARGET" 2>/dev/null \
        || die "AITER_JIT_TARGET='$AITER_JIT_TARGET' does not exist inside the image (detected: ${detected_jit:-?})."
else
    AITER_JIT_TARGET="$detected_jit"
fi
cache_bind=()
if [[ "${AITER_JIT_TARGET:-}" != /* ]]; then
    warn "Could not locate aiter's jit/ in the image — startup will likely die at graph capture
  with 'Read-only file system: .../aiter/jit/...'. Set AITER_JIT_TARGET."
else
    mkdir -p "$AITER_JIT_DIR" || die "Cannot create $AITER_JIT_DIR"
    seed_stamp="$SGLANG_IMAGE|$AITER_JIT_TARGET|$(sif_stamp)"
    if [[ "$(cat "$AITER_JIT_DIR/.seeded" 2>/dev/null || true)" != "$seed_stamp" ]]; then
        case "$AITER_JIT_DIR" in ""|/|"$HOME"|"$MODEL_CACHE_DIR") die "Refusing to clear AITER_JIT_DIR='$AITER_JIT_DIR'." ;; esac
        [[ -e "$AITER_JIT_DIR/.seeded" ]] && log "Image changed since $AITER_JIT_DIR was seeded — re-seeding."
        rm -rf "${AITER_JIT_DIR:?}"/* "${AITER_JIT_DIR:?}"/.[!.]* 2>/dev/null || true
        log "Seeding writable aiter JIT dir from the image (one-off copy) ..."
        apptainer exec --bind "$MODEL_CACHE_DIR":"$MODEL_CACHE_DIR" "$SIF_PATH" \
            cp -a "$AITER_JIT_TARGET/." "$AITER_JIT_DIR/" || die "Failed to copy $AITER_JIT_TARGET into $AITER_JIT_DIR"
        printf '%s\n' "$seed_stamp" > "$AITER_JIT_DIR/.seeded"
        log "Seeded $AITER_JIT_DIR ($(du -sh "$AITER_JIT_DIR" 2>/dev/null | cut -f1))"
    fi
    cache_bind=(--bind "$AITER_JIT_DIR":"$AITER_JIT_TARGET")
fi

# ── Recipe preflight against the image (before any weight is read) ──────────

make_loader_cfg
log "Recipe: $(recipe_summary)"
[[ -n "$HICACHE_NOTE" ]] && log "HiCache: $HICACHE_NOTE"
[[ "$SPECULATIVE" == "none" ]] && warn "SPECULATIVE=none: upstream verified every MI355X cell WITH DFLASH; non-spec is 'in progress'."
recipe_preflight
build_cmd

# ── Container argv ──────────────────────────────────────────────────────────

make_rocminfo_shim
presharded_bind=()
if [[ "$LOAD_FORMAT" == "presharded" && -n "$PRESHARDED_PATH" ]]; then
    mkdir -p "$PRESHARDED_PATH" 2>/dev/null || true
    [[ "$SPECULATIVE" == "dflash" ]] && mkdir -p "$PRESHARDED_PATH/dflash" 2>/dev/null || true
    [[ -d "$PRESHARDED_PATH" && -w "$PRESHARDED_PATH" ]] || die "PRESHARDED_PATH '$PRESHARDED_PATH' is not writable."
    [[ "$PRESHARDED_PATH" != "$MODEL_CACHE_DIR"/* ]] && presharded_bind=(--bind "$PRESHARDED_PATH":"$PRESHARDED_PATH")
fi

# ── TTM pages_limit shim (HiCache only) ─────────────────────────────────────
# SGLang caps each rank's HiCache host pool at its share of the amdgpu TTM page
# limit, read from /sys/module/ttm/parameters/pages_limit. On Bunya that reads 0
# ("driver default" = half of RAM, i.e. the 1133 GiB GTT), and SGLang takes the 0
# literally: the budget goes negative and EVERY HiCache launch dies after the
# weight load with "Not enough host memory available ... only have -4.29 GB free"
# (hit 10 Nov 2026). Bind a file holding the node's real GTT limit, in pages,
# over that path — same pattern as the rocminfo shim.
ttm_args=()
if [[ "$USE_HICACHE" == "on" && "$TTM_SHIM" == "auto" ]]; then
    host_ttm="$(cat /sys/module/ttm/parameters/pages_limit 2>/dev/null || true)"
    if [[ -z "$host_ttm" || "$host_ttm" == "0" ]]; then
        gtt_bytes=""
        for f in /sys/class/drm/card*/device/mem_info_gtt_total; do
            [[ -r "$f" ]] && { gtt_bytes="$(<"$f")"; break; }
        done
        if is_uint "${gtt_bytes:-x}" && (( gtt_bytes > 0 )); then
            ttm_file="$MODEL_CACHE_DIR/ttm-pages-limit-k31"
            echo $(( gtt_bytes / $(getconf PAGESIZE) )) > "$ttm_file"
            ttm_args=(--bind "$ttm_file":/sys/module/ttm/parameters/pages_limit)
            log "TTM shim: host pages_limit reads '${host_ttm:-unreadable}'; binding the real GTT limit ($(( gtt_bytes / 1073741824 )) GiB) for SGLang's HiCache budget."
        else
            warn "HiCache is on but neither ttm pages_limit nor mem_info_gtt_total is usable — SGLang will likely refuse the host pool."
        fi
    fi
fi

gpu_env=()
for v in ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES; do
    val="${!v:-}"; [[ -z "$val" ]] && continue
    if visible_devices_ok "$val"; then gpu_env+=(--env "$v=$val")
    else warn "$v='$val' is not an index list — not forwarding it (an unparseable value stops ROCr enumerating GPUs)."; fi
done

recipe_env_args=()
for e in "${R_ENV[@]}"; do recipe_env_args+=(--env "$e"); done

base_args=(--bind "$MODEL_CACHE_DIR":"$MODEL_CACHE_DIR"
    ${cache_bind[@]+"${cache_bind[@]}"}
    ${AITER_CONFIG_ARGS[@]+"${AITER_CONFIG_ARGS[@]}"}
    ${shim_args[@]+"${shim_args[@]}"}
    ${presharded_bind[@]+"${presharded_bind[@]}"}
    ${ttm_args[@]+"${ttm_args[@]}"}
    --bind "$SGLANG_CACHE_DIR":"$SGLANG_CACHE_DIR"
    --env SGLANG_CACHE_DIR="$SGLANG_CACHE_DIR"
    --env HF_HOME="$MODEL_CACHE_DIR"
    --env HF_HUB_OFFLINE=1
    --env HF_TOKEN="${HF_TOKEN:-}"
    --env SGLANG_SET_CPU_AFFINITY="$SET_CPU_AFFINITY"
    "${recipe_env_args[@]}"
    ${gpu_env[@]+"${gpu_env[@]}"})

# ── GPU passthrough mode ────────────────────────────────────────────────────

chosen_mode=""; cache_key="$(rocm_cache_key)"
if [[ "$ROCM_MODE" != "auto" ]]; then
    gpu_probe "$ROCM_MODE" "${base_args[@]}" && chosen_mode="$ROCM_MODE" \
        || die "ROCM_MODE=$ROCM_MODE cannot reach the GPUs (devices=$PROBE_DEVICES gfx=$PROBE_GFX: $PROBE_ERR). Run gpucheck."
elif [[ "$(cut -d' ' -f1 "$ROCM_MODE_CACHE" 2>/dev/null || true)" == "$cache_key" ]]; then
    chosen_mode="$(cut -d' ' -f2 "$ROCM_MODE_CACHE" 2>/dev/null || true)"
    log "GPU passthrough: $chosen_mode (remembered for this node+image; delete $ROCM_MODE_CACHE to re-probe)"
else
    log "Probing GPU passthrough modes (~1 min; cached per node+image) ..."
    for m in "${ROCM_MODES[@]}"; do
        if gpu_probe "$m" "${base_args[@]}"; then
            log "  $m: OK — $PROBE_DEVICES GPU(s), $PROBE_GFX"; chosen_mode="$m"; break
        fi
        warn "  $m: devices=$PROBE_DEVICES gfx=$PROBE_GFX — $PROBE_ERR"
    done
    [[ -n "$chosen_mode" ]] && printf '%s %s\n' "$cache_key" "$chosen_mode" > "$ROCM_MODE_CACHE" 2>/dev/null || true
fi
[[ -n "$chosen_mode" ]] || die "No GPU passthrough mode works on this node. Run './serve-kimik31.sh gpucheck'."
set_rocm_mode_args "$chosen_mode"
apptainer_args=(exec "${rocm_args[@]}" "${base_args[@]}")

if [[ "${AITER_JIT_TARGET:-}" == /* ]]; then
    probe="$AITER_JIT_TARGET/flydsl_cache/.write-test.$$"
    probe_err="$(apptainer "${apptainer_args[@]}" "$SIF_PATH" \
        sh -c "mkdir -p '$AITER_JIT_TARGET/flydsl_cache' && touch '$probe' && rm -f '$probe'" 2>&1)" \
        || die "aiter's JIT dir is NOT writable inside the container ($probe): ${probe_err:-no output}"
    log "Preflight OK: $AITER_JIT_TARGET is writable in the container."
fi

# ── Launch ──────────────────────────────────────────────────────────────────

redact_args() {
    local out=() prev="" a
    for a in "$@"; do
        if [[ "$prev" == "--api-key" ]]; then out+=("<redacted>")
        elif [[ "$a" == HF_TOKEN=* ]]; then out+=("HF_TOKEN=<redacted>")
        else out+=("$a"); fi
        prev="$a"
    done
    printf '%s' "${out[*]}"
}

: > "$LOG_FILE"
{
    printf '### serve-kimik31.sh  %s  host=%s job=%s\n' "$(date)" "$(hostname -s)" "${SLURM_JOB_ID:-}"
    printf '### recipe: %s\n' "$(recipe_summary)"
    printf '### image=%s\n' "$SGLANG_IMAGE"
    printf '### apptainer %s %s\n' "$(redact_args "${apptainer_args[@]}")" "$SIF_PATH"
    printf '### command: %s\n\n' "$(redact_args "${cmd[@]}")"
} >> "$LOG_FILE"

log "Starting SGLang: $MODEL_ID  ($(recipe_summary))"
log "Image: $SIF_PATH"
log "Logs:  $LOG_FILE"

apptainer "${apptainer_args[@]}" "$SIF_PATH" "${cmd[@]}" >>"$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"

cleanup() {
    log "Shutting down server ..."
    kill "$SERVER_PID" 2>/dev/null || true
    pkill -u "$(id -u)" -f 'sglang.launch_server' 2>/dev/null || true
    pkill -u "$(id -u)" -f 'sglang serve'         2>/dev/null || true
    rm -f "$PID_FILE"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

log "Waiting for the server to become healthy (timeout ${READY_TIMEOUT}s)."
log "A cold start reads ~1.5 TB off scratch, then captures prefill + decode graphs."
start_ts="$(date +%s)"
while true; do
    if curl -fsS -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        ready_secs=$(( $(date +%s) - start_ts ))
        ready_gbs="$(awk -v g="$WEIGHTS_GB" -v s="$ready_secs" 'BEGIN {printf "%.2f", (s > 0 ? g / s : 0)}')"
        log "Ready in ${ready_secs}s (${ready_gbs} GB/s effective over ${WEIGHTS_GB} GB)"
        printf '%s\t%s\tprofile=%s\tagents=%s\tspec=%s\tthreads=%s\tformat=%s\t%ss\t%s GB/s\n' \
            "$(date -Is)" "$(hostname -s)" "$PROFILE" "$AGENTS" "$SPECULATIVE" \
            "${WEIGHT_LOAD_THREADS:-auto}" "${LOAD_FORMAT:-auto}" "$ready_secs" "$ready_gbs" \
            >> "$LOADTIMES_FILE" 2>/dev/null || true
        grep -qi "falling back to single-threaded" "$LOG_FILE" 2>/dev/null \
            && warn "SGLang fell back to SINGLE-THREADED weight loading. See './serve-kimik31.sh loadstat'."
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo; tail -n 60 "$LOG_FILE" 2>&1 || true
        rm -f "$PID_FILE"
        die "Server process exited during startup. Last 60 log lines above ($LOG_FILE).
     hipHostMalloc / GTT failure?   lower HICACHE_GTT_BUDGET or set HICACHE=off
     KV/graph OOM?                   lower MEM_FRACTION (0.85 is upstream's) or CONTEXT_LEN=262144
     unknown flag?                   ./serve-kimik31.sh flagcheck
     /tmp/aiter_configs denied?      another user owns it on this node (docs/troubleshooting.md)"
    fi
    (( $(date +%s) - start_ts > READY_TIMEOUT )) && die "Server not healthy within ${READY_TIMEOUT}s. Check: tail -f $LOG_FILE"
    sleep 10
done

NODE_HOST="$(hostname -s 2>/dev/null || hostname)"
JOBID="${SLURM_JOB_ID:-<jobid>}"
cat <<EOF

============================================================================
  $SERVED_MODEL_NAME is up and serving on Bunya's MI355X.

  Model:       $MODEL_ID @ ${MODEL_REVISION:0:8}
  Recipe:      $(recipe_summary)
  Node:        $NODE_HOST     (job $JOBID)
  Endpoint:    http://$NODE_HOST:$PORT/v1   (OpenAI-compatible)
  Model name:  $SERVED_MODEL_NAME
  API key:     $API_KEY_FILE
               export KIMIK31_API_KEY="\$(cat $API_KEY_FILE)"

  K3.1 always thinks. Set depth per request with the top-level field
  "reasoning_effort": "low" | "high" | "max" (default max), and pass the whole
  assistant message (reasoning_content + tool_calls) back on the next turn.

  Verify:      ./serve-kimik31.sh toolcheck
               ./serve-kimik31.sh status     (monokernel engaged? DFLASH accept len)
  Second shell into this job:
               srun --overlap --jobid $JOBID --pty /bin/bash -l
  From your laptop:
               ssh -N -L $PORT:$NODE_HOST:$PORT \${USER}@bunya1.rcc.uq.edu.au
  Clients:     ./opencode-setup.sh --host $NODE_HOST --port $PORT
               ./kimicode-setup.sh --host $NODE_HOST --port $PORT
  Benchmark:   ./bench-kimik31.sh sweep      ./agentstress.py (real agentic shape)

  Stop with Ctrl-C, 'scancel $JOBID', or './serve-kimik31.sh stop'.
============================================================================

EOF

if [[ "$DETACH" -eq 1 ]]; then
    trap - EXIT INT TERM
    disown "$SERVER_PID" 2>/dev/null || true
    log "Detached. The server keeps running on this node until 'stop' or the job ends."
    log "  Logs:  tail -f $LOG_FILE"
    exit 0
fi

log "Attached. Ctrl-C (or scancel) stops the server. Streaming logs:"
tail -f "$LOG_FILE" &
TAIL_PID=$!
wait "$SERVER_PID"
kill "$TAIL_PID" 2>/dev/null || true
