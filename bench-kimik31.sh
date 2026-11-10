#!/usr/bin/env bash
#
# bench-kimik31.sh — measure Kimi K3.1 Pro serving (tok/s, TTFT, TPOT/ITL) with
# SGLang's bench client, run inside the same .sif against the live endpoint.
#
# Usage:
#   ./bench-kimik31.sh                 sweep BENCH_CONCURRENCY (default 1 2 4)
#   ./bench-kimik31.sh latency         single stream (concurrency 1)
#   ./bench-kimik31.sh throughput      saturate at BENCH_MAX_CONCURRENCY
#   ./bench-kimik31.sh longcontext     100k in / 512 out, concurrency 1, n=4
#
# Tunables (env or kimik31.env): BENCH_INPUT_LEN, BENCH_OUTPUT_LEN,
#   BENCH_NUM_PROMPTS, BENCH_CONCURRENCY, BENCH_MAX_CONCURRENCY, BENCH_REPEATS,
#   BENCH_RANGE_RATIO, BENCH_EXTRA_ARGS (appended verbatim).
#
# READ THIS BEFORE QUOTING A NUMBER. With DFLASH on (the default), random-token
# prompts make the draft look far better than real text does — the K3 repo
# measured accept ~7 on the bench vs ~2 on real agent text with DSpark, which
# flipped the conclusion at long context. Use ./agentstress.py for the agentic
# shape, and compare 'accept len' in the server log between the two.
#
# Upstream's MI355X reference for K3.1 is AgentX (real agent traces, DFLASH
# acceptance simulated at 3.399), not a random sweep: 16 agents 60.2 tok/s/user,
# 32 → 36.3, 48 → 28.6, 56 → 25.6 (1x8, HiCache). Not comparable to this tool.
#
# Pure HTTP client, but Apptainer exists only on compute nodes: run it on the
# serving node (srun --overlap --jobid <jobid> --pty /bin/bash -l).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="sweep"
for arg in "$@"; do
    case "$arg" in
        latency|throughput|sweep|longcontext) MODE="$arg" ;;
        *) printf 'Unknown argument: %s (use latency | throughput | sweep | longcontext)\n' "$arg" >&2; exit 1 ;;
    esac
done

# Capture the caller's shape BEFORE the config is sourced: kimik31.env sets all
# of these unconditionally, and a named shape mode must outrank it (K3, 9 Aug
# 2026: 'longcontext' silently ran at 1024/512 because of exactly this).
for _v in BENCH_INPUT_LEN BENCH_OUTPUT_LEN BENCH_NUM_PROMPTS BENCH_CONCURRENCY; do
    declare "CLI_$_v=${!_v:-}"
done

log()  { printf '\033[1;34m[bench]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[bench WARN]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[bench ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

ENV_FILE="${KIMIK31_ENV:-$SCRIPT_DIR/kimik31.env}"
# shellcheck disable=SC1090
[[ -f "$ENV_FILE" ]] && source "$ENV_FILE"

# If the caller had already sourced kimik31.env, the capture above recorded the
# config file's values as if typed. An exact match with what the file sets is
# treated as "from the config", so the named mode still wins (K3, 12 Aug 2026).
if [[ -f "$ENV_FILE" ]]; then
    cfg_shape="$(env -u BENCH_INPUT_LEN -u BENCH_OUTPUT_LEN -u BENCH_NUM_PROMPTS -u BENCH_CONCURRENCY \
        bash -c 'source "$1" >/dev/null 2>&1
                 printf "%s\n%s\n%s\n%s\n" "${BENCH_INPUT_LEN:-}" "${BENCH_OUTPUT_LEN:-}" \
                     "${BENCH_NUM_PROMPTS:-}" "${BENCH_CONCURRENCY:-}"' _ "$ENV_FILE" 2>/dev/null || true)"
    { read -r CFG_BENCH_INPUT_LEN; read -r CFG_BENCH_OUTPUT_LEN
      read -r CFG_BENCH_NUM_PROMPTS; read -r CFG_BENCH_CONCURRENCY; } <<<"$cfg_shape"
    for _v in BENCH_INPUT_LEN BENCH_OUTPUT_LEN BENCH_NUM_PROMPTS BENCH_CONCURRENCY; do
        _cli="CLI_$_v"; _cfg="CFG_$_v"
        [[ -n "${!_cli:-}" && "${!_cli}" == "${!_cfg:-}" ]] && declare "$_cli="
    done
fi

PORT="${PORT:-30000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-kimi-k3.1-pro}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-}"
[[ -n "$MODEL_CACHE_DIR" ]] || die "MODEL_CACHE_DIR is not set (source kimik31.env or export it)."
SIF_PATH="${SIF_PATH:-$MODEL_CACHE_DIR/kimik31-mi355x.sif}"
[[ "$SIF_PATH" == */ || -d "$SIF_PATH" ]] && SIF_PATH="${SIF_PATH%/}/kimik31-mi355x.sif"
[[ -f "$SIF_PATH" ]] || die "No .sif at $SIF_PATH (run './serve-kimik31.sh pull' first)."
command -v apptainer >/dev/null 2>&1 || die "apptainer not found — run this on the serving compute node."

if [[ "$MODE" == "longcontext" ]]; then
    BENCH_INPUT_LEN="${CLI_BENCH_INPUT_LEN:-100000}"
    BENCH_OUTPUT_LEN="${CLI_BENCH_OUTPUT_LEN:-512}"
    BENCH_NUM_PROMPTS="${CLI_BENCH_NUM_PROMPTS:-4}"
    BENCH_CONCURRENCY="${CLI_BENCH_CONCURRENCY:-1}"
else
    BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-1024}"
    BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-512}"
    BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-200}"
    BENCH_CONCURRENCY="${BENCH_CONCURRENCY:-1 2 4}"
fi
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-4}"
BENCH_EXTRA_ARGS="${BENCH_EXTRA_ARGS:-}"
# sglang's random dataset defaults to ratio 0.0 = requests average HALF their
# nominal size (K3, 29 Jul 2026). 1.0 = exactly the configured lengths.
BENCH_RANGE_RATIO="${BENCH_RANGE_RATIO:-1.0}"
BENCH_REPEATS="${BENCH_REPEATS:-1}"
for v in BENCH_INPUT_LEN BENCH_OUTPUT_LEN BENCH_NUM_PROMPTS BENCH_REPEATS; do
    [[ "${!v}" =~ ^[0-9]+$ ]] || die "$v must be a positive integer, got '${!v}'."
done
if [[ "${CONTEXT_LEN:-}" =~ ^[0-9]+$ ]] && (( BENCH_INPUT_LEN + BENCH_OUTPUT_LEN > CONTEXT_LEN )); then
    die "This shape needs $((BENCH_INPUT_LEN + BENCH_OUTPUT_LEN)) tokens but CONTEXT_LEN=$CONTEXT_LEN."
fi

if [[ -z "${KIMIK31_API_KEY:-}" && -r "$MODEL_CACHE_DIR/kimik31-api-key" ]]; then
    KIMIK31_API_KEY="$(<"$MODEL_CACHE_DIR/kimik31-api-key")"
fi
[[ -n "${KIMIK31_API_KEY:-}" ]] || warn "No API key resolved — bench will 401 (set KIMIK31_API_KEY)."

curl -fsS -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 \
    || die "No healthy server on http://127.0.0.1:${PORT}. Start it with './serve-kimik31.sh serve --detach' first."
log "Server on port $PORT is healthy."

LOG_FILE="${LOG_FILE:-$MODEL_CACHE_DIR/kimik31-server.log}"
recipe=""
if [[ -r "$LOG_FILE" ]]; then
    recipe="$(sed -n 's/^### recipe: //p' "$LOG_FILE" | head -1)"
    [[ -n "$recipe" ]] && log "Server recipe: $recipe"
    srv_maxrun="$(grep -oE 'max-running=[0-9]+' <<<"$recipe" | cut -d= -f2 || true)"
    if [[ -n "$srv_maxrun" ]]; then
        for c in $BENCH_CONCURRENCY $([[ "$MODE" == throughput ]] && echo "$BENCH_MAX_CONCURRENCY"); do
            if (( c > srv_maxrun )); then warn "Concurrency $c exceeds the server's max-running=$srv_maxrun: the excess queues and TTFT measures the queue. Restart with a larger AGENTS to measure it."; fi
        done
    fi
    if grep -qiE 'no tuned FlyDSL config|heuristic FlyDSL fallback' "$LOG_FILE"; then
        warn "MoE is on the SLOW heuristic FlyDSL fallback — check ENABLE_AITER=1."
    fi
fi
[[ "$recipe" == *"spec=dflash"* ]] && warn "DFLASH is on: random-token prompts inflate acceptance. Treat TPOT here as optimistic; confirm with ./agentstress.py."

BENCH_DIR="$MODEL_CACHE_DIR/bench-k31"
mkdir -p "$BENCH_DIR"
OUT_FILE="$BENCH_DIR/$(date +%Y%m%d-%H%M%S)-${MODE}.txt"
{
    echo "# kimik31 benchmark  $(date)"
    echo "# host=$(hostname -s 2>/dev/null || echo '?')  rocm=$(cat /opt/rocm/.info/version 2>/dev/null || amd-smi version 2>/dev/null | grep -o 'ROCm version: [0-9.]*' || echo '?')"
    echo "# mode=$MODE model=$SERVED_MODEL_NAME port=$PORT  MODEL_ID=${MODEL_ID:-?}@${MODEL_REVISION:0:8}"
    echo "# image=${SGLANG_IMAGE:-?}"
    echo "# recipe: ${recipe:-unknown}"
    echo "# in=$BENCH_INPUT_LEN out=$BENCH_OUTPUT_LEN num_prompts=$BENCH_NUM_PROMPTS range_ratio=$BENCH_RANGE_RATIO"
    echo
} > "$OUT_FILE"
log "Saving results to $OUT_FILE"

# Prefer the lighter sglang.bench_serving shim: the new module imports aiter at
# module scope, which a pure HTTP client has no use for (K3, 9 Aug 2026).
BENCH_MODULE="${BENCH_MODULE:-}"
if [[ -z "$BENCH_MODULE" ]]; then
    BENCH_MODULE="sglang.bench_serving"
    apptainer exec "$SIF_PATH" python3 -c \
        'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec("sglang.bench_serving") else 1)' 2>/dev/null \
        || BENCH_MODULE="sglang.benchmark.serving"
fi
log "Bench module: $BENCH_MODULE"

# Same writable-state binds as the server: an empty AITER_JIT_DIR poisons any
# aiter import, and /tmp/aiter_configs is shared between users.
SGLANG_CACHE_DIR="${SGLANG_CACHE_DIR:-$MODEL_CACHE_DIR/sglang-cache-k31}"
AITER_JIT_DIR="${AITER_JIT_DIR:-$MODEL_CACHE_DIR/aiter-jit-k31}"
mkdir -p "$SGLANG_CACHE_DIR" 2>/dev/null || true
bench_binds=()
[[ -d "$AITER_JIT_DIR" ]] && bench_binds+=(--bind "$AITER_JIT_DIR")
[[ -d "$SGLANG_CACHE_DIR" ]] && bench_binds+=(--bind "$SGLANG_CACHE_DIR")
AITER_CONFIG_DIR="${AITER_CONFIG_DIR:-$MODEL_CACHE_DIR/aiter-configs/k31-${SLURM_JOB_ID:-$$}}"
if mkdir -p "$AITER_CONFIG_DIR" 2>/dev/null && [[ -w "$AITER_CONFIG_DIR" ]] \
   && apptainer exec --bind "$AITER_CONFIG_DIR":/tmp/aiter_configs "$SIF_PATH" \
        sh -c 'touch /tmp/aiter_configs/.w && rm -f /tmp/aiter_configs/.w' >/dev/null 2>&1; then
    bench_binds+=(--bind "$AITER_CONFIG_DIR":/tmp/aiter_configs)
fi

run_one() {
    local conc="$1" rate="$2"
    log "Run: concurrency=$conc${rep:+ repeat $rep/$BENCH_REPEATS} (in=$BENCH_INPUT_LEN out=$BENCH_OUTPUT_LEN n=$BENCH_NUM_PROMPTS)"
    echo "=== concurrency=$conc request-rate=$rate${rep:+ repeat=$rep} ===" >> "$OUT_FILE"
    # shellcheck disable=SC2086
    apptainer exec \
        --env "OPENAI_API_KEY=${KIMIK31_API_KEY:-}" \
        --env "AITER_JIT_DIR=$AITER_JIT_DIR" --env "SGLANG_CACHE_DIR=$SGLANG_CACHE_DIR" \
        ${bench_binds[@]+"${bench_binds[@]}"} "$SIF_PATH" \
        python3 -m "$BENCH_MODULE" \
            --backend sglang-oai \
            --base-url "http://127.0.0.1:${PORT}" \
            --model "$SERVED_MODEL_NAME" \
            --dataset-name random \
            --random-input-len "$BENCH_INPUT_LEN" \
            --random-output-len "$BENCH_OUTPUT_LEN" \
            --random-range-ratio "$BENCH_RANGE_RATIO" \
            --num-prompts "$BENCH_NUM_PROMPTS" \
            --max-concurrency "$conc" \
            --request-rate "$rate" \
            $BENCH_EXTRA_ARGS \
        2>&1 | tee -a "$OUT_FILE" \
        | grep -iE 'throughput|TTFT|TPOT|ITL|latency|Successful|concurrency|accept' || true
    echo >> "$OUT_FILE"
}

rep=""
case "$MODE" in
    latency)    run_one 1 inf ;;
    throughput) run_one "$BENCH_MAX_CONCURRENCY" inf ;;
    sweep|longcontext)
        for c in $BENCH_CONCURRENCY; do
            for rep in $(seq 1 "$BENCH_REPEATS"); do run_one "$c" inf; done
        done ;;
esac

acc="$(grep -oE ', accept len: [0-9.]+' "$LOG_FILE" 2>/dev/null | tail -n 50 | awk '{s+=$NF; n++} END {if (n) printf "%.2f", s/n}')"
[[ -n "$acc" ]] && { log "DFLASH accept length (last 50 log lines): $acc"; echo "# accept_len_tail50=$acc" >> "$OUT_FILE"; }
log "Done. Full output: $OUT_FILE"
if (( BENCH_REPEATS <= 1 )); then
    warn "Single run per point — no error bar. Set BENCH_REPEATS=3 before concluding two configs differ."
fi
log "Compare runs with:  ls -t $BENCH_DIR"
