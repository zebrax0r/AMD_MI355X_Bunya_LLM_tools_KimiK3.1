# Operating runbook

[Documentation home](../README.md) | [Configuration](configuration.md) | [Troubleshooting](troubleshooting.md)

## 1. Checkout and configuration (login node)

```bash
cd "/scratch/user/$USER/builds"
git clone <this repo> AMD_MI355X_Bunya_LLM_tools_KimiK3.1 && cd AMD_MI355X_Bunya_LLM_tools_KimiK3.1
cp -n kimik31-env.example kimik31.env && chmod 600 kimik31.env
${EDITOR:-vi} kimik31.env        # MODEL_CACHE_DIR, HF_TOKEN_FILE, AGENTS
```

Keep the token in a mode-600 file (`HF_TOKEN_FILE`) rather than in
`kimik31.env`. An existing K3 `MODEL_CACHE_DIR` can be reused, because every
K3.1 artefact has its own name (`kimik31-*`, `*-k31`).

## 2. Allocate a node

```bash
salloc --partition=gpu_rocm --account=a_rcc --qos=sdf \
       --nodes=1 --gres=gpu:mi355x:8 --ntasks-per-node=1 \
       --cpus-per-task=96 --mem=1800G --time=24:00:00
srun --jobid "$SLURM_JOB_ID" --pty /bin/bash -l    # if salloc left you on the login node
```

These are the K3 repo's site values. Confirm yours with RCC. Keep `--mem` at
1800G or more if you will use `AGENTS>8` (HiCache pins host RAM).

## 3. Image, compatibility, weights (compute node)

```bash
./serve-kimik31.sh pull         # ~14 GB compressed; 15-30 min to squash on GPFS
./serve-kimik31.sh gpucheck     # expect: rocm or devices OK, gfx950
./serve-kimik31.sh check        # expect: OK KimiK31ForConditionalGeneration + levers
./serve-kimik31.sh flagcheck    # expect: image accepts every flag of the recipe
./serve-kimik31.sh download     # ~1.56 TB; ~80 min at ~340 MB/s measured on bun159
```

`download` resumes if interrupted. Fetching another revision transfers only the
files that changed.

## 4. Launch

```bash
./serve-kimik31.sh serve --detach              # AGENTS from kimik31.env (default 4)
AGENTS=16 ./serve-kimik31.sh serve --detach     # or pick a tier for this run
tail -f "$MODEL_CACHE_DIR/kimik31-server.log"
```

Batch alternative: `mkdir -p logs && sbatch serve-kimik31.sbatch`, submitted
from the repository root.

A cold start reads 1.56 TB, then captures the prefill and decode graphs. The
history of time-to-ready is kept in `$MODEL_CACHE_DIR/kimik31-loadtimes.log`
(`./serve-kimik31.sh loadstat`).

## 5. Verify

```bash
export KIMIK31_API_KEY="$(cat "$MODEL_CACHE_DIR/kimik31-api-key")"
./serve-kimik31.sh status        # health, recipe, monokernel engaged?, DFLASH accept length
./serve-kimik31.sh toolcheck     # 2-turn tool loop with reasoning_content replay
./agentstress.py http://127.0.0.1:30000 "$KIMIK31_API_KEY" kimi-k3.1-pro 150000 20000
```

Read an actual reply. A healthy `/health` does not show that the output is
coherent.

## 6. Clients and stop

Configure clients with [clients](clients.md). Stop with `scancel <jobid>`, or run
`./serve-kimik31.sh stop` on the serving node.
