# Troubleshooting

[Documentation home](../README.md) | [Configuration](configuration.md)

Start with `./serve-kimik31.sh status`, then the last 60 lines of
`$MODEL_CACHE_DIR/kimik31-server.log`. Logs record the recipe and the redacted
argv at the top.

| Symptom | Cause | Fix |
|---|---|---|
| `apptainer not found` | Running on a login node | Run inside the allocation (`srun --jobid … --pty bash -l`) |
| `flagcheck`: image does not accept `--…` | Day-0 image renamed or dropped a flag | Use the pinned digest, or override the recipe knob / `EXTRA_ENGINE_ARGS` |
| `hipHostMalloc` failure / GTT errors at startup | HiCache host pools exceed GTT | Lower `HICACHE_GTT_BUDGET` (e.g. 0.7) or `HICACHE=off`. Check `cat /sys/class/drm/card0/device/mem_info_gtt_total` |
| `Not enough host memory available … only have -4.29 GB free` (HiCache profiles) | SGLang caps HiCache at the TTM `pages_limit`, which reads **0** (driver default) on Bunya, so the budget goes negative | The launcher binds the real GTT limit over it (`TTM_SHIM=auto`, the default). Check for the `TTM shim:` log line |
| `--enable-amd-monokernel cannot run this setup` | The monokernel needs DFLASH/DSpark, ReplaySSM and 6 or 8 draft tokens | The launcher turns it off automatically for `SPECULATIVE=none` or other block sizes. Do not force `MONOKERNEL=1` |
| OOM during graph capture | `MEM_FRACTION` too high for this recipe | Keep 0.85 (upstream saw 0.87+ fail). Then try `CONTEXT_LEN=262144` |
| SGLang stops at startup listing monokernel reasons | Setup outside what `--enable-amd-monokernel` supports (e.g. with HiCache) | `MONOKERNEL=0`, or use the profile's defaults |
| `Kimi-K3.1 requires --kv-cache-dtype fp8_e4m3` | A non-FP8 KV dtype was set | Unset `KV_CACHE_DTYPE` |
| `PermissionError … /tmp/aiter_configs/*.lock` | Another user owns `/tmp/aiter_configs` on this node | The launcher binds over it. If the bind fails, ask RCC for `mkdir -m 1777 /tmp/aiter_configs`, or move nodes |
| `Read-only file system: …/aiter/jit/…` | aiter JIT dir not bound writable | Check the `Preflight OK` line. Set `AITER_JIT_TARGET` if detection failed |
| `Only N of 96 weight shards` | Download incomplete or still running | `./serve-kimik31.sh download` (resumes) |
| `check`: registry EMPTY | Every model module failed to import (probe environment) | Look for `Ignore import error` lines. See the K3 repo's AITER_JIT_DIR / aiter_configs notes |
| No GPUs in container | Driver/ROCm mismatch | `./serve-kimik31.sh gpucheck`. The image is ROCm 10; nodes on ROCm 7.x will not work |
| toolcheck: empty `reasoning_content` | Wrong reasoning parser | `REASONING_PARSER=kimi_k3` |
| Agent loses track after tool calls | Client drops `reasoning_content` on replay | See [clients](clients.md#two-k31-rules-every-client-must-follow) |
| Decode slower than expected on long sessions | DFLASH acceptance low on real text | Compare `status` accept length, then A/B `SPECULATIVE=none` ([validation](validation.md#speculative-decoding-ab)) |
| Single-threaded load warning | `enable_multithread_load` missing | Keep `WEIGHT_LOAD_THREADS` empty or set; see `loadstat` |
