# Validation

[Documentation home](../README.md) | [Compatibility](compatibility.md)

## CPU checks (anywhere)

```bash
python3 -m unittest discover -s tests -v
```

These assert that every profile's argv and environment match upstream's
verified MI355X cells, that HiCache is scaled below Bunya's GTT, that invalid
combinations are rejected before launch, that `agentstress.py` replays
`reasoning_content`, and that the shell syntax and doc links are valid. No GPU,
model or endpoint is involved.

## Bunya qualification (on the node)

Run these in order. Each step is cheaper than the one after it.

1. `./serve-kimik31.sh gpucheck`: the container sees 8 × gfx950.
2. `./serve-kimik31.sh check`: the registry knows `KimiK31ForConditionalGeneration`;
   the lever report shows monokernel, DFLASH, ReplaySSM and DCP.
3. `AGENTS=<n> ./serve-kimik31.sh flagcheck` for each tier you will use.
4. `./serve-kimik31.sh serve --detach`, then `status` (monokernel engaged once
   a request has decoded).
5. `./serve-kimik31.sh toolcheck`: the two-turn tool loop with reasoning replay.
6. `./agentstress.py http://127.0.0.1:30000 "$KIMIK31_API_KEY" kimi-k3.1-pro 150000 20000`:
   a growing real-text session with prefix-cached prefill. It must finish
   clean, with no `[PAD]` runs and the needles found.
7. `./bench-kimik31.sh sweep` (concurrency 1/2/4 for the default profile).

## Speculative decoding A/B

DFLASH acceptance on random bench prompts overstates real-text acceptance. The
K3 repo lost a conclusion to this. For the workload that matters:

```bash
./serve-kimik31.sh serve --detach
./agentstress.py http://127.0.0.1:30000 "$KIMIK31_API_KEY" kimi-k3.1-pro 150000 20000 | tee logs/stress-dflash.log
./serve-kimik31.sh status            # note the accept length
./serve-kimik31.sh stop
SPECULATIVE=none ./serve-kimik31.sh serve --detach
./agentstress.py http://127.0.0.1:30000 "$KIMIK31_API_KEY" kimi-k3.1-pro 150000 20000 | tee logs/stress-nospec.log
```

Compare the per-context decode tok/s summary at the end of each log. Keep DFLASH
on if it wins in the context range your sessions actually reach.
