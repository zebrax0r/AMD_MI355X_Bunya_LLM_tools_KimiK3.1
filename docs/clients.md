# Client setup

[Documentation home](../README.md) | [Runbook](runbook.md)

The endpoint is OpenAI-compatible at `http://<node>:30000/v1` with model
`kimi-k3.1-pro` and `Authorization: Bearer $KIMIK31_API_KEY`. The key is in
`$MODEL_CACHE_DIR/kimik31-api-key`.

## Two K3.1 rules every client must follow

1. **Replay the full assistant message.** On the next turn, send back
   `reasoning_content` and `tool_calls` along with `content`. Moonshot's model
   card requires this. A client that strips reasoning loses context mid-task.
   `./serve-kimik31.sh toolcheck` exercises exactly this.
2. **Thinking depth** is the top-level request field
   `"reasoning_effort": "low" | "high" | "max"` (default `max`). The model always
   thinks; `low` is the fast setting for routine agent turns. The pinned image
   maps OpenAI's other tiers onto these (`minimal`→low, `xhigh`→max), and
   `"none"` or an absent field means the model default (`max`). Recommended
   sampling is `temperature=1.0`, `top_p=0.95`.

## opencode

```bash
./opencode-setup.sh --host bun159 --port 30000          # on the cluster
./opencode-setup.sh --host localhost --api-key "$KEY"   # laptop, with a tunnel
export KIMIK31_API_KEY="$(cat "$MODEL_CACHE_DIR/kimik31-api-key")"
```

This adds provider `kimik31-bunya` (model "Kimi K3.1 Pro") to
`~/.config/opencode/opencode.json`, merged with `jq` when available. Pick it
with `/models`.

## Kimi Code

```bash
./kimicode-setup.sh --host bun159 --port 30000
kimi -m kimik31-bunya/kimi-k3.1-pro
```

This writes a managed block into `~/.kimi-code/config.toml` (mode 600, with the
key inline). Use `--no-key` to get `KIMI_MODEL_*` environment variables instead.
This targets the Node.js Kimi Code CLI, not the legacy Python `kimi-cli`.

## Tunnel from a laptop

```bash
ssh -N -L 30000:bun159:30000 "$USER"@bunya1.rcc.uq.edu.au
```

The advertised context limit follows `CONTEXT_LEN` (empty = 1,048,576).
