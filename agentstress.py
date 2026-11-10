#!/usr/bin/env python3
"""Agentic-shape stress test: one conversation that grows like a kimicode
session, with real text, streaming, temperature 1.0 / top_p 0.95 (K3.1's
recommended sampling), and no cache-friendly randomness. Each turn appends
~TURN_TOKENS of new material containing a fresh needle, asks for it back, and
checks the reply for corruption.

Usage: ./agentstress.py <base_url> <api_key> <model> [max_ctx_tokens] [turn_tokens]
Env:   AGENTSTRESS_EFFORT=low|high|max   reasoning_effort per request (default low)
       AGENTSTRESS_MAX_TOKENS=N          per-turn completion cap (default 4096)
Exit 0 = all turns clean, 1 = corruption seen, 2 = request failed (server down?).

Why this exists next to bench-kimik31.sh: the bench sends random prompts with no
shared prefix, so it never exercises prefix-cached prefill (where the K3 repo's
13 Oct 2026 aiter fault lived), and random tokens make speculative decoding look
far better than it is on real text.

K3.1 specifics: the model always thinks, and requires the COMPLETE assistant
message — reasoning_content included — to be passed back each turn. This test
does that, so a server/parser that mishandles replayed reasoning shows up here.
"""
import json, os, random, sys, time, urllib.request

base, key, model = sys.argv[1], sys.argv[2], sys.argv[3]
max_ctx = int(sys.argv[4]) if len(sys.argv) > 4 else 150000
turn_tok = int(sys.argv[5]) if len(sys.argv) > 5 else 20000
effort = os.environ.get("AGENTSTRESS_EFFORT", "low")
max_tokens = int(os.environ.get("AGENTSTRESS_MAX_TOKENS", "4096"))

here = os.path.dirname(os.path.abspath(__file__))
repo = os.path.join(here, "tests", "fixtures", "corpus")
corpus = []
for f in sorted(os.listdir(repo)):
    with open(os.path.join(repo, f), errors="replace") as fh:
        corpus += [p for p in fh.read().split("\n\n") if p.strip()]
rng = random.Random(11)


def chunk(n_tokens, code):
    # ~3.5 chars/token for this mix of prose and shell
    out, size = [], 0
    while size < n_tokens * 3.5:
        p = rng.choice(corpus)
        out.append(p)
        size += len(p)
    out.insert(len(out) // 3, f"NOTE: the deployment ticket for this section is {code}.")
    return "\n\n".join(out)


def stream(messages):
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
               "temperature": 1.0, "top_p": 0.95, "stream": True,
               "stream_options": {"include_usage": True}}
    if effort:
        payload["reasoning_effort"] = effort
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    content, reasoning, t0, ttft, ntok, prompt_tok, cached = [], [], time.time(), None, 0, 0, 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            u = d.get("usage")
            if u:
                ntok = u.get("completion_tokens") or ntok
                prompt_tok = u.get("prompt_tokens") or prompt_tok
                cached = ((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or cached
            if not d.get("choices"):
                continue
            delta = d["choices"][0].get("delta", {})
            rc = delta.get("reasoning_content") or delta.get("reasoning") or ""
            if ttft is None and (delta.get("content") or rc):
                ttft = time.time() - t0
            content.append(delta.get("content") or "")
            reasoning.append(rc)
    return "".join(content), "".join(reasoning), ttft or 0, time.time() - t0, ntok, prompt_tok, cached


def corrupt(text):
    if "[PAD]" in text or "!!!!!!!!!!" in text:
        return "pad/bang run"
    if len(text) > 200 and len(set(text[-200:])) < 6:
        return "degenerate repetition"
    return None


messages = [{"role": "system", "content": "You are a coding assistant reviewing a deployment repository."}]
ctx, turn, ok, rates = 0, 0, True, []
print(f"agentstress: model={model} effort={effort or 'server default'} max_ctx~{max_ctx} turn~{turn_tok}", flush=True)
while ctx < max_ctx:
    turn += 1
    code = f"{rng.randint(100000, 999999)}-{rng.choice(['amber', 'cobalt', 'violet', 'saffron'])}"
    messages.append({"role": "user", "content": chunk(turn_tok, code) +
                     "\n\nBriefly: what is the deployment ticket for this section, and what is one "
                     "thing this section configures? Answer in under 80 words."})
    ctx += turn_tok
    try:
        content, reasoning, ttft, total, ntok, ptok, cached = stream(messages)
        rates.append((ctx, ntok / max(total - ttft, 1e-3)))
    except Exception as e:
        print(f"turn {turn:2d} ~{ctx//1000:3d}k  REQUEST FAILED: {e}")
        sys.exit(2)
    bad = corrupt(reasoning) or corrupt(content)
    found = code in content
    status = "CORRUPT: " + bad if bad else ("ok" if found else "needle missed")
    ok &= not bad
    hit = f" cached {cached}/{ptok}" if ptok else ""
    print(f"turn {turn:2d} ~{ctx//1000:3d}k  ttft {ttft:5.1f}s total {total:5.1f}s  {ntok:4d} tok "
          f"{rates[-1][1]:5.1f} tok/s{hit}  {status}  reply={content.strip()[:40]!r}", flush=True)
    if bad:
        print("  reasoning tail:", repr(reasoning[-160:]))
        print("  content tail:  ", repr(content[-160:]))
        sys.exit(1)
    asst = {"role": "assistant", "content": content}
    if reasoning:
        asst["reasoning_content"] = reasoning
    messages.append(asst)
for lo, hi in ((0, 60000), (60000, 120000), (120000, 10**9)):
    r = [v for c, v in rates if lo < c <= hi]
    if r:
        print(f"decode tok/s, context {lo//1000}-{min(hi, ctx)//1000}k: mean {sum(r)/len(r):.1f}")
sys.exit(0 if ok else 1)
