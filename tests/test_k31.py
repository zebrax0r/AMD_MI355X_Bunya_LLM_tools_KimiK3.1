"""CPU-only checks for the Kimi K3.1 Bunya tooling.

No model, GPU, Apptainer, endpoint or Slurm job is needed: the launcher's
'plan' mode resolves the recipe in pure bash, and agentstress.py is driven
against an in-process fake server.

    python3 -m unittest discover -s tests -v
"""
import contextlib
import io
import json
import os
from pathlib import Path
import re
import runpy
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SERVE = ROOT / "serve-kimik31.sh"


def plan(**env):
    base = {"PATH": os.environ.get("PATH", os.defpath), "HOME": tempfile.gettempdir(),
            "KIMIK31_ENV": "/nonexistent"}
    base.update({k: str(v) for k, v in env.items()})
    r = subprocess.run(["bash", str(SERVE), "plan"], env=base, capture_output=True, text=True, timeout=30)
    return r


def parse_plan(out):
    env, argv, summary, hicache = {}, [], "", ""
    lines = out.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("# profile="):
            summary = line[2:]
        elif line.startswith("# hicache:"):
            hicache = line
        elif line.startswith("  ") and "=" in line and not line.strip().startswith("sglang"):
            k, v = line.strip().split("=", 1)
            env[k] = v
        elif line == "# argv:":
            argv = shlex.split(lines[i + 1])
    return summary, env, argv, hicache


def flagval(argv, flag):
    i = argv.index(flag)
    vals = []
    for a in argv[i + 1:]:
        if a.startswith("--"):
            break
        vals.append(a)
    return vals


@unittest.skipUnless(shutil.which("bash"), "bash required")
class RecipeTests(unittest.TestCase):
    """The argv must match upstream's verified MI355X Kimi-K3.1-Pro cells."""

    COMMON = ["--trust-remote-code", "--weight-loader-prefetch-checkpoints",
              "--enable-linear-replayssm-spec", "--speculative-use-block-verification"]

    def resolve(self, **env):
        r = plan(**env)
        self.assertEqual(r.returncode, 0, r.stderr)
        return parse_plan(r.stdout)

    def test_lowlatency_matches_cookbook_cell(self):
        s, env, argv, _ = self.resolve(AGENTS=4)
        self.assertIn("profile=lowlatency", s)
        for f in self.COMMON + ["--enable-amd-monokernel"]:
            self.assertIn(f, argv)
        self.assertNotIn("--dcp-size", argv)
        self.assertNotIn("--enable-hierarchical-cache", argv)
        expect = {"--tp-size": ["8"], "--mem-fraction-static": ["0.85"], "--chunked-prefill-size": ["16384"],
                  "--max-running-requests": ["4"], "--cuda-graph-max-bs-decode": ["4"],
                  "--schedule-policy": ["lpm"], "--prefill-decode-interval": ["16"],
                  "--max-mamba-cache-size": ["150"], "--mamba-radix-cache-strategy": ["extra_buffer_lazy"],
                  "--cuda-graph-backend-prefill": ["breakable"],
                  "--cuda-graph-bs-prefill": "256 512 768 1024 1536 2048 3072 4096".split(),
                  "--speculative-algorithm": ["DFLASH"], "--speculative-dflash-block-size": ["6"],
                  "--speculative-draft-attention-backend": ["triton"],
                  "--speculative-draft-kv-cache-dtype": ["fp8_e4m3"],
                  "--reasoning-parser": ["kimi_k3"], "--tool-call-parser": ["kimi_k3"]}
        for f, v in expect.items():
            self.assertEqual(flagval(argv, f), v, f)
        self.assertEqual(env["SGLANG_AMD_MONO_K31_WIDE"], "1")
        self.assertEqual(env["SGLANG_ROCM_RUNTIME_RESERVE_GB"], "6")
        for k in ("SGLANG_USE_AITER", "SGLANG_AITER_K3_OPT", "AITER_FLYDSL_FORCE", "AITER_SITUV2_A8W4",
                  "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK"):
            self.assertEqual(env[k], "1", k)
        self.assertEqual(env["HSA_USERPTR_FOR_PAGED_MEM"], "0")
        self.assertEqual(env["SGLANG_SET_CPU_AFFINITY"], "0")

    def test_single_agent_is_agentx_c1_point(self):
        _, _, argv, _ = self.resolve(AGENTS=1)
        self.assertEqual(flagval(argv, "--max-running-requests"), ["1"])
        self.assertEqual(flagval(argv, "--max-mamba-cache-size"), ["60"])
        self.assertNotIn("--prefill-decode-interval", argv)

    def test_mono_dcp8(self):
        s, env, argv, _ = self.resolve(AGENTS=8)
        self.assertIn("profile=mono-dcp8", s)
        self.assertEqual(flagval(argv, "--dcp-size"), ["8"])
        self.assertEqual(flagval(argv, "--dcp-comm-backend"), ["a2a"])
        self.assertEqual(flagval(argv, "--mamba-full-memory-ratio"), ["0.65"])
        self.assertIn("--enable-amd-monokernel", argv)
        self.assertEqual(env["SGLANG_ROCM_RUNTIME_RESERVE_GB"], "10")

    def test_throughput_has_hicache_and_no_monokernel(self):
        s, env, argv, hc = self.resolve(AGENTS=32)
        self.assertIn("profile=throughput ", s + " ")
        self.assertNotIn("--enable-amd-monokernel", argv)
        self.assertNotIn("SGLANG_AMD_MONO_K31_WIDE", env)
        self.assertIn("--enable-hierarchical-cache", argv)
        self.assertEqual(flagval(argv, "--hicache-write-policy"), ["write_back"])
        self.assertEqual(flagval(argv, "--max-running-requests"), ["32"])
        self.assertEqual(env["RUSTUP_TOOLCHAIN"], "stable")
        self.assertIn("SGLANG_HICACHE_MAMBA_HOST_RATIO", env)
        self.assertTrue(hc)

    def test_throughput_wide_shape(self):
        s, env, argv, _ = self.resolve(AGENTS=48)
        self.assertIn("profile=throughput-wide", s)
        self.assertEqual(flagval(argv, "--chunked-prefill-size"), ["32768"])
        self.assertEqual(flagval(argv, "--prefill-decode-interval"), ["8"])
        self.assertEqual(flagval(argv, "--speculative-dflash-block-size"), ["5"])
        for k in ("SGLANG_ENABLE_ROCM_K3_ROW_SHARDED_ROUTED_FRONT", "SGLANG_ENABLE_ROCM_K3_ROW_SHARDED_STREAM",
                  "SGLANG_OPT_DFLASH_RING_LOCS_FUSED"):
            self.assertEqual(env[k], "1")
        _, _, argv56, _ = self.resolve(AGENTS=56)
        self.assertEqual(flagval(argv56, "--max-running-requests"), ["64"])

    def test_hicache_scaled_to_bunya_gtt_not_cookbook(self):
        # Upstream's 2.5 / 10 pins ~1.4 TB: over Bunya's 1133 GiB GTT. Must scale down.
        _, env, argv, hc = self.resolve(AGENTS=16)
        ratio = float(flagval(argv, "--hicache-ratio")[0])
        self.assertLess(ratio, 2.5)
        self.assertGreater(ratio, 1.0)
        self.assertLess(int(env["SGLANG_HICACHE_MAMBA_HOST_RATIO"]), 10)
        m = re.search(r"~(\d+) GiB pinned of (\d+) GiB", hc)
        self.assertTrue(m, hc)
        self.assertLess(int(m.group(1)), int(m.group(2)) * 0.9)

    def test_explicit_overrides_win(self):
        _, env, argv, _ = self.resolve(AGENTS=4, MAX_RUNNING_REQUESTS=2, DFLASH_BLOCK_SIZE=4,
                                       CONTEXT_LEN=262144, HICACHE_RATIO="2.0", PROFILE="throughput")
        self.assertEqual(flagval(argv, "--max-running-requests"), ["2"])
        self.assertEqual(flagval(argv, "--speculative-dflash-block-size"), ["4"])
        self.assertEqual(flagval(argv, "--context-length"), ["262144"])
        self.assertEqual(flagval(argv, "--hicache-ratio"), ["2.0"])

    def test_spec_none_drops_all_draft_flags(self):
        _, env, argv, _ = self.resolve(SPECULATIVE="none")
        self.assertFalse([a for a in argv if a.startswith("--speculative")])
        self.assertNotIn("--enable-linear-replayssm-spec", argv)
        # SGLang's monokernel refuses non-speculative setups at startup.
        self.assertNotIn("--enable-amd-monokernel", argv)
        self.assertNotIn("SGLANG_AMD_MONO_K31_WIDE", env)

    def test_monokernel_needs_block_6_or_8(self):
        _, _, argv, _ = self.resolve(DFLASH_BLOCK_SIZE=5)
        self.assertNotIn("--enable-amd-monokernel", argv)
        _, _, argv, _ = self.resolve(DFLASH_BLOCK_SIZE=8)
        self.assertIn("--enable-amd-monokernel", argv)
        for env in (dict(MONOKERNEL=1, SPECULATIVE="none"), dict(MONOKERNEL=1, DFLASH_BLOCK_SIZE=5)):
            r = plan(**env)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("DFLASH_BLOCK_SIZE 6 or 8", r.stderr)

    def test_no_k3_only_flags_leak_into_k31(self):
        # K3.1 pins DSA to triton on ROCm and forces fp8 KV; the K3 launcher's
        # --attention-backend/--dtype defaults must not appear.
        _, env, argv, _ = self.resolve()
        for f in ("--attention-backend", "--dtype", "--kv-cache-dtype", "--speculative-dspark-block-size"):
            self.assertNotIn(f, argv)
        self.assertNotIn("SGLANG_MLA_DECODE_TUNE", env)
        self.assertIn("<redacted>", argv)

    def test_rejections(self):
        for env, msg in ((dict(AGENTS=57), "56"), (dict(KV_CACHE_DTYPE="bfloat16"), "fp8"),
                         (dict(SPECULATIVE="dspark"), "DFLASH"), (dict(AGENTS=0), "positive"),
                         (dict(MONOKERNEL=1, HICACHE="on"), "KV"),
                         (dict(AITER_GPU_ARCHS="gfx942;gfx950"), "exactly one")):
            with self.subTest(env=env):
                r = plan(**env)
                self.assertNotEqual(r.returncode, 0)
                self.assertIn(msg, r.stderr)

    def test_extra_env_and_args(self):
        _, env, argv, _ = self.resolve(EXTRA_ENV="FOO_BAR=1 SGLANG_X=y", EXTRA_ENGINE_ARGS="--foo 3")
        self.assertEqual(env["FOO_BAR"], "1")
        self.assertEqual(argv[-2:], ["--foo", "3"])
        self.assertNotEqual(plan(EXTRA_ENV="bad").returncode, 0)

    def test_presharded_loader_config(self):
        _, _, argv, _ = self.resolve(LOAD_FORMAT="presharded", PRESHARDED_PATH="/s/p", WEIGHT_LOAD_THREADS=8)
        cfg = json.loads(flagval(argv, "--model-loader-extra-config")[0])
        self.assertEqual(cfg, {"enable_multithread_load": True, "num_threads": 8,
                               "presharded_path": "/s/p", "draft_presharded_path": "/s/p/dflash"})


class EnvTemplateTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("bash"), "bash required")
    def test_example_resolves_to_launcher_defaults(self):
        # Sourcing the example with nothing set must produce the same plan as no config.
        with tempfile.TemporaryDirectory() as d:
            a = plan(KIMIK31_ENV=str(ROOT / "kimik31-env.example"))
            b = plan()
            self.assertEqual(a.returncode, 0, a.stderr)
            self.assertEqual(a.stdout, b.stdout)

    def test_example_has_no_secrets(self):
        text = (ROOT / "kimik31-env.example").read_text()
        self.assertNotRegex(text, r"hf_[A-Za-z0-9]{20,}")
        self.assertIn('HF_TOKEN="${HF_TOKEN:-}"', text)

    def test_gitignore_keeps_local_config_out(self):
        self.assertIn("kimik31.env", (ROOT / ".gitignore").read_text().split())


def run_stress(outcome="clean", args=("60000", "20000"), env=None):
    requests = []

    def fake_urlopen(request, timeout):
        body = json.loads(request.data)
        requests.append(body)
        if outcome == "transport":
            raise OSError("synthetic connection failure")
        text = body["messages"][-1]["content"]
        code = re.findall(r"ticket for this section is ([0-9]{6}-[a-z]+)\.", text)[-1]
        reply = {"pad": "[PAD]", "missing": "No ticket found."}.get(outcome, code + "; it configures the image.")
        events = [{"choices": [{"delta": {"reasoning_content": "Looking for the ticket."}}]},
                  {"choices": [{"delta": {"content": reply}}]},
                  {"choices": [], "usage": {"completion_tokens": 24, "prompt_tokens": 100,
                                            "prompt_tokens_details": {"cached_tokens": 60}}}]
        data = b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events)
        return io.BytesIO(data + b"data: [DONE]\n\n")

    argv = [str(ROOT / "agentstress.py"), "http://unused.invalid", "k", "kimi-k3.1-pro", *args]
    out = io.StringIO()
    with patch.object(sys, "argv", argv), patch("urllib.request.urlopen", fake_urlopen), \
            patch.dict(os.environ, env or {}), contextlib.redirect_stdout(out):
        try:
            runpy.run_path(argv[0], run_name="__main__")
        except SystemExit as exc:
            return exc.code, requests, out.getvalue()
    raise AssertionError("agentstress did not exit")


class StressTests(unittest.TestCase):
    def test_clean_run_replays_reasoning_content(self):
        code, reqs, out = run_stress()
        self.assertEqual(code, 0, out)
        self.assertEqual(len(reqs), 3)
        last = reqs[-1]["messages"]
        assistants = [m for m in last if m["role"] == "assistant"]
        self.assertEqual(len(assistants), 2)
        self.assertTrue(all(m.get("reasoning_content") for m in assistants))
        self.assertEqual(reqs[0]["reasoning_effort"], "low")
        self.assertEqual((reqs[0]["temperature"], reqs[0]["top_p"]), (1.0, 0.95))
        self.assertIn("cached 60/100", out)

    def test_effort_env(self):
        _, reqs, _ = run_stress(env={"AGENTSTRESS_EFFORT": "max"})
        self.assertEqual(reqs[0]["reasoning_effort"], "max")

    def test_outcomes(self):
        for outcome, status in (("pad", 1), ("transport", 2), ("missing", 0)):
            with self.subTest(outcome=outcome):
                self.assertEqual(run_stress(outcome)[0], status)


class SyntaxAndDocsTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("bash"), "bash required")
    def test_shell_syntax(self):
        for p in list(ROOT.glob("*.sh")) + list(ROOT.glob("*.sbatch")) + [ROOT / "kimik31-env.example"]:
            with self.subTest(path=p.name):
                subprocess.run(["bash", "-n", str(p)], check=True, capture_output=True, timeout=10)

    def test_executables(self):
        for n in ("serve-kimik31.sh", "bench-kimik31.sh", "serve-kimik31.sbatch", "agentstress.py",
                  "opencode-setup.sh", "kimicode-setup.sh"):
            self.assertTrue(os.access(ROOT / n, os.X_OK), n)

    def test_doc_links_resolve(self):
        def slugs(path):
            out, seen, fenced = set(), {}, False
            for line in path.read_text().splitlines():
                if line.startswith("```"):
                    fenced = not fenced
                    continue
                m = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
                if fenced or not m:
                    continue
                s = re.sub(r"[^\w\- ]", "", re.sub(r"<[^>]*>", "", m.group(1).lower())).replace(" ", "-")
                n = seen.get(s, 0); seen[s] = n + 1
                out.add(s + (f"-{n}" if n else ""))
            return out
        for path in [ROOT / "README.md"] + sorted((ROOT / "docs").rglob("*.md")):
            for target in re.findall(r"(?<!!)\[[^\]]+\]\(([^\s)]+)\)", path.read_text()):
                parts = urlsplit(target)
                if parts.scheme or parts.netloc:
                    continue
                with self.subTest(source=path.name, target=target):
                    linked = (path.parent / unquote(parts.path)).resolve() if parts.path else path
                    self.assertTrue(linked.exists(), "missing link target")
                    if parts.fragment and linked.suffix == ".md":
                        self.assertIn(parts.fragment, slugs(linked))


if __name__ == "__main__":
    unittest.main()
