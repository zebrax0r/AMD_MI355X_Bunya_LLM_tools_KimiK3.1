"""K3.1 serving shims, loaded into the SGLang container by serve-kimik31.sh.

serve-kimik31.sh binds this directory into the container and puts it first on
PYTHONPATH, so Python imports this file at startup in every process. It only
installs an import hook; nothing heavy is imported here.

reasoning_effort "medium" (K31_MEDIUM_EFFORT)
    The pinned image (sglang de0f4918) treats "medium" as a native K3.1 tier
    and passes it through to the chat template. The pinned checkpoint's
    tokenizer (encoding_k3.py @ d9e99b07) only accepts low/high/max and
    asserts on anything else. Every client that defaults to medium (most
    OpenAI-style agent clients) then gets HTTP 500 on every request. When
    K31_MEDIUM_EFFORT is set (the launcher sets it to "high" unless the
    checkpoint accepts "medium"), this remaps medium to that tier once
    sglang's serving_chat module loads. That is what SGLang already does
    for Kimi K3.
"""

import importlib.abc
import importlib.util
import os
import sys

_SERVING_CHAT = "sglang.srt.entrypoints.openai.serving_chat"
_MEDIUM = os.environ.get("K31_MEDIUM_EFFORT", "").strip()


def _patch_serving_chat(module):
    tiers = getattr(module, "_KIMI_K31_EFFORT_TIERS", None)
    if not isinstance(tiers, dict):
        print("[k31 shim] _KIMI_K31_EFFORT_TIERS not found in serving_chat; "
              "medium-effort remap NOT applied", file=sys.stderr, flush=True)
        return
    if tiers.get("medium") == _MEDIUM:
        return
    old = tiers.get("medium")
    tiers["medium"] = _MEDIUM  # mutate in place: _kimi_k3_effort_tiers() returns this dict
    print(f"[k31 shim] reasoning_effort 'medium' -> thinking_effort '{_MEDIUM}' "
          f"(was {old!r}; checkpoint accepts low/high/max)", file=sys.stderr, flush=True)


class _PatchingLoader(importlib.abc.Loader):
    def __init__(self, inner):
        self._inner = inner

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        _patch_serving_chat(module)

    def __getattr__(self, name):  # get_source/get_code etc. for tracebacks
        return getattr(self._inner, name)


class _ServingChatFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name != _SERVING_CHAT:
            return None
        sys.meta_path.remove(self)  # one-shot; also avoids recursing into ourselves
        spec = importlib.util.find_spec(name)
        if spec is not None and spec.loader is not None:
            spec.loader = _PatchingLoader(spec.loader)
        return spec


if _MEDIUM and _MEDIUM.lower() != "off":
    if _MEDIUM not in ("low", "high", "max"):
        print(f"[k31 shim] K31_MEDIUM_EFFORT={_MEDIUM!r} is not low/high/max; ignoring",
              file=sys.stderr, flush=True)
    else:
        sys.meta_path.insert(0, _ServingChatFinder())


# Chain to the next sitecustomize on sys.path (the image ships Debian's
# apport hook), so shadowing it changes nothing else.
def _chain():
    here = os.path.dirname(os.path.abspath(__file__))
    for d in sys.path:
        if os.path.abspath(d or os.curdir) == here:
            continue
        f = os.path.join(d, "sitecustomize.py")
        if os.path.isfile(f):
            with open(f) as fh:
                code = compile(fh.read(), f, "exec")
            exec(code, {"__name__": "sitecustomize", "__file__": f})
            return


try:
    _chain()
except Exception as exc:  # never let a chained hook stop the server starting
    print(f"[k31 shim] chained sitecustomize failed: {exc}", file=sys.stderr, flush=True)
