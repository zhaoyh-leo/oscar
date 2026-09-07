"""Prompt-invariance harness — proves LLM messages are byte-identical across code refactors.

Every LLM call in OSCAR flows through ``oscar.utils.cache.llm_cache_get``.
This harness wraps that function, records the exact cache key
(sha256 of messages+model+temperature+salt) of every call, and **fails
immediately (exit 2) on the first cache MISS** — a miss means the prompt
bytes changed and the run would silently re-sample from the API.

Usage (args are forwarded to main.py unchanged):

    python scripts/verify_prompts.py <repo_url> [--paper ARXIV] [--no-cleanup]

Output: key list written to the path in env var OSCAR_KEYS_FILE
(default: verify_keys.txt). Compare two runs:

    diff pre_keys.txt post_keys.txt     # must be empty
    diff -r output/ output_ref/         # must be empty

Requires a warm LLM cache (run the audit once before refactoring).
"""

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import oscar.utils.cache as cache_mod  # noqa: E402
from oscar.utils.cache import _llm_key  # noqa: E402

_orig_get = cache_mod.llm_cache_get
_keys: list[str] = []
_misses: list[str] = []


def _spy_get(messages, model, temperature=0.0, salt=""):
    import time as _time
    key = _llm_key(messages, model, temperature, salt)
    _keys.append(key)
    got = _orig_get(messages, model, temperature, salt)
    if got is None:
        _misses.append(key)
        user = "".join(m.get("content", "") for m in messages if m.get("role") == "user")
        m = __import__("re").search(r"Claim ID: (\S+)", user)
        cid = m.group(1) if m else "?"
        line = (f"[verify_prompts] MISS #{len(_keys)} key={key[:16]} claim={cid} "
                f"model={model} msgsha16={__import__('hashlib').sha256(str(messages).encode()).hexdigest()[:8]}")
        # 现场复现:文件是否存在?读到了什么?为什么 miss?
        p = cache_mod._LLM_DIR / f"{key}.json"
        line += f" dir={cache_mod._LLM_DIR}"
        try:
            line += f" exists={p.exists()}"
            if p.exists():
                raw = p.read_text(encoding="utf-8")
                line += f" bytes={len(raw)}"
                try:
                    data = __import__("json").loads(raw)
                    age = _time.time() - data["cached_at"]
                    line += (f" cached_at={data['cached_at']:.0f} age_s={age:.0f} "
                             f"ttl={cache_mod._TTL_DEFAULT} expired={age > cache_mod._TTL_DEFAULT}")
                    line += f" has_content={'content' in data.get('response', {})}"
                except Exception as exc:
                    line += f" json_error={type(exc).__name__}: {exc}"
            else:
                line += f" cached_dir_files={len(list(cache_mod._LLM_DIR.glob(f'{key[:4]}*.json')))}"
        except Exception as exc:
            line += f" probe_error={type(exc).__name__}: {exc}"
        print(line, file=sys.stderr)
    return got


cache_mod.llm_cache_get = _spy_get

import main as main_mod  # noqa: E402

try:
    main_mod.main()
finally:
    out = Path(os.environ.get("OSCAR_KEYS_FILE", "verify_keys.txt"))
    out.write_text("\n".join(_keys) + "\n", encoding="utf-8")
    print(f"[verify_prompts] recorded {len(_keys)} LLM calls ({len(_misses)} misses) -> {out}",
          file=sys.stderr)
    if _misses:
        sys.exit(2)
