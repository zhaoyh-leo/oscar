# OSCAR — Open-Source Completeness Audit & Review

> 🎬 OSCAR
>
> Open-Source Completeness Audit & Review
>
> *"Every claim deserves evidence."*

> Given a GitHub repository (and optionally its paper), OSCAR checks — at the
> level of actual code — whether the repository really delivers everything the
> paper and README claim: core methods, training/inference code, datasets,
> checkpoints, benchmarks, demos, APIs and licenses.

[中文版说明 (Chinese)](README.zh-CN.md)

---

## Why

Research code on GitHub is frequently *partial*: a paper promises a method, a
dataset, a benchmark — the repository ships a demo and a license, while the
core implementation lives somewhere else or nowhere at all.

Auditing that by hand means reading claims, searching a cloned tree, and
deciding "is this really implemented here?" — claim by claim, file by file.
OSCAR automates that audit:

- extracts the claims a paper/README makes about its own contribution;
- searches the actual repository code with **local hybrid retrieval**
  (CodeBERT embeddings + FAISS, BM25, and keyword/AST exact match);
- has an LLM judge each claim **against the retrieved code chunks**, with
  verdicts anchored to concrete files and line numbers — not file-name guessing;
- investigates still-missing pieces in GitHub issues/PRs
  (explicit `PLANNED` / `RESTRICTED` statements are authoritative and cheap to
  respect);
- writes a Markdown report + machine-readable JSON.

## What OSCAR does not do

- **No code-quality review** — performance bottlenecks, style, or security
  are not audited; completeness is the only question.
- **No runnability check** — nothing is installed, no training/eval is ever
  run, no metric is reproduced.
- **No code generation** — OSCAR writes audit reports, never code: no
  completion, no fixes, nothing synthesized from scratch.

## Features

- **Claim-level verdicts**: each finding ends as `VERIFIED` / `INCOMPLETE` /
  `MISSING` / `UNCERTAIN` (plus `RESTRICTED` / `PLANNED` when issue evidence says
  so), with an evidence list of anchored code locations and LLM-written
  explanations of *what the shown code actually does*.
- **6 audit dimensions in parallel** (LangGraph fan-out): core methods,
  training, inference, API surface, resources (datasets/checkpoints/benchmarks),
  and license.
- **No LLM needed for retrieval**: search is entirely local and deterministic;
  the LLM only judges code that was actually retrieved.
- **Grounding over naming**: the same component often ships under a different
  name. Retrieval uses the paper's functional sentence, and grounding rules
  forbid declaring `MISSING` merely because no symbol matches.
- **Deterministic, cached, cheap to re-run**: LLM responses are cached on the
  exact message hash; an unchanged configuration re-runs a repository audit
  with byte-identical results and **zero API calls**.
- **Provider-agnostic**: DeepSeek (default), OpenAI, Anthropic, or any
  OpenAI-compatible endpoint — key supplied only via environment/.env.

## Pipeline

```
START → Repository Loader → Paper Resolver → Paper Analyzer →
        Repository Analyzer → Planner ─┬─ Core Methods ─┐
                                        ├─ Training ─────┤
                                        ├─ Inference ────┤
                                        ├─ API ──────────┤→ Issue Investigator →
                                        ├─ Resources ────┤
                                        └─ License ──────┘
             → Evidence Aggregator → Evidence Grounder → Report Generator → END
```

1. **Repository Loader** — clones the repo (cached under `.oscar_cache/repos/`)
   and builds the file manifest.
2. **Paper Resolver / Analyzer** — downloads the paper (PDF/arXiv) and extracts
   claims: conservative rule patterns plus an LLM pass that captures the
   *function* of each contribution in the paper's own wording.
3. **Repository Analyzer** — fingerprints files, chunks code (class/function
   bodies with line ranges), and builds the local hybrid index with CodeBERT.
4. **Planner → 6 audit nodes** — one per dimension; each claim is mapped onto
   repository code via hybrid retrieval (keyword → BM25 → vector fused by RRF,
   with a rare-token sweep and mapper-candidate fallback for non-eponymous
   implementations).
5. **Issue Investigator** — for `MISSING` / `INCOMPLETE` / `UNCERTAIN`
   findings, searches the repo's GitHub issues/PRs; explicit author statements
   (`code will be released later`, `code cannot be shared`) become
   `PLANNED` / `RESTRICTED`.
6. **Evidence Grounder** — the core step: for every code-verifiable finding the
   assembled candidate chunks (≤ a configured budget) are sent to the LLM, which
   returns one verdict + per-chunk "what this code does" notes; locations are
   anchored on the chunk records, and un-anchored verdicts degrade to
   `UNCERTAIN` instead of trusting the model's self-reported coordinates.
7. **Report Generator** — writes `audit_report.md`, `audit_result.json`, and
   `repository_manifest.json` under `output/<project>/`.

## Verdicts and categories

| Verdict | Meaning |
|---|---|
| `VERIFIED` | The repository code implements the claim. |
| `INCOMPLETE` | Only part of the claimed functionality exists. |
| `MISSING` | No code implements it (also set by rule when nothing retrievable exists). |
| `UNCERTAIN` | Retrieval/LLM could not anchor the claim — never a silent guess. |
| `RESTRICTED` / `PLANNED` | Authors state the code can't be shared / will come later. |

Claim categories: `core_method`, `training`, `inference`, `evaluation`,
`demo`, `benchmark`, `api_interface`, `dataset`, `checkpoint`, `license`,
`release`. Delivery promises (`release` — "code/weights/datasets will be
released") are verified against the repository's actual artifacts under the
Resources dimension and listed informational: they never affect the score.

## Requirements

- Python **3.9+** (developed and verified on 3.9.12; packaging metadata targets
  3.11+)
- `git` on PATH
- One LLM API key (default provider: DeepSeek)
- ~2 GB disk for the CodeBERT model, plus whatever the cloned repository needs

## Installation

```bash
git clone <this repository> && cd oscar
pip install -r requirements.txt

# Fetch the local embedding model (~1.9 GB, into bert/, git-ignored):
python scripts/download_models.py
```

Set your API key in a `.env` file at the repository root (git-ignored; a
template lives in `.env.example`):

```bash
DEEPSEEK_API_KEY=sk-...
```

Providers map to environment variables: `DEEPSEEK_API_KEY` (deepseek),
`OPENAI_API_KEY` (openai), `ANTHROPIC_API_KEY` (anthropic),
`OPENAI_COMPATIBLE_API_KEY` (any OpenAI-compatible endpoint, set
`base_url` via config). If downloads fail on a restricted network, retry with
`HF_ENDPOINT=https://hf-mirror.com`.

## Usage

```bash
python main.py https://github.com/org/repo                          # repo only
python main.py https://github.com/org/repo --paper 2401.12345       # + paper (URL or arXiv ID)
python main.py https://github.com/org/repo --paper ... --output ./my_audit
```

Without `--paper`, claims are extracted from the repository's own README —
the run checks the repo's self-description (is what it advertises actually
there?) rather than fidelity to a paper; the report says so explicitly.

| Flag | Meaning |
|---|---|
| `repo_url` | GitHub repository to audit. |
| `--paper / -p` | Paper URL or bare arXiv ID (optional). |
| `--output / -o` | Output directory (default `./output`). |
| `--no-cleanup` | Keep the cloned repository after the audit (overrides `config.yaml`). |
| `--verbose / -v` | Full traceback on failure. |

Outputs in `output/<project>/`:

- `audit_report.md` — human-readable report: executive summary, claim
  table with verdicts, per-category evidence, issue investigation notes.
- `audit_result.json` — machine-readable findings, evidence details
  (file/line/snippet/explanation), per-category stats, overall weighted score.
- `repository_manifest.json` — files/classes/chunks snapshot for traceability.

## Configuration

Tuning parameters live in the committed root `config.yaml`, ready to run
as-is — edit in place. API keys never go here (see `.env`). **Precedence:**

```
CLI arguments  >  env var OSCAR_<SECTION>_<FIELD>  >  config.yaml  >  built-in defaults
```

| Section | What it controls | Example env var |
|---|---|---|
| `llm` | provider, endpoint, model, sampling | `OSCAR_LLM_MODEL` |
| `audit` | cleanup, clone retries/timeouts, issue search limits & keywords | `OSCAR_AUDIT_CLEANUP_REPO` |
| `retrieval` | code-search window & verdict thresholds | `OSCAR_RETRIEVAL_TOP_K` |
| `cache` | TTLs (seconds) for LLM/repo caches and the vector index | `OSCAR_CACHE_TTL_SECONDS` |

```bash
OSCAR_RETRIEVAL_TOP_K=5 python main.py https://github.com/org/repo
```

List-valued options (e.g. `audit.issue_search_keywords`) are YAML-only.
**API keys must never go into code or `config.yaml`** — a key under
`llm.api_key` is rejected with an error; keys belong exclusively in the
git-ignored `.env` (or exported shell variables).

## Caching & determinism

All LLM responses, paper PDFs/text, repository clones, and vector indexes live
under `.oscar_cache/` (git-ignored):

```
.oscar_cache/
├── llm/        LLM responses, keyed by sha256(exact messages+model+sampling)
├── papers/     downloaded papers by arXiv ID
├── repos/      promoted clones (per-URL, with validity markers)
└── vectors/    FAISS index + chunk metadata per paper and per repo
```

- Cache keys are hashes of the *exact* request bytes — any change to a prompt,
  model or temperature invalidates that entry (and only that entry).
- A warm cache makes re-audits **byte-identical and free**: no API calls.
  Determinism acceptance is a double run with `diff -r` on the outputs.
- TTL defaults to 7 days for LLM/repo caches and the code vector index —
  adjust via `config.yaml → cache`.
- Models using structured output may transparently fall back to a raw-JSON
  degrade path when the provider's function-calling is unavailable; both paths
  are cached and deterministic.

## Privacy & repository hygiene

- Secrets: `.env` only (`.env.example` is the committed template).
- The cloned repositories, LLM responses, paper PDFs, vector indexes, model
  weights, run outputs, and internal design documents are all git-ignored and
  stay local. Nothing is uploaded except the GitHub metadata the audit itself
  requests and the LLM API calls you configured.

## Repository layout

```
main.py                      CLI entry point
oscar/
├── config.py                layered config (defaults) — stdlib+dotenv+yaml only
├── prompts.py               the single home of all LLM prompts
├── llm/client.py            provider factory, cache-aware chat/structured calls
├── paper/                   paper download, text extraction, claim analysis
├── audit/                   GitHub issues/PRs investigation
├── mapping/                 code vector store (CodeBERT+FAISS/BM25/keyword)
├── graph/workflow.py        LangGraph pipeline
├── report/                  markdown/JSON report generation
├── models/schemas.py        pydantic state & report models
└── utils/                   cache (disk, utf-8 safe), git, progress, misc
scripts/
├── download_models.py       fetch bert/ models (idempotent, --dry-run)
└── verify_prompts.py        prompt-invariance harness (internal QA)
config.yaml                  committed runtime config (no keys)
.env.example                 committed secrets template (fill in .env)
```

## FAQ

**Does it cost money?** Only the LLM calls — a few per audit on the first run;
re-runs with a warm cache cost nothing.

**Can it run fully offline?** Audits need the LLM for judging and (when given)
issues investigation. Retrieval and indexing run locally regardless.

**Is the LLM trusted for coordinates?** No — locations are anchored to the
retrieved chunk records; LLM-reported positions that don't resolve are dropped
and un-anchorable verdicts become `UNCERTAIN`.

**Why download models at all?** Retrieval embeddings are computed locally with
CodeBERT for reproducibility and zero marginal API cost. `bert/` is git-ignored
by design — run `scripts/download_models.py` once per machine.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 The OSCAR Authors.
