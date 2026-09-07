"""Code Vector Store — hybrid retrieval using FAISS + BM25 + Keyword.

Supports three retrieval strategies fused via RRF:
1. Keyword (AST-based) exact match — class/function/filename lookup
2. BM25 keyword search — rank_bm25 on code text
3. Vector semantic search — CodeBERT embeddings indexed by FAISS

FAISS 替代了 numpy 暴力搜索，使向量检索复杂度从 O(n) 降至 O(log n)。

持久化:chunk 元数据(含 content)与 FAISS 二进制索引落盘到
.oscar_cache/vectors/code/{repo_key}.faiss/.json。仓库文件指纹
(relpath:mtime_ns:size)不变且未超 TTL 时直接加载索引,跳过整库
embedding——本机 CPU 上 1300+ chunk 的编码一次要 15-25 分钟。
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import time
from collections import defaultdict
from typing import Optional

import faiss
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoModel, AutoTokenizer

from oscar.config import config
from oscar.models.schemas import RepositoryManifest
from oscar.utils.cache import (
    code_vector_meta_read, code_vector_meta_write, code_vector_paths,
    atomic_replace,
)

# CodeBERT checkpoint 由 MLM 预训练导出，含 lm_head.* 权重；加载为纯编码器
# (AutoModel) 时 lm_head 为 UNEXPECTED（编码用不到）。pooler 缺失时会由
# transformers 随机初始化并报告 MISSING —— 本类编码只用 encoder 的
# last_hidden_state 做均值池化，不触碰 pooler_output，属预期行为。
# 将 transformers.modeling_utils 日志级别提到 ERROR，静默该无意义报告。
# 另：CPU 场景下 TP（张量并行）plan 警告同样无意义，一并静默。
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.distributed.tensor_parallel").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "bert", "codebert-base")
_ENCODE_BATCH = config.retrieval.encode_batch  # 编码批大小:整批一次前向在大仓库
# (上千代码块)上会形成超大 batch,CPU 上 attention 长时间空转且内存可达数 GB;
# 分批后每批 ≤32 条,量级下降(默认 32,config.yaml retrieval.encode_batch 可调)。

_INDEX_VERSION = "code-index-v1"  # 索引格式版本(结构性常量,不随配置走)
_INDEX_TTL = config.cache.index_ttl_seconds  # 与仓库缓存同周期:超期后仓库会
# 重克隆,索引必过期(默认 7 天,config.yaml cache.index_ttl_seconds 可调)


class CodeChunk:
    """A single code chunk (child) with its parent file context."""

    def __init__(
        self,
        chunk_id: str,
        file_path: str,
        content: str,
        *,
        class_name: str = "",
        function_name: str = "",
        parent_content: str = "",
        line_start: int = 0,
        line_end: int = 0,
    ):
        self.chunk_id = chunk_id
        self.file_path = file_path
        self.content = content
        self.class_name = class_name
        self.function_name = function_name
        self.parent_content = parent_content
        self.line_start = line_start
        self.line_end = line_end

    @property
    def search_text(self) -> str:
        parts = [self.file_path]
        if self.class_name:
            parts.append(f"class {self.class_name}")
        if self.function_name:
            parts.append(f"def {self.function_name}")
        parts.append(self.content[:500])
        return "\n".join(parts)


class CodeVectorStore:
    """Hybrid retrieval store for code using FAISS + BM25 + Keyword.

    与旧版（numpy 暴力搜索）相比：
    - FAISS IndexFlatIP 提供精确余弦相似度搜索，速度快 10-100 倍。
    - BM25 保留，用于弥补语义检索在短关键词上的不足。
    - 三路 RRF 融合保持不变。
    """

    def __init__(self, model_path: str = _MODEL_PATH):
        self.model = AutoModel.from_pretrained(model_path, attn_implementation="eager")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self._device)
        self.model.eval()

        self.chunks: list[CodeChunk] = []
        self.bm25: Optional[BM25Okapi] = None
        self._faiss_index: Optional[faiss.Index] = None
        self._tokenized: list[list[str]] = []
        self._dim = 768  # CodeBERT output dimension

    # -- Indexing ---------------------------------------------------------

    def index_repository(
        self,
        manifest: RepositoryManifest,
        repo_path: str,
        *,
        repo_url: str = "",
    ) -> bool:
        """Index (or restore) code units. Returns True when a valid on-disk
        index was loaded and embedding was skipped."""
        fingerprint = _repo_fingerprint(repo_path, manifest)

        if repo_url and fingerprint:
            meta = code_vector_meta_read(repo_url)
            if self._try_restore(meta, fingerprint):
                logger.info(
                    "Code vector index cache hit (%d chunks, fingerprint %s)",
                    len(self.chunks), fingerprint[:8],
                )
                return True

        self._build_from_manifest(manifest, repo_path)

        if repo_url and fingerprint:
            try:
                self._persist(repo_url, fingerprint)
            except Exception:
                logger.warning("Failed to persist code vector index", exc_info=True)
        return False

    def _try_restore(self, meta: Optional[dict], fingerprint: str) -> bool:
        """Restore FAISS + chunks from disk. All conditions must match, else
        the caller rebuilds from scratch — never silently reuse a stale index."""
        if not meta or meta.get("version") != _INDEX_VERSION:
            return False
        if not meta.get("repo_url") or meta.get("fingerprint") != fingerprint:
            return False
        if time.time() - meta.get("indexed_at", 0) > _INDEX_TTL:
            return False

        index_path, _ = code_vector_paths(meta["repo_url"])
        chunks_meta = meta.get("chunks")
        if not isinstance(chunks_meta, list) or not chunks_meta:
            return False
        try:
            faiss_index = faiss.read_index(str(index_path))
            if faiss_index.ntotal != len(chunks_meta):
                return False
        except Exception:
            return False

        try:
            chunks = [
                CodeChunk(
                    chunk_id=c["chunk_id"], file_path=c["file_path"],
                    content=c.get("content", ""),
                    class_name=c.get("class_name", "") or "",
                    function_name=c.get("function_name", "") or "",
                    line_start=c.get("line_start", 0),
                    line_end=c.get("line_end", 0),
                )
                for c in chunks_meta
            ]
        except Exception:
            return False

        self.chunks = chunks
        self._faiss_index = faiss_index
        self._rebuild_bm25()
        return True

    def _persist(self, repo_url: str, fingerprint: str):
        """Persist current index: .faiss first, meta json last (completion marker)."""
        _, meta_path = code_vector_paths(repo_url)
        os.makedirs(meta_path.parent, exist_ok=True)

        tmp_index = meta_path.with_suffix(".faiss.tmp")
        try:
            faiss.write_index(self._faiss_index, str(tmp_index))
            if not atomic_replace(tmp_index, meta_path.with_suffix(".faiss")):
                raise OSError("atomic replace failed for faiss index")
        except Exception:
            try:
                tmp_index.unlink(missing_ok=True)
            except Exception:
                pass
            raise

        meta = {
            "version": _INDEX_VERSION,
            "repo_url": repo_url,
            "indexed_at": time.time(),
            "fingerprint": fingerprint,
            "chunk_count": len(self.chunks),
            "chunks": [
                {
                    "chunk_id": c.chunk_id, "file_path": c.file_path,
                    "content": c.content,
                    "class_name": c.class_name, "function_name": c.function_name,
                    "line_start": c.line_start, "line_end": c.line_end,
                }
                for c in self.chunks
            ],
        }
        if not code_vector_meta_write(repo_url, meta):
            raise OSError("meta write failed for code vector index")

    def _rebuild_bm25(self):
        """Rebuild BM25 from chunks (pure local text processing, no encoding)."""
        self._tokenized = [_tokenize(c.search_text) for c in self.chunks]
        self.bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

    def _build_from_manifest(self, manifest: RepositoryManifest, repo_path: str):
        """Build chunks by slicing the exact line spans recorded by the AST
        manifest (class_locations / function_locations).

        旧实现按 ``def {func_name}(`` 文本搜索提取函数体,类内方法以
        ``"Client.method"`` 带点名存于 manifest.functions,精确查找必然失败,
        content 退化为 ``def Client.method(...)`` 占位串。按行号切片后每个
        chunk 都含真实函数/类体,并携带 line_start/line_end 供证据定位。
        """
        self.chunks.clear()
        self._tokenized.clear()

        file_contents: dict[str, str] = {}
        for f in manifest.python_modules:
            try:
                with open(os.path.join(repo_path, f), "r", encoding="utf-8", errors="replace") as fh:
                    file_contents[f] = fh.read()
            except Exception:
                file_contents[f] = ""

        # Per-method / top-level function chunks, sliced by manifest line spans
        for mod, locs in manifest.function_locations.items():
            parent = file_contents.get(mod, "")
            for loc in locs:
                body = _slice_lines(parent, loc.line_start, loc.line_end)
                name = loc.function_name or ""
                if loc.class_name:
                    body = body or f"def {loc.class_name}.{name}(...)"
                    chunk_id = f"fn-{mod}-{loc.class_name}.{name}"
                    self.chunks.append(CodeChunk(
                        chunk_id=chunk_id, file_path=mod, content=body,
                        class_name=loc.class_name, function_name=name,
                        parent_content=parent,
                        line_start=loc.line_start, line_end=loc.line_end,
                    ))
                else:
                    body = body or f"def {name}(...)"
                    chunk_id = f"fn-{mod}-{name}"
                    self.chunks.append(CodeChunk(
                        chunk_id=chunk_id, file_path=mod, content=body,
                        function_name=name, parent_content=parent,
                        line_start=loc.line_start, line_end=loc.line_end,
                    ))

        # Per-class chunks
        for mod, locs in manifest.class_locations.items():
            parent = file_contents.get(mod, "")
            for loc in locs:
                cls = loc.class_name or ""
                body = _slice_lines(parent, loc.line_start, loc.line_end) or f"class {cls}"
                self.chunks.append(CodeChunk(
                    chunk_id=f"cls-{mod}-{cls}", file_path=mod, content=body,
                    class_name=cls, parent_content=parent,
                    line_start=loc.line_start, line_end=loc.line_end,
                ))

        # Modules that failed AST parsing have no location entries: fall back
        # to whole-file chunks so their code is still searchable.
        located = set(manifest.function_locations) | set(manifest.class_locations)
        for mod in manifest.python_modules:
            if mod in located:
                continue
            content = file_contents.get(mod, "")
            self.chunks.append(CodeChunk(
                chunk_id=f"file-{mod}", file_path=mod,
                content=content or "", parent_content=content,
                line_start=1,
                line_end=max(content.count("\n") + 1, 1),
            ))

        if not self.chunks:
            for f in manifest.python_modules:
                content = file_contents.get(f, "")
                self.chunks.append(CodeChunk(
                    chunk_id=f"file-{f}", file_path=f,
                    content=content[:500], parent_content=content,
                ))

        self._rebuild_bm25()

        # FAISS index (Inner Product = cosine after L2 norm)
        texts = [c.search_text for c in self.chunks]
        embeds = self._encode(texts)
        faiss.normalize_L2(embeds)
        self._faiss_index = faiss.IndexFlatIP(self._dim)
        if embeds.shape[0] > 0:
            self._faiss_index.add(embeds)

    # -- Individual retrieval strategies ---------------------------------

    def _encode(self, texts: list[str]) -> np.ndarray:
        """CodeBERT encoding with mean pooling → (N, 768) float32.

        分批编码:整批一次前向在仓库上千代码块时会构造 (N, 256) 的超大 batch,
        CPU 上 attention 软计算需数十分钟、内存数 GB。按 _ENCODE_BATCH 分批
        后内存/耗时下降两个数量级,输出结果一致(同一模型、同一池化)。
        """
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)
        embs = []
        for start in range(0, len(texts), _ENCODE_BATCH):
            batch = texts[start:start + _ENCODE_BATCH]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True, max_length=256,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                outputs = self.model(**encoded)
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            emb = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            embs.append(emb.cpu().numpy().astype(np.float32))
        return np.concatenate(embs, axis=0)

    def _result(self, chunk: CodeChunk, score: float, source: str) -> dict:
        """Unified retrieval result dict (content/parent_content included for
        downstream LLM grounding; line spans for evidence locations)."""
        return {
            "chunk_id": chunk.chunk_id,
            "file_path": chunk.file_path,
            "class_name": chunk.class_name,
            "function_name": chunk.function_name,
            "content": chunk.content,
            "parent_content": chunk.parent_content,
            "line_start": chunk.line_start,
            "line_end": chunk.line_end,
            "score": score,
            "source": source,
        }

    def chunks_for_files(
        self, file_paths: list[str], budget: int = 6
    ) -> list[dict]:
        """Structured recall: chunks of the mapper-identified files, up to
        ``budget`` total, spread round-robin across files (class-level first,
        then method-level, in store order per file).

        Paper method names rarely equal code names, and pure retrieval can
        miss the implementing module (which may sit under an unrelated class
        name).  The mapper's file-level
        strategies (exact/partial filename, symbol name, category patterns)
        DO lock such files — the grounder feeds their chunks to the LLM as
        structured context on top of hybrid results.  Depth comes from the
        hybrid pass; this pass favors breadth so one noisy file cannot eat
        the whole budget before the real implementation file is reached.
        """
        # Per-file ordered chunk lists: class chunks first — biggest class
        # first (实现往往集中在主导大类里),then method chunks in store
        # order (deterministic).
        buckets: list[list[CodeChunk]] = []
        for fp in file_paths:
            pool = [c for c in self.chunks if c.file_path == fp]
            if not pool:
                continue
            classes = sorted(
                [c for c in pool if c.class_name and not c.function_name],
                key=lambda c: (c.line_end - c.line_start, -c.line_start),
                reverse=True,
            )
            ordered = classes + [c for c in pool if c.function_name]
            if ordered:
                buckets.append(ordered)

        out: list[dict] = []
        seen: set[str] = set()
        # 轮转:每轮每个文件出一个未用块,预算用尽即停;整轮无新块则终止
        while len(out) < budget and buckets:
            added_this_round = 0
            for bucket in buckets:
                if len(out) >= budget:
                    break
                while bucket:
                    c = bucket.pop(0)
                    if c.chunk_id not in seen:
                        seen.add(c.chunk_id)
                        out.append(self._result(c, 0.0, "mapper-file"))
                        added_this_round += 1
                        break
            if added_this_round == 0:
                break
        return out

    def keyword_search(self, query: str, top_k: int = 10) -> list[dict]:
        """Exact-term match on symbol/filename, plus content-term overlap.

        论文方法语义词几乎不与代码同名,但实现方常把"这段代码是论文里的什么"
        写进 docstring/注释(直引论文概念措辞)。因此当
        statement/论文原句里的实义词(≥4 字母、非停用词)在 chunk 文本里出现
        ≥2 个时,按命中数给分——docstring 是作者自己写的"代码↔论文概念"映射,
        比语义向量可靠。
        """
        keywords = _extract_keywords(query)
        content_terms = sorted({
            t for t in _tokenize(query)
            if len(t) >= 4 and t not in _CONTENT_STOPWORDS
        })
        # 命中按 IDF 加权:docstring 里的论文概念词在语料中稀有 → 高权重;
        # 代码库高频泛词(module/model/train 等)到处都是 → 低权重,避免
        # 高词频文件淹没真正的"docstring 即论文概念"命中。
        idf: dict[str, float] = {}
        if content_terms:
            n = max(len(self.chunks), 1)
            df = {t: 0 for t in content_terms}
            for c in self.chunks:
                hay = c.search_text.lower()
                for t in content_terms:
                    if t in hay:
                        df[t] += 1
            idf = {t: math.log(1.0 + n / max(d, 1)) for t, d in df.items()}
        results = []
        seen = set()
        for chunk in self.chunks:
            key = chunk.chunk_id
            if key in seen:
                continue
            score = 0.0
            for kw in keywords:
                kw_lower = kw.lower()
                fname = os.path.basename(chunk.file_path).lower()
                fname_no_ext = os.path.splitext(fname)[0]
                if kw_lower == fname_no_ext:
                    score = max(score, 1.0)
                elif chunk.class_name and kw_lower == chunk.class_name.lower():
                    score = max(score, 1.0)
                elif chunk.function_name and kw_lower == chunk.function_name.lower():
                    score = max(score, 1.0)
                # 子串匹配只对「像代码名的关键词」生效(In/Since/Alg 这类句首
                # 大写英文词会子串命中几乎所有类名,如 "in" ⊂ "Recording")
                elif (
                    len(kw_lower) >= 4
                    and kw_lower not in _CONTENT_STOPWORDS
                    and (kw_lower in fname
                         or (chunk.class_name and kw_lower in chunk.class_name.lower()))
                ):
                    score = max(score, 0.5)
            if score <= 0 and content_terms:
                hay = chunk.search_text.lower()
                matched = [t for t in content_terms if t in hay]
                if len(matched) >= 2:
                    # 原始 IDF 累加、高上限归一:稀有概念词组合(罕见词 idf
                    # 合计 ≈ 16)必须显著高于常见泛词组合(≈ 3),不能靠 9.0
                    # 封顶把两者都拍成 1.0 让并列回退到 store 序
                    score = min(sum(idf[t] for t in matched), 30.0) / 30.0
            if score > 0:
                seen.add(key)
                results.append(self._result(chunk, score, "keyword"))
        # 确定性:同分按 store 顺序(keyword 按 chunk 序 append),不做 set 排序
        return sorted(results, key=lambda r: -r["score"])[:top_k]

    def rare_token_search(self, query: str, top_k: int = 3, df_cap: int = 30) -> list[dict]:
        """Rare-content-term scan (last-resort recall for grounding).

        论文 claim 与实现代码有时只共享单个低频短 token —— 如论文以功能句
        描述某机制,代码里只有对应概念的局部变量。这类词常因长度(<4)
        进不了 content-match、代码又不同名、语义向量也够不着。这里把 query
        中语料稀有(df ≤ df_cap)的**短代码味 token**
        (3-4 字母纯字母、非停用词)直接对 chunk 全文做子串扫描,把含这些词
        的代码块带回候选窗口。

        只收短 token 的原因:5 字母以上的普通英文词(partial/layer/weighted…)
        在论文句与库代码 docstring 里都常见,df 过滤挡不住,会把无关实现
        扫进上下文;而 3-4 字母 token(如 tgt/seg/idx)绝大多数
        是代码标识符,df 稀有过滤有效。泛词(高频)由 df_cap 挡掉,常见短
        虚词(post/any/for…)由 _SHORT_CODE_STOPWORDS 挡掉。
        匹配要求「标识符边界」:短词必须作为独立标识符出现('seg = …' 命中,
        'seg_id'/'SegmentRef' 不命中),避免 arg 选项名与类名前缀
        把无关文件扫进窗口。
        """
        tokens = {
            t for t in _tokenize(query)
            if 3 <= len(t) <= 4 and t.isalpha()
            and t not in _CONTENT_STOPWORDS and t not in _SHORT_CODE_STOPWORDS
        }
        if not tokens:
            return []
        haystack = [(c, c.content.lower()) for c in self.chunks]
        n = max(len(self.chunks), 1)
        terms: list[tuple[str, float, "re.Pattern[str]"]] = []
        for t in sorted(tokens):  # 确定性:df 与候选词表与 token 顺序无关
            pat = re.compile(r"(?<![a-z0-9_])" + re.escape(t) + r"(?![a-z0-9_])")
            df = sum(1 for _, h in haystack if pat.search(h))
            if 0 < df <= df_cap:
                terms.append((t, math.log(1.0 + n / df), pat))
        if not terms:
            return []
        scored: list[tuple[float, CodeChunk]] = []
        for c, h in haystack:
            # 权重 = Σ idf × min(出现次数, 5);且要求单块内出现 ≥2 次才计权:
            # 真正使用该术语的计算代码(循环里反复出现的同一标识符)排在最前,
            # 「参数行/注释里提过一次」的块(task/more 等英文词偶然命中)不计。
            w = 0.0
            for _, wt, pat in terms:
                cnt = len(pat.findall(h))
                if cnt >= 2:
                    w += wt * min(cnt, 5)
            if w > 0:
                scored.append((w, c))
        # 确定性:权重降序、同权按 store 顺序(list.sort 稳定)
        scored.sort(key=lambda pair: -pair[0])
        return [
            self._result(c, round(w / 30.0, 4), "rare-token")
            for w, c in scored[:top_k]
        ]

    def bm25_search(self, query: str, top_k: int = 10) -> list[dict]:
        if not self.bm25 or not self.chunks:
            return []
        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        results = []
        for i in top_indices:
            if scores[i] > 0:
                results.append(self._result(self.chunks[i], float(scores[i]), "bm25"))
        return results

    def vector_search(self, query: str, top_k: int = 10) -> list[dict]:
        """FAISS 语义检索（余弦相似度，L2 归一化后等价于 Inner Product）。"""
        if self._faiss_index is None or self._faiss_index.ntotal == 0 or not self.chunks:
            return []
        query_emb = self._encode([query])
        faiss.normalize_L2(query_emb)
        scores, indices = self._faiss_index.search(query_emb, top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            results.append(self._result(self.chunks[idx], float(scores[0][i]), "vector"))
        return results

    # -- Fused search ----------------------------------------------------

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        *,
        keyword_weight: float = 0.4,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.3,
        rrf_k: int = 60,
    ) -> list[dict]:
        """Hybrid search fusing keyword, BM25, and FAISS via RRF."""
        kw_results = self.keyword_search(query, top_k=top_k * 2)
        bm25_results = self.bm25_search(query, top_k=top_k * 2)
        vec_results = self.vector_search(query, top_k=top_k * 2)

        rank_scores: dict[str, float] = defaultdict(float)
        for rank, r in enumerate(kw_results):
            rank_scores[r["chunk_id"]] += keyword_weight / (rrf_k + rank + 1)
        for rank, r in enumerate(bm25_results):
            rank_scores[r["chunk_id"]] += bm25_weight / (rrf_k + rank + 1)
        for rank, r in enumerate(vec_results):
            rank_scores[r["chunk_id"]] += vector_weight / (rrf_k + rank + 1)

        chunk_map = {c.chunk_id: c for c in self.chunks}
        sorted_ids = sorted(rank_scores, key=lambda x: rank_scores[x], reverse=True)[:top_k]

        return [
            self._result(chunk_map[cid], round(rank_scores[cid], 4), "hybrid")
            for cid in sorted_ids if cid in chunk_map
        ]


# -- Helpers ---------------------------------------------------------------

def _repo_fingerprint(repo_path: str, manifest: RepositoryManifest) -> str:
    """Hash of (relpath:mtime_ns:size) over python modules — cheap, no reads."""
    h = hashlib.sha256()
    for rel in sorted(manifest.python_modules):
        try:
            st = os.stat(os.path.join(repo_path, rel))
            h.update(f"{rel}:{st.st_mtime_ns}:{st.st_size}\n".encode("utf-8"))
        except OSError:
            h.update(f"{rel}:missing\n".encode("utf-8"))
    return h.hexdigest()


def _slice_lines(content: str, line_start: int, line_end: int) -> str:
    """Return lines [line_start, line_end] (1-indexed, inclusive) as text."""
    if not content or line_start < 1:
        return ""
    lines = content.split("\n")
    hi = min(line_end, len(lines))
    lo = line_start - 1
    if lo >= hi:
        return ""
    return "\n".join(lines[lo:hi])


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{1,}", text.lower())


# 检索内容词命中时过滤掉的通用词(散文虚词 + 代码高频泛词)。太泛的词会让
# 每个 dataset/models 文件都 ≥2 命中,淹没真正的 docstring 映射命中。
_CONTENT_STOPWORDS = frozenset({
    "that", "this", "these", "those", "with", "from", "have", "been",
    "between", "without", "used", "when", "lets", "handled", "could",
    "would", "should", "into", "does", "will", "was", "were", "are",
    "but", "than", "then", "only", "also", "other", "such", "over",
    "under", "while", "after", "before", "their", "there", "where",
    "which", "about", "because", "however", "method", "methods",
    "process", "paper", "context", "model", "models", "data", "train",
    "training", "view", "views", "image", "images", "batch", "sample",
    "samples", "set", "sets", "file", "files", "path", "paths", "list",
    "lists", "dict", "dicts", "return", "value", "values", "base",
    "based", "default", "current", "same", "different", "several",
    "first", "second", "other", "part", "parts", "both", "each", "our",
    "you", "your", "using", "given", "shown", "need", "allows", "allow",
    "since", "have", "been", "lets", "over", "after", "from",
})

# 3-4 字母代码味 token 路径的补充停用词:这些短英文虚词/泛词在论文句与库
# docstring 里到处都是(post/any/for/the…),长度过滤挡不住,df 过滤也常误放。
_SHORT_CODE_STOPWORDS = frozenset({
    "for", "the", "and", "any", "all", "but", "can", "not", "our", "out",
    "its", "was", "who", "how", "why", "too", "two", "via", "use", "new",
    "end", "one", "may", "per", "non", "non", "e.g", "i.e", "say", "let",
    "get", "set", "has", "had", "did", "his", "her", "she", "him", "are",
    "did", "yet", "now", "off", "own", "few", "far", "ago", "put", "run",
    "end", "try", "fit", "add", "see", "say", "okay", "ok", "oh", "eh",
    "post", "pre", "sub", "anti", "semi", "geo",
})


def _extract_keywords(text: str) -> list[str]:
    keywords = re.findall(r"[A-Z][a-zA-Z0-9_]+", text)
    for pattern in [r"'(.*?)'", r"\"(.*?)\"", r"`(.*?)`"]:
        keywords.extend(re.findall(pattern, text))
    # 排序后再截断:set 迭代序跨进程随机(PYTHONHASHSEED),会扰动关键词命中的
    # 先后顺序,进而使 RRF 并列时的 top-k 检索结果在两次运行间翻转。
    return sorted(set(keywords))[:10]


# Global singleton
code_vector_store = CodeVectorStore()
