"""Paper vector store — CodeBERT embeddings + FAISS indexing.

FAISS 替换了原有的 numpy 暴力搜索，使搜索复杂度从 O(n) 降至 O(log n)。
支持持久化到磁盘：FAISS .index 文件 + metadata JSON。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from oscar.models.schemas import PaperInfo

# 本仓库 checkpoint 由 MLM 预训练模型导出，含 lm_head.* 权重；加载为纯编码器
# (AutoModel) 时 lm_head 为 UNEXPECTED、pooler 为 MISSING（GraphCodeBERT 无
# pooler）。编码只用 encoder 的 last_hidden_state，两者均不会使用，属预期行为。
# 将 transformers.modeling_utils 日志级别提到 ERROR，静默该无意义报告。
# 另：CPU 场景下 TP（张量并行）plan 警告同样无意义，一并静默。
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.distributed.tensor_parallel").setLevel(logging.ERROR)

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "bert", "codebert-base")
_ENCODE_BATCH = 32  # 编码批大小:整批一次前向会形成超大 batch,CPU 上极慢且吃内存
# (与 code_vector_store 同理);分批后每批 ≤32 条,结果一致。


class PaperVectorStore:
    """FAISS-backed vector store for paper chunks.

    与旧版（numpy 余弦相似度）相比：
    - FAISS 使用 IVF 或 Flat 索引，检索速度无关 chunk 数量。
    - 持久化为二进制 .index 文件，加载毫秒级，远快于 JSON 序列化。
    """

    def __init__(self, model_path: str = _MODEL_PATH):
        self.model = AutoModel.from_pretrained(model_path, attn_implementation="eager")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self._device)
        self.model.eval()

        # FAISS 索引：dim=768 (CodeBERT 输出维度)
        self._dim = 768
        self._index: Optional[faiss.Index] = None
        self.chunks: list[dict] = []
        self.paper_info: Optional[PaperInfo] = None

    def index_paper(self, paper: PaperInfo):
        """Index all chunks into a FAISS flat index (exact search, L2)."""
        self.paper_info = paper
        self.chunks.clear()

        if not paper.chunks:
            self._index = None
            return

        texts = [c.text for c in paper.chunks]
        embeds = self._encode(texts)  # (N, 768)

        # FAISS 使用 L2 距离；归一化后等价于余弦相似度
        faiss.normalize_L2(embeds)
        self._index = faiss.IndexFlatIP(self._dim)  # Inner Product = cosine after L2 norm
        self._index.add(embeds)

        self.chunks = [
            {"chunk_id": c.chunk_id, "section": c.section, "text": c.text, "char_offset": c.char_offset}
            for c in paper.chunks
        ]

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """FAISS 检索：O(log n) 近似搜索，返回 Top-K chunk。

        返回结果按相似度降序排列，每项包含 chunk_id, section, text, score。
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        query_emb = self._encode([query])  # (1, 768)
        faiss.normalize_L2(query_emb)
        scores, indices = self._index.search(query_emb, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            results.append({**self.chunks[idx], "score": float(scores[0][i])})
        return results

    def get_section_by_name(self, name: str) -> Optional[dict]:
        name_lower = name.lower()
        for c in self.chunks:
            if name_lower in c["section"].lower():
                return c
        return None

    @property
    def section_count(self) -> int:
        return len(self.chunks)

    # -- FAISS 持久化 ------------------------------------------------------

    def save(self, arxiv_id: str):
        """将 FAISS 索引和元数据持久化到磁盘。

        - .faiss 文件：二进制索引，加载毫秒级
        - .json 文件：chunk 元数据
        """
        if self._index is None or not self.chunks:
            return

        cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", ".oscar_cache", "vectors")
        os.makedirs(cache_dir, exist_ok=True)

        index_path = os.path.join(cache_dir, f"{arxiv_id}.faiss")
        meta_path = os.path.join(cache_dir, f"{arxiv_id}.json")

        faiss.write_index(self._index, index_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"chunks": self.chunks}, f, ensure_ascii=False)

    def load(self, arxiv_id: str) -> bool:
        """从磁盘加载 FAISS 索引和元数据。"""
        cache_dir = os.path.join(os.path.dirname(__file__), "..", "..", ".oscar_cache", "vectors")
        index_path = os.path.join(cache_dir, f"{arxiv_id}.faiss")
        meta_path = os.path.join(cache_dir, f"{arxiv_id}.json")

        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            return False

        self._index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.chunks = data["chunks"]
        return True

    # -- 内部工具 ----------------------------------------------------------

    def _encode(self, texts: list[str]) -> np.ndarray:
        """CodeBERT 编码，mean pooling，返回 (N, 768) numpy 数组。

        分批编码：整批一次前向会构造超大 batch，CPU 上 attention 计算极慢且
        内存随 chunk 数线性膨胀；按 _ENCODE_BATCH 分批后耗时/内存量级下降，
        输出结果一致。
        """
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)
        embs = []
        for start in range(0, len(texts), _ENCODE_BATCH):
            batch = texts[start:start + _ENCODE_BATCH]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True, max_length=512,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                outputs = self.model(**encoded)
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            emb = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            embs.append(emb.cpu().numpy().astype(np.float32))
        return np.concatenate(embs, axis=0)


# 全局单例
paper_vector_store = PaperVectorStore()