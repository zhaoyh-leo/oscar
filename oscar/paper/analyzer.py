"""Paper analyzer - extract claims from paper text.

Uses a combination of rule-based extraction and LLM-based structured summarization.
"""

import re
from typing import Optional

from oscar.llm.client import llm_client
from oscar.prompts import SYSTEM_PAPER_ASSISTANT, build_claim_extraction_prompt
from oscar.models.schemas import (
    Claim,
    ClaimCategory,
    ClaimSource,
    PaperInfo,
)


def extract_claims_from_paper(
    paper: Optional[PaperInfo], readme_text: Optional[str] = None
) -> list[Claim]:
    """Extract claims from paper and README text.

    First uses rule-based extraction, then LLM for structured summarization.
    """
    claims = []

    # Rule-based extraction from paper
    if paper and (paper.full_text or paper.abstract):
        paper_text = paper.full_text or paper.abstract or ""
        claims.extend(_rule_based_claims(paper_text, paper))

    # Rule-based extraction from README
    if readme_text:
        claims.extend(_readme_claims(readme_text))

    # Use LLM to supplement and refine claims
    if paper and (paper.full_text or paper.abstract):
        llm_claims = _llm_claim_extraction(paper, readme_text)
        claims.extend(llm_claims)

    # Deduplicate
    claims = _deduplicate_claims(claims)
    return claims


def extract_claims_from_chunk(
    chunk_text: str,
    readme_text: Optional[str] = None,
) -> list[Claim]:
    """Extract claims from a single paper chunk (no PaperInfo object needed)."""
    claims = _rule_based_claims(chunk_text, PaperInfo())
    llm_claims = _llm_claim_extraction(
        PaperInfo(full_text=chunk_text),
        readme_text,
    )
    if llm_claims:
        claims.extend(llm_claims)
    return _deduplicate_claims(claims)


def _rule_based_claims(text: str, paper: PaperInfo) -> list[Claim]:
    """Extract basic claims using rule-based patterns.

    规则层只做「保守兜底」,不做语义扩写(语义语句由 LLM 抽取产出):
    - 数据集语境句子(dataset/benchmark + 构建语境)归 DATASET,绝不再把
      "dataset named …" 之类误抽成 CORE_METHOD(数据集句内抓名是历史误判根因);
    - 方法语境句子(we propose/… )在句内/句尾抓方法名 → CORE_METHOD,
      source.content 记录整句原文(≈350 字符),供下游检索与 grounding 用语义;
    - 规则产出的仍只是名字,是否实现交由 evidence_grounder 按代码裁决。
    """
    claims = []
    counter = [1]

    # 论文转出的 txt 每行 ~75 字符硬回行 → 先并行走、再切句
    normalized = re.sub(r"\s+", " ", text.replace("\n", " "))
    sentences = _split_sentences(normalized)

    dataset_ctx = re.compile(
        r"\b(dataset|benchmark|image pairs|imagery|corpus)\b", re.IGNORECASE
    )
    construct_ctx = re.compile(
        r"\b(we|our|this paper|this work|introduce|construct|collect|build|"
        r"create|propose|release|generate|utilize)\b",
        re.IGNORECASE,
    )
    method_ctx = re.compile(
        r"\b(we propose|we introduce|we present|we design|our method|our "
        r"approach|our framework|our model|our network|proposed method|"
        r"proposed approach|proposed framework)\b",
        re.IGNORECASE,
    )

    for sentence in sentences:
        low = sentence.lower()
        is_dataset_sentence = bool(dataset_ctx.search(low))

        # --- 数据集句 → DATASET claim(仅当句中有构建/自述语境) ---
        named = re.search(r"\b(?:named|called)\s+([A-Z][A-Za-z0-9_-]+)", sentence)
        if is_dataset_sentence and named and construct_ctx.search(low):
            ds_name = named.group(1)
            claims.append(Claim(
                claim_id=f"DATASET-{counter[0]:03d}",
                category=ClaimCategory.DATASET,
                statement=f"Dataset/benchmark '{ds_name}' is constructed.",
                source=ClaimSource(
                    type="paper", location="Full text",
                    content=sentence[:350],
                ),
            ))
            counter[0] += 1
            continue  # 数据集句内不再抓方法名(见上方规则说明)

        # --- 方法句 → CORE_METHOD 名字 ---
        if method_ctx.search(low):
            method_name = None
            # 句中紧跟大写名的变体(we propose FooNet / our method Bar)
            m = re.search(
                r"(?:we propose|we introduce|we present|we design|our method|"
                r"our approach)\s+(?:a |an |the |novel |new )?"
                r"([A-Z][A-Za-z0-9]+(?:Net|Former|Transformer|GAN|Model|"
                r"Module|Block|Layer|Network|VLAD|NCE|Former|Encoder|Decoder)\b)",
                sentence,
            )
            if m:
                method_name = m.group(1)
            # 小写起头连字符技术名(xxx-NCE 形态):限定论文技术后缀
            # 白名单(NCE/Net/Former/...),避免 cross-view、real-world 等普通
            # 连字符形容词误报。
            if method_name is None:
                m = re.search(
                    r"\b[a-z][A-Za-z0-9_]*-(?:[A-Z][A-Za-z0-9_-]*"
                    r"(?:NCE|Net|Former|GAN|Loss|VLAD|Encoder|Decoder|Model|"
                    r"Block|Layer|Adapter|SAM|CLIP))\b",
                    sentence,
                )
                if m:
                    method_name = m.group(0)
            if method_name:
                claims.append(Claim(
                    claim_id=f"METHOD-{counter[0]:03d}",
                    category=ClaimCategory.CORE_METHOD,
                    statement=method_name,  # 名字本身;语义由 LLM claim 承载
                    source=ClaimSource(
                        type="paper", location="Full text",
                        content=sentence[:350],
                    ),
                ))
                counter[0] += 1

    # Training keywords (保持原语义:论文提及训练过程即给一条占位 claim)
    if re.search(r"train", text, re.IGNORECASE):
        claims.append(
            Claim(
                claim_id=f"TRAIN-{counter[0]:03d}",
                category=ClaimCategory.TRAINING,
                statement="The paper describes a training procedure.",
                source=ClaimSource(
                    type="paper",
                    location="Full text",
                    content="Training-related keywords found.",
                ),
            )
        )
        counter[0] += 1

    # Evaluation keywords
    if re.search(r"(?:evaluat|benchmark|experiment)", text, re.IGNORECASE):
        claims.append(
            Claim(
                claim_id=f"EVAL-{counter[0]:03d}",
                category=ClaimCategory.EVALUATION,
                statement="The paper describes evaluation/experiments.",
                source=ClaimSource(
                    type="paper",
                    location="Full text",
                    content="Evaluation-related keywords found.",
                ),
            )
        )
        counter[0] += 1

    return claims


def _split_sentences(text: str) -> list[str]:
    """Split normalized text into sentences (end with . ! ? 后跟大写/数字)."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", text)
    return [p.strip() for p in parts if p.strip()]


def _readme_claims(text: str) -> list[Claim]:
    """Extract claims from README."""
    claims = []
    counter = [100]

    # Check for training instructions
    if re.search(r"(?:train|training)", text, re.IGNORECASE):
        claims.append(
            Claim(
                claim_id=f"TRAIN-{counter[0]:03d}",
                category=ClaimCategory.TRAINING,
                statement="README claims training functionality is available.",
                source=ClaimSource(
                    type="readme",
                    location="README",
                    content="Training keywords found in README.",
                ),
            )
        )
        counter[0] += 1

    # Check for inference/demo
    if re.search(r"(?:infer|demo|predict|example)", text, re.IGNORECASE):
        claims.append(
            Claim(
                claim_id=f"INFER-{counter[0]:03d}",
                category=ClaimCategory.INFERENCE,
                statement="README claims inference/demo functionality is available.",
                source=ClaimSource(
                    type="readme",
                    location="README",
                    content="Inference/demo keywords found.",
                ),
            )
        )
        counter[0] += 1

    # Check for evaluation
    if re.search(r"(?:evaluat|benchmark|test)", text, re.IGNORECASE):
        claims.append(
            Claim(
                claim_id=f"EVAL-{counter[0]:03d}",
                category=ClaimCategory.EVALUATION,
                statement="README claims evaluation functionality is available.",
                source=ClaimSource(
                    type="readme",
                    location="README",
                    content="Evaluation keywords found.",
                ),
            )
        )
        counter[0] += 1

    # Check for API/interface claims
    if re.search(r"(?:api|interface|function|class|method|CLI)", text, re.IGNORECASE):
        claims.append(
            Claim(
                claim_id=f"API-{counter[0]:03d}",
                category=ClaimCategory.API_INTERFACE,
                statement="README references API/interfaces.",
                source=ClaimSource(
                    type="readme",
                    location="README",
                    content="API/interface keywords found.",
                ),
            )
        )
        counter[0] += 1

    # Check for dataset/checkpoint references
    if re.search(
        r"(?:dataset|checkpoint|pretrained|model.*weight|download)", text, re.IGNORECASE
    ):
        claims.append(
            Claim(
                claim_id=f"RES-{counter[0]:03d}",
                category=ClaimCategory.DATASET,
                statement="README references external resources (datasets/checkpoints).",
                source=ClaimSource(
                    type="readme", location="README", content="Resource keywords found."
                ),
            )
        )
        counter[0] += 1

    # Check for license
    if re.search(r"(?:license|licence|mit|apache|cc-by)", text, re.IGNORECASE):
        claims.append(
            Claim(
                claim_id=f"LIC-{counter[0]:03d}",
                category=ClaimCategory.LICENSE,
                statement="README mentions license information.",
                source=ClaimSource(
                    type="readme", location="README", content="License keywords found."
                ),
            )
        )
        counter[0] += 1

    return claims


def _llm_claim_extraction(
    paper: PaperInfo, readme_text: Optional[str] = None
) -> list[Claim]:
    """Use LLM to extract structured claims from paper text.

    核心要求(2025-09 改):claim 必须承载论文对方法/贡献的**功能语义**,
    而非裸名 —— 下游检索与证据接地只有拿到"方法做什么"才能命中真正实现它的
    代码(仓库代码极少与论文同名,实现常换名发布)。
    """
    paper_text = paper.full_text or paper.abstract or ""
    window = _paper_claim_window(paper_text) if paper.full_text else (paper_text or "")[:8000]

    prompt = build_claim_extraction_prompt(window, readme_text)

    # 偶发 API 抖动会返回空/无效 JSON → 重试一次(消息相同,命中缓存则免费;
    # 落空则真实重调)。两次失败才认输返回 []。
    last_result: object = {}
    for attempt in range(2):
        try:
            result = llm_client.chat_json(
                [
                    {
                        "role": "system",
                        "content": SYSTEM_PAPER_ASSISTANT,
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            # result may be a bare list, {"items": [...]}, {"claims": [...]}, or
            # a single claim object — accept all shapes; {} failure → no claims.
            raw_items: object = result
            if isinstance(result, dict):
                for key in ("items", "claims"):
                    if isinstance(result.get(key), list):
                        raw_items = result[key]
                        break
                else:
                    if any(
                        k in result for k in ("category", "statement", "snippet", "location")
                    ):
                        raw_items = [result]
                    else:
                        raw_items = []
            if isinstance(raw_items, list) and raw_items:
                claims = []
                for i, item in enumerate(raw_items):
                    if not isinstance(item, dict):
                        continue
                    try:
                        category = ClaimCategory(item.get("category", "core_method"))
                    except ValueError:
                        category = ClaimCategory.CORE_METHOD
                    claims.append(
                        Claim(
                            claim_id=f"LLM-{i+1:03d}",
                            category=category,
                            statement=(item.get("statement", "") or "")[:250],
                            source=ClaimSource(
                                type="paper",
                                location=item.get("location", "Unknown"),
                                content=(item.get("snippet", "") or "")[:350],
                            ),
                        )
                    )
                return _refine_core_method_categories(claims)
            last_result = result
        except Exception:
            last_result = {}
    return []


def _paper_claim_window(full_text: str, max_chars: int = 18000) -> str:
    """Build the LLM extraction window: abstract/intro head + claim sentences.

    arXiv 转出的全文可达数万字符,旧实现直接取 [:8000] 常常截断正文方法段
    (方法细节在 8000 字符之后)。这里拼接:头部(摘要+引言,常含贡献清单)+
    全文命中"自述/方法词汇"的句子(各带 ±1 句上下文),总量封顶 max_chars。
    """
    normalized = re.sub(r"\s+", " ", full_text.replace("\n", " "))
    sentences = _split_sentences(normalized)

    verb_kw = re.compile(
        r"\b(propose|proposed|introduce|introduced|present|presented|design|"
        r"develop|contribute|construct|build|create|collect|adopt|utilize|"
        r"demonstrate|we release|we open[- ]source)\b",
        re.IGNORECASE,
    )
    ctx_kw = re.compile(
        r"\b(method|approach|framework|loss|model|network|architecture|module|"
        r"dataset|benchmark|training|objective|paradigm|strategy|procedure|"
        r"descriptor|retrieval|localization)\b",
        re.IGNORECASE,
    )

    selected: set[int] = set()
    for idx, sentence in enumerate(sentences):
        if verb_kw.search(sentence) and ctx_kw.search(sentence):
            for j in range(max(0, idx - 1), min(len(sentences), idx + 2)):
                selected.add(j)

    # 头 6000 字符(abstract/intro)必须在内
    head_len = 0
    for idx, sentence in enumerate(sentences):
        if head_len >= 6000:
            break
        selected.add(idx)
        head_len += len(sentence) + 1

    parts = [s for i, s in enumerate(sentences) if i in selected]
    out = ""
    for part in parts:
        if len(out) + len(part) + 1 > max_chars:
            break
        out += part + " "
    return out.strip()


def _refine_core_method_categories(claims: list[Claim]) -> list[Claim]:
    """类别守卫:论文方法语境外的「评估协议/指标」内容不得占 core_method。

    LLM 偶把 Evaluation-protocol/metric(Recall@K/AP/SDM@K/Dis@1)或
    benchmark 对照实验类陈述归进 core_method —— 这类 claim 进了 core_method
    审计后要么无代码可比(协议/对照无独立实现)被误判 MISSING,拉低方法类
    分数,要么语义错位。此处只对明显的评估类措辞保守重分类:
    指标/评估协议 → EVALUATION;benchmark 对照设置 → BENCHMARK。
    """
    metric_terms = re.compile(
        r"\b(evaluation protocol|evaluation metric|evaluation settings|"
        r"dis@1|sdm@\s?k|recall@\s?k|map@\s?k|top-1)\b",
        re.IGNORECASE,
    )
    eval_ctx = re.compile(
        r"\b(evaluat|metric|report|retrieval|localiz|result)\b",
        re.IGNORECASE,
    )
    out = []
    for c in claims:
        cat = c.category
        if cat == ClaimCategory.CORE_METHOD:
            text = f"{c.statement} {c.source.content if c.source else ''}"
            low = text.lower()
            if metric_terms.search(low) and eval_ctx.search(low):
                cat = ClaimCategory.EVALUATION
            elif re.search(r"\bbenchmark\b", low) and re.search(
                r"\b(compares|comparison|settings|against)\b", low
            ):
                cat = ClaimCategory.BENCHMARK
        if cat is not c.category:
            c = c.model_copy(update={"category": cat})
        out.append(c)
    return out


def claim_search_text(claim: Claim) -> str:
    """Semantic search text for a claim: statement + paper source sentence.

    mapper 的 hybrid 召回与 grounder 的检索查询共用:statement 常只有方法名,
    论文原句(claim.source.content)提供"这个方法做什么"的语义词。
    """
    text = claim.statement or ""
    if claim.source and claim.source.type == "paper" and claim.source.content:
        text = f"{text} paper context: {claim.source.content}"
    return text.strip()


def _deduplicate_claims(claims: list[Claim]) -> list[Claim]:
    """Deduplicate claims by statement (case-insensitive), then fold rule-level
    bare-name claims that a richer LLM claim already covers.

    规则层只保底,产出的裸方法名 claim 只有名字、没有语义;LLM 功能句
    以同名开头且语义是其超集。两者并立会让同一方法被审计两次、重复计分
    (报告里同一标题出现两次)。LLM 功能句存在时按名折叠规则裸名;LLM 全缺
    时规则 claim 原样保留(仍是兜底)。
    """
    seen = set()
    unique = []
    for c in claims:
        key = c.statement.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(c)

    llm_statements = [
        c.statement.lower().strip()
        for c in unique
        if c.claim_id.startswith("LLM-")
        and c.statement
    ]
    if not llm_statements:
        return unique

    folded = []
    for c in unique:
        if c.claim_id.startswith("LLM-"):
            folded.append(c)
            continue
        low = c.statement.lower().strip()
        if len(low) >= 3 and any(low in s and low != s for s in llm_statements):
            continue  # 被同名功能句覆盖的规则裸名,折叠掉
        folded.append(c)
    return folded
