"""Category policy — scoring weights, display order, grounding scope.

策略常量集中放这里(不放 config.py:权重是产品口径而非运行时配置;
不放 schemas.py:model 层不依赖 policy)。report/generator 与
mapping/evidence_grounder 共用。

用户确认的口径:
- 核心实现类参与完整度计分并按类别加权(core_method/training 最重)。
- api_interface / license / checkpoint / external_resource / release 属
  「有更好,没有也不减分」的外围内容——单独列出供参考,不进入计分域。
- 非代码内容(数据来源、README/URL 类证据)不做代码接地,但类别本身
  仍可能计分(dataset 的资源可得性 finding 即属此类)。
"""

from oscar.models.schemas import ClaimCategory

# 计分类别 → 权重(加权平均:score = Σ w_c·(v+0.5i)/n_c / Σ w_c)
CATEGORY_WEIGHTS: dict[ClaimCategory, float] = {
    ClaimCategory.CORE_METHOD: 3.0,
    ClaimCategory.TRAINING: 3.0,
    ClaimCategory.DATASET: 2.0,
    ClaimCategory.EVALUATION: 2.0,
    ClaimCategory.INFERENCE: 2.0,
    ClaimCategory.BENCHMARK: 1.5,
    ClaimCategory.DEMO: 1.0,
    ClaimCategory.IMPLEMENTATION: 1.0,
}

# 不计分、仅参考列出(license 作为独立警示项)
INFO_ONLY_CATEGORIES: set[ClaimCategory] = {
    ClaimCategory.API_INTERFACE,
    ClaimCategory.LICENSE,
    ClaimCategory.CHECKPOINT,
    ClaimCategory.EXTERNAL_RESOURCE,
    ClaimCategory.RELEASE,
}

# 展示顺序:计分区内类别(按重要性),再补 Informational 区
SCORED_DISPLAY_ORDER: list[ClaimCategory] = [
    ClaimCategory.CORE_METHOD,
    ClaimCategory.TRAINING,
    ClaimCategory.INFERENCE,
    ClaimCategory.EVALUATION,
    ClaimCategory.DATASET,
    ClaimCategory.BENCHMARK,
    ClaimCategory.DEMO,
    ClaimCategory.IMPLEMENTATION,
]

INFO_DISPLAY_ORDER: list[ClaimCategory] = [
    ClaimCategory.API_INTERFACE,
    ClaimCategory.LICENSE,
    ClaimCategory.CHECKPOINT,
    ClaimCategory.EXTERNAL_RESOURCE,
    ClaimCategory.RELEASE,
]

# 可代码接地的类别:grounder 用「代码原文 → LLM 审查」产出定位+自然语言摘要。
# 注意 api_interface 在 INFO_ONLY 里但仍要接地(它的证据空洞正是用户抱怨之一);
# dataset/checkpoint/license/... 的资源性 finding 保持 README/URL 证据不动。
CODE_GROUNDED_CATEGORIES: set[ClaimCategory] = {
    ClaimCategory.CORE_METHOD,
    ClaimCategory.TRAINING,
    ClaimCategory.INFERENCE,
    ClaimCategory.EVALUATION,
    ClaimCategory.DEMO,
    ClaimCategory.BENCHMARK,
    ClaimCategory.API_INTERFACE,
    ClaimCategory.IMPLEMENTATION,
}

# 类别 → 检索查询补充词(statement 常只是类名/短语,补类别语义词提升 BM25 召回)
_CATEGORY_HINTS: dict[ClaimCategory, str] = {
    ClaimCategory.CORE_METHOD: "model network module layer forward",
    ClaimCategory.TRAINING: "train loss optimizer scheduler dataset epoch",
    ClaimCategory.INFERENCE: "inference predict load model forward",
    ClaimCategory.EVALUATION: "evaluate metric score benchmark",
    ClaimCategory.BENCHMARK: "benchmark experiment result",
    ClaimCategory.DEMO: "demo example visualization",
    ClaimCategory.API_INTERFACE: "api cli argument parser config interface",
    ClaimCategory.IMPLEMENTATION: "",
}


def category_search_query(statement: str, category: ClaimCategory) -> str:
    """Build the retrieval query for a finding: statement + category hints."""
    hint = _CATEGORY_HINTS.get(category, "")
    return f"{statement} {hint}".strip()
