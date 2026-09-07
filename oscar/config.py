"""Global configuration for OSCAR — 分层加载,所有可调默认值内联于此。

优先级(高 → 低):
  1. CLI(main.py 参数,直接改写本模块的单例 config)
  2. 环境变量 OSCAR_<段>_<字段>(如 OSCAR_RETRIEVAL_TOP_K)
  3. 根目录 config.yaml(随仓库提交、直接使用;不含密钥)
  4. 本文件的代码默认值

密钥例外:API key **只**从环境/根 .env 读取(绝不进入代码或已提交配置)。
本模块在 import 时自动加载根 .env(override=False → 已显式 export 的变量优先)。

不变量:本模块只依赖标准库 + python-dotenv + PyYAML,**永不 import
oscar 包**——它是全库最底层的被依赖方,任何反向依赖都会构成循环导入。
"""

from __future__ import annotations

import os
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path

import yaml
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"
_CONFIG_FILE = _PROJECT_ROOT / "config.yaml"
_ENV_PREFIX = "OSCAR_"

# 根 .env(存在时)灌入环境变量;override=False → 已显式 export 的变量优先。
# python-dotenv 对不存在的 dotenv_path 静默返回 False,无 .env 也安全。
load_dotenv(_ENV_FILE, override=False)


@dataclass
class LLMConfig:
    """LLM 调用配置(provider/endpoint/采样),不含任何密钥。

    密钥只经环境变量提供(映射表见 oscar/llm/client.py::_ENV_KEY_MAP):
      - "deepseek"(默认): DEEPSEEK_API_KEY
      - "openai":          OPENAI_API_KEY
      - "anthropic":       ANTHROPIC_API_KEY
      - "openai_compatible": OPENAI_COMPATIBLE_API_KEY
    在仓库根 .env(参照 .env.example,已 gitignore)或 shell 环境里设置。
    """
    provider: str = "deepseek"
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-v4-flash"
    temperature: float = 0.1
    max_tokens: int = 16384
    timeout: int = 120
    structured_output_method: str = "function_calling"


@dataclass
class AuditConfig:
    """Audit configuration."""
    cleanup_repo: bool = False  # 默认保留克隆仓库，方便复查
    max_clone_retries: int = 3
    clone_timeout: int = 120
    max_issues_to_fetch: int = 50
    max_prs_to_fetch: int = 20
    issue_search_keywords: list = field(default_factory=lambda: [
        "training", "train", "evaluation", "eval", "release",
        "not released", "code release", "dataset", "checkpoint",
        "missing", "implementation", "issue", "copyright",
        "proprietary", "license",
    ])


@dataclass
class RetrievalConfig:
    """代码检索与证据接地阈值(默认 = 原 evidence_grounder / code_vector_store
    硬编码常量现值,零行为漂移)。改动会影响送入 LLM 的代码窗口与裁决宽严
    ——确定性验收(双跑字节一致)请保持默认。
    """
    top_k: int = 12              # hybrid 检索送入 LLM 的候选代码块数
    max_chunks: int = 16         # hybrid + mapper 结构化召回合计上限
    max_chunk_chars: int = 3500  # 每块代码截断长度(完整类体常 >1.5k)
    max_locations: int = 6       # 报告 evidence details 的定位上限
    conf_cap: float = 0.85       # LLM 置信度上限(防过度自信)
    degrade_conf_cap: float = 0.5  # LLM 不可用降级时的置信度封顶
    rare_slots: int = 3          # 稀有内容词扫描专用槽位数
    mapper_min: int = 3          # mapper 候选文件块保底
    encode_batch: int = 32       # CodeBERT 编码批大小(CPU 内存护栏)


@dataclass
class CacheConfig:
    """缓存有效期(秒)。默认 7 天,与原硬编码常量一致。"""
    ttl_seconds: int = 86400 * 7         # LLM 响应 / 仓库克隆缓存 TTL
    index_ttl_seconds: int = 86400 * 7   # 代码向量索引 TTL(与仓库缓存同周期)


@dataclass
class PathConfig:
    """Path configuration."""
    work_dir: Path = Path(os.getcwd())
    clone_dir: Path = field(init=False)
    output_dir: Path = field(init=False)

    def __post_init__(self):
        self.clone_dir = self.work_dir / "clones"
        self.output_dir = self.work_dir / "output"


@dataclass
class Config:
    """Global application configuration."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    def ensure_dirs(self):
        """Ensure all required directories exist."""
        self.paths.clone_dir.mkdir(parents=True, exist_ok=True)
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 分层加载
# ---------------------------------------------------------------------------

def _coerce_env(section: str, name: str, default, raw: str):
    """按字段默认值类型把环境变量字符串转成布尔/整数/浮点(否则原样字符串)。"""
    env_name = f"{_ENV_PREFIX}{section}_{name}".upper()
    if isinstance(default, bool):  # 须先于 int 判断(bool 是 int 子类)
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(
                f"{env_name}={raw!r} is not a valid integer for {section}.{name}"
            ) from exc
    if isinstance(default, float):
        try:
            return float(raw)
        except ValueError as exc:
            raise ValueError(
                f"{env_name}={raw!r} is not a valid number for {section}.{name}"
            ) from exc
    return raw


def _apply_env_overrides(section: str, obj) -> None:
    """OSCAR_<SECTION>_<FIELD> 环境变量逐字段覆盖 yaml/默认值。

    default_factory 字段(如 issue_search_keywords 列表)没有合理的单值
    环境表示 → 仅 yaml 可配,跳过。
    """
    for f in fields(obj):
        if f.default is MISSING:
            continue
        env_name = f"{_ENV_PREFIX}{section}_{f.name}".upper()
        raw = os.environ.get(env_name)
        if raw is None:
            continue
        setattr(obj, f.name, _coerce_env(section, f.name, f.default, raw))


def _yaml_section(name: str, cls, data: dict):
    """把 config.yaml 顶层段落安全构造成配置对象;校验未知键与密钥误放。

    显式写 null 的字段回落代码默认值(yaml 里删键等价)。"""
    section = data.get(name)
    if section is None:
        return cls()
    if not isinstance(section, dict):
        raise ValueError(
            f"{_CONFIG_FILE.name}: [{name}] must be a mapping, "
            f"got {type(section).__name__}"
        )
    if name == "llm" and "api_key" in section:
        raise ValueError(
            f"{_CONFIG_FILE.name}: llm.api_key is not allowed. API keys live "
            "in the git-ignored root .env only (see .env.example) — never in "
            "committed files."
        )
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(str(k) for k in section if k not in allowed)
    if unknown:
        raise ValueError(
            f"{_CONFIG_FILE.name}: unknown keys under [{name}]: "
            + ", ".join(unknown)
        )
    return cls(**{k: v for k, v in section.items() if v is not None})


def load_config() -> Config:
    """默认值 →(存在时)config.yaml → 环境变量,逐层覆盖。

    CLI 覆盖发生得更晚(main.py 解析参数后直接改写单例 config)→
    CLI > env > yaml > 代码默认的完整优先级由此闭合。
    """
    raw: dict = {}
    if _CONFIG_FILE.exists():
        try:
            loaded = yaml.safe_load(_CONFIG_FILE.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"Bad YAML in {_CONFIG_FILE.name}: {exc}") from exc
        loaded = loaded or {}
        if not isinstance(loaded, dict):
            raise ValueError(
                f"{_CONFIG_FILE.name}: top level must be a mapping of sections"
            )
        raw = loaded

    cfg = Config(
        llm=_yaml_section("llm", LLMConfig, raw),
        audit=_yaml_section("audit", AuditConfig, raw),
        retrieval=_yaml_section("retrieval", RetrievalConfig, raw),
        cache=_yaml_section("cache", CacheConfig, raw),
    )
    for section, obj in (
        ("llm", cfg.llm),
        ("audit", cfg.audit),
        ("retrieval", cfg.retrieval),
        ("cache", cfg.cache),
    ):
        _apply_env_overrides(section, obj)
    return cfg


# Global singleton
config = load_config()
