# OSCAR — 开源仓库完整性审计(Open-Source Completeness Audit & Review)

> 🎬 OSCAR
>
> Open-Source Completeness Audit & Review
>
> *"Every claim deserves evidence."*

> 输入一个 GitHub 仓库地址(可选附带论文),OSCAR 在**真实代码层面**核查:论文和
> README 声称的内容,仓库是否真的交付了——核心方法、训练/推理代码、数据集、
> checkpoint、benchmark、demo、API 与许可证。

[English README](README.md)

---

## 为什么需要它

GitHub 上的研究代码常常是**残缺**的:论文承诺了方法、数据集、benchmark,仓库里
却只有 demo 和 license,核心实现在别处、或根本不存在。

人工核查意味着逐条读声称、在克隆的代码里搜索、再判断"这到底是不是真的实现了"。
OSCAR 把这条审计流水线自动化:

- 抽取论文/README 对**自身贡献**的声称(claim);
- 用**本地混合检索**(CodeBERT 向量 + FAISS、BM25、关键词/AST 精确匹配)在真实
  仓库代码里召回;
- 让 LLM 把每条 claim 对着**检索到的代码块**裁决,verdict 锚定到具体文件与行号
  ——不是靠文件名猜;
- 对仍未落实的部分去 GitHub issues/PR 里调查(作者明确说"代码后续发布/不能公开"
  的,尊重其为 PLANNED/RESTRICTED,成本极低且权威);
- 输出 Markdown 报告 + 机器可读 JSON。

## OSCAR 不做的事

- **不做代码质量审查**——性能瓶颈、代码风格、安全性不在审计范围;只回答
  "开源完整性"这一个问题。
- **不检查项目能否运行**——不装依赖、不跑训练/评估、不复现任何指标。
- **不生成或补全代码**——产出是审计报告,不是代码。

## 特性

- **claim 级裁决**:每条 finding 最终为 `VERIFIED` / `INCOMPLETE` / `MISSING` /
  `UNCERTAIN`(issue 证据支持时为 `RESTRICTED` / `PLANNED`),带锚定代码位置清单
  与 LLM 对"这段代码实际做了什么"的自然语言说明。
- **6 个审计维度并行**(LangGraph fan-out):核心方法、训练、推理、API 面、
  资源(数据集/checkpoint/benchmark)、许可证。
- **检索不依赖 LLM**:搜索完全本地、确定;LLM 只裁决"真正被检索到的代码"。
- **不按名字下结论**:同一组件常常换名发布。检索用论文的功能语义句,
  裁决规则禁止"没有同名符号就报 MISSING"。
- **确定性、缓存、重跑几乎免费**:LLM 响应按精确消息哈希缓存;配置不变时重跑
  同一仓库的审计结果字节级一致,**零 API 调用**。
- **Provider 无关**:DeepSeek(默认)、OpenAI、Anthropic 或任意 OpenAI 兼容
  端点——密钥只经环境/.env 提供。

## 流水线

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

1. **Repository Loader** — 克隆仓库(缓存于 `.oscar_cache/repos/`),构建文件清单。
2. **Paper Resolver / Analyzer** — 下载论文(PDF/arXiv)并抽 claim:保守规则层 +
   LLM 层(LLM 用论文原句表达每个贡献的**功能语义**)。
3. **Repository Analyzer** — 文件指纹、代码分块(类/函数体带行号范围)、用
   CodeBERT 建本地混合索引。
4. **Planner → 6 个审计节点** — 每维度一个;每条 claim 经混合检索
   (keyword → BM25 → 向量,RRF 融合;稀有词扫描 + mapper 候选文件兜底,覆盖
   非同名的实现)映射到仓库代码。
5. **Issue Investigator** — 对 `MISSING`/`INCOMPLETE`/`UNCERTAIN` 的 finding
   搜索仓库 GitHub issues/PR;作者的显式陈述("代码将稍后发布"/"代码无法共享")
   落为 `PLANNED`/`RESTRICTED`。
6. **Evidence Grounder** — 核心步骤:对每条可代码验证的 finding,把组装好的候选
   代码块(受配置预算约束)交给 LLM,返回整体裁决 + 每块 "what this code does";
   位置一律锚定 chunk 记录,锚不上的裁决降级为 `UNCERTAIN`,不信任模型自报坐标。
7. **Report Generator** — 在 `output/<项目>/` 写出 `audit_report.md`、
   `audit_result.json`、`repository_manifest.json`。

## 裁决与类别

| 裁决 | 含义 |
|---|---|
| `VERIFIED` | 仓库代码实现了该 claim。 |
| `INCOMPLETE` | 声称的功能只实现了一部分。 |
| `MISSING` | 没有任何代码实现(检索无果时也按规则置为 MISSING)。 |
| `UNCERTAIN` | 检索/LLM 无法锚定——绝不静默乱猜。 |
| `RESTRICTED` / `PLANNED` | 作者声明代码不能共享 / 后续会发布。 |

Claim 类别:`core_method`、`training`、`inference`、`evaluation`、`demo`、
`benchmark`、`api_interface`、`dataset`、`checkpoint`、`license`、
`implementation`、`release`。

## 环境要求

- Python **3.9+**(在 3.9.12 上开发验证;打包元数据声明 3.11+)
- `git` 在 PATH
- 一个 LLM API key(默认 provider:DeepSeek)
- CodeBERT 模型约 2GB 磁盘,外加克隆仓库所需空间

## 安装

```bash
git clone <本仓库> && cd oscar
pip install -r requirements.txt

# 下载本地 embedding 模型(约 1.9GB,存到 bert/,已 gitignore):
python scripts/download_models.py     # 加 --also-graphcodebert 下载可选模型
```

在仓库根目录写 `.env`(已 gitignore;模板见 `.env.example`):

```bash
DEEPSEEK_API_KEY=sk-...
```

Provider 与密钥环境变量:`DEEPSEEK_API_KEY`(deepseek)、`OPENAI_API_KEY`
(openai)、`ANTHROPIC_API_KEY`(anthropic)、`OPENAI_COMPATIBLE_API_KEY`
(任意 OpenAI 兼容端点,base_url 走配置)。受限网络下载模型失败时,重试前设置
`HF_ENDPOINT=https://hf-mirror.com`。

## 用法

```bash
python main.py https://github.com/org/repo                          # 仅仓库
python main.py https://github.com/org/repo --paper 2401.12345       # + 论文(URL 或 arXiv ID)
python main.py https://github.com/org/repo --paper ... --output ./my_audit
```

| 参数 | 含义 |
|---|---|
| `repo_url` | 待审计的 GitHub 仓库。 |
| `--paper / -p` | 论文 URL 或裸 arXiv ID(可选)。 |
| `--output / -o` | 输出目录(默认 `./output`)。 |
| `--no-cleanup` | 审计后保留克隆仓库(覆盖 config.yaml)。 |
| `--verbose / -v` | 失败时打印完整 traceback。 |

`output/<项目>/` 下的产物:

- `audit_report.md` — 人读报告:执行摘要、claim 裁决表、各维度证据、
  issue 调查备注。
- `audit_result.json` — 机器读:findings 与证据明细(file/line/snippet/
  explanation)、分维度统计、加权总分。
- `repository_manifest.json` — 文件/类/分块快照,便于追溯。

## 配置

可调参数放在已提交的根目录 `config.yaml` —— 直接使用、就地修改(密钥绝不进
本文件,只放根 .env)。
**优先级:**

```
CLI 参数  >  环境变量 OSCAR_<段>_<字段>  >  config.yaml  >  代码内置默认值
```

| 段 | 控制内容 | 示例环境变量 |
|---|---|---|
| `llm` | provider、endpoint、model、采样 | `OSCAR_LLM_MODEL` |
| `audit` | 清理、克隆重试/超时、issue 检索上限与关键词 | `OSCAR_AUDIT_CLEANUP_REPO` |
| `retrieval` | 代码检索窗口与裁决阈值 | `OSCAR_RETRIEVAL_TOP_K` |
| `cache` | LLM/仓库缓存与向量索引的 TTL(秒) | `OSCAR_CACHE_TTL_SECONDS` |

```bash
OSCAR_RETRIEVAL_TOP_K=5 python main.py https://github.com/org/repo
```

列表型选项(如 `audit.issue_search_keywords`)只能走 yaml。**密钥绝不进代码或
config.yaml**——`llm.api_key` 出现在 yaml 里会直接报错;密钥只属于已
gitignore 的 `.env`(或 shell 显式 export)。

## 缓存与确定性

LLM 响应、论文 PDF/文本、仓库克隆、向量索引都在 `.oscar_cache/` 下
(已 gitignore):

```
.oscar_cache/
├── llm/        LLM 响应,键 = sha256(精确消息+model+采样参数)
├── papers/     按 arXiv ID 缓存的论文
├── repos/      提升后的克隆(按 URL,带有效性标记)
└── vectors/    每篇论文/每个仓库的 FAISS 索引 + chunk 元数据
```

- 缓存键是**精确请求字节**的哈希:prompt、model、temperature 任何一处改动只
  使对应条目失效(不会波及其他)。
- 热缓存下重跑审计**字节级一致且免费**:零 API 调用。确定性验收 = 双跑后对
  输出 `diff -r` 为空。
- LLM/仓库缓存与代码向量索引默认 TTL 7 天,`config.yaml → cache` 可调。
- 使用结构化输出时,provider 不支持 function calling 会自动退到手写 JSON 的
  降级路径;两条路径都缓存、都确定。

## 隐私与仓库卫生

- 密钥只在 `.env`(提交的模板是 `.env.example`)。
- 克隆仓库、LLM 响应、论文 PDF、向量索引、模型权重、运行输出、内部设计文档
  全部 git-ignore,留在本地。除了审计本身请求的 GitHub 元数据和你配置的 LLM
  API 调用,不会上传任何东西。
- 若密钥曾泄露进工作区,开源前**务必**到 provider 控制台轮换。

## 目录结构

```
main.py                      CLI 入口
oscar/
├── config.py                分层配置(默认值)——只依赖 stdlib+dotenv+yaml
├── prompts.py               全部 LLM 提示词的唯一持有处
├── llm/client.py            provider 工厂、缓存感知的 chat/结构化调用
├── paper/                   论文下载、文本抽取、claim 分析
├── audit/                   GitHub issues/PR 调查
├── mapping/                 代码向量库(CodeBERT+FAISS/BM25/keyword)
├── graph/workflow.py        LangGraph 流水线
├── report/                  markdown/JSON 报告生成
├── models/schemas.py        pydantic 状态与报告模型
└── utils/                   缓存(磁盘、utf-8 安全)、git、进度、杂项
scripts/
├── download_models.py       下载 bert/ 模型(幂等、--dry-run)
└── verify_prompts.py        提示词不变量 harness(内部 QA)
config.yaml                  提交的运行时配置(无密钥)
.env.example                 提交的密钥模板(实际填写进 .env)
```

## 常见问题

**要花钱吗?** 只有 LLM 调用花钱——首次审计每仓库几次;热缓存重跑免费。

**能完全离线吗?** 裁决需要 LLM(给了论文时 issue 调查也需要)。检索与索引本身
完全本地。

**LLM 报的坐标可信吗?** 不可信——位置一律锚定检索到的 chunk 记录;LLM 自报
位置解析不到就丢弃,锚不上的裁决降级为 `UNCERTAIN`。

**为什么还要下载模型?** 检索 embedding 用 CodeBERT 本地算,保证可复现且没有
边际 API 成本。`bert/` 按设计 git-ignore——每台机器跑一次
`scripts/download_models.py` 即可。

## 许可证

MIT —— 见 [LICENSE](LICENSE)。Copyright (c) 2026 The OSCAR Authors。
