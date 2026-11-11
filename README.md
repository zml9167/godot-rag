# Godot 文档 RAG

[中文](README.md) | [English](README.en.md)

基于 Godot 官方文档构建本地检索索引的实验项目。流程为：将 RST 文档转换为 MyST Markdown，按结构切分为 chunks，构建 BM25、API 符号和 Qdrant 向量索引，再通过符号直达 + 向量 + BM25 的混合检索为大模型提供上下文，并附带一套检索/生成评估脚本。

## 项目结构

```text
.
├── godot-docs-master/            # 从 godotengine/godot-docs 克隆的文档
├── docs/                         # 过程记录文档
│   ├── GPU_TUNING.md             # 本地 GPU 编码时代的显存调优记录（历史文档）
│   └── RETRIEVAL_OPTIMIZATION.md # 检索优化记录
├── evaluate/
│   ├── questions.jsonl           # 评估问题集
│   └── misses.json               # 评估输出的失败案例（运行时生成）
├── scripts/
│   ├── run_pipeline.py           # 一键执行转换、切分、建索引
│   ├── rst2md.py                 # RST -> MyST Markdown
│   ├── chunk_documents.py        # Markdown -> data/chunks.jsonl
│   ├── build_all_index.py        # 并行构建向量、BM25、符号索引，随后校验
│   ├── retrieve.py               # 符号 + 双路向量 + BM25 混合检索与重排
│   ├── generate.py               # 调用大模型生成回答
│   ├── ask.py                    # 检索 + 生成的一体化入口
│   ├── evaluate.py               # 检索与生成评估
│   └── libs/
│       ├── utils.py              # 路径常量与 API 客户端单例
│       ├── siliconflow.py        # SiliconFlow embeddings/rerank API 客户端
│       ├── tokenizer.py          # BM25 分词
│       ├── pattern_utils.py      # API 符号提取正则
│       └── index_builder/
│           ├── qdrant.py         # Qdrant 向量索引
│           ├── bm25.py           # BM25 索引
│           ├── symbol.py         # API 符号索引
│           └── validate.py       # 索引完整性校验
├── data/
│   ├── chunks.jsonl
│   ├── bm25_index/
│   ├── symbol_index/
│   └── qdrant_data/
├── models_cache/                 # tokenizer.json 等本地缓存（首次使用时自动下载）
└── logs/                         # 各阶段日志
```

`godot-docs-master/`、`data/`、`logs/` 和 `models_cache/` 属于本地输入或生成数据，不纳入版本控制。

## 环境准备

建议在仓库根目录创建虚拟环境并安装依赖：

```bash
pip install -r requirements.txt
```

`requirements.txt` 不包含 PyTorch：嵌入与重排序均已改为调用 SiliconFlow API，本项目不再依赖 GPU、PyTorch 或本地模型权重。

**嵌入与重排序依赖 SiliconFlow API，需要先导出密钥：**

```bash
export SILICONFLOW_API_KEY="你的密钥"
```

密钥缺失时，建索引或检索会在初始化客户端时报错。相关环境变量（均可省略，括号内为默认值）：嵌入模型名 `GODOT_RAG_EMBEDDING_MODEL`（`BAAI/bge-m3`）、重排序模型名 `GODOT_RAG_RERANKER_MODEL`（`BAAI/bge-reranker-v2-m3`）、服务地址 `SILICONFLOW_API_BASE`（`https://api.siliconflow.cn/v1`）、嵌入单请求条数 `GODOT_RAG_API_BATCH_SIZE`（`32`，API 硬上限）、单请求 token 预算 `GODOT_RAG_API_BATCH_TOKENS`（`24576`）、单条文本 token 上限 `GODOT_RAG_EMBEDDING_MAX_TOKENS`（`8176`，bge-m3 单条上限 8192 token；超限 chunk 会用 bge-m3 同款 tokenizer 按 token 精确截断后编码并在日志告警，tokenizer.json 首次使用时自动从 ModelScope/HF 下载到 `models_cache/tokenizers/`（约 9MB，本机已有 bge-m3 模型缓存时直接复用，不重复下载）、tokenizer 文件路径 `GODOT_RAG_TOKENIZER_FILE`（`models_cache/tokenizers/bge-m3/tokenizer.json`）、重排单请求文档数 `GODOT_RAG_RERANK_BATCH_SIZE`（`64`）、超时 `GODOT_RAG_API_TIMEOUT`（`60` 秒）、最大重试次数 `GODOT_RAG_API_MAX_RETRIES`（`8`）、重试初始等待 `GODOT_RAG_API_RETRY_BASE_DELAY`（`2.0` 秒）、重试等待上限 `GODOT_RAG_API_RETRY_MAX_DELAY`（`60.0` 秒）、写入端攒批条数 `GODOT_RAG_UPSERT_BATCH`（`512`）。429（限流）/503（过载）/504（超时）按指数退避自动重试；400/401/403/404 属于配置类错误，会直接报错中止。

> 所有命令都需要在仓库根目录执行，并使用 `python -m scripts.<module>` 形式。脚本使用包内相对导入，直接 `python scripts/xxx.py` 会因相对导入失败。

## 构建索引

```bash
python -m scripts.run_pipeline
```

脚本会先校验各阶段脚本存在且非空，然后按顺序执行：

1. `scripts.rst2md`：若 `godot-docs-master/` 不存在，先浅克隆 `godotengine/godot-docs`；随后把 `DOCS_DIRS` 中的 RST 转换为同目录下的 `.md`。
2. `scripts.chunk_documents`：按结构切分 Markdown，写入 `data/chunks.jsonl`。
3. `scripts.build_all_index`：以三个子进程并行构建 Qdrant、BM25 和符号索引，全部成功后自动运行索引完整性校验（`scripts.libs.index_builder.validate`）。

任一阶段失败会立即停止后续步骤。

注意：

- 首次运行才会克隆文档仓库，后续运行**不会**自动更新。需要更新文档时，请手动在 `godot-docs-master/` 中执行 `git pull`。
- 转换和切分目录由 `scripts/libs/utils.py` 中的 `DOCS_DIRS` 配置（`rst2md.py` 与 `chunk_documents.py` 共用），默认包含 `about`、`classes`、`community`、`engine_details`、`getting_started`、`tutorials`。运行前请确认这些目录确实存在于 `godot-docs-master/` 下。
- 切分使用 `chunkana` 的 `structural` 策略并开启自适应大小（`AdaptiveSizeConfig(base_size=2048)`）。每次运行都会重建 `data/chunks.jsonl`，空白 chunk 会被跳过不写入。
- 内容重建后若与原文不一致，会按文档记录警告并区分"内容丢失"与"仅空行差异"（空行差异只计数不刷屏），运行结束汇总；不再写入报告文件。

每个 chunk 是 `data/chunks.jsonl` 中的一行 JSON，包含：

| 字段 | 说明 |
| --- | --- |
| `chunk_id` | `source:start_line-end_line` 形式的唯一标识 |
| `source` | 相对 `godot-docs-master/` 的文档路径 |
| `header_path` | 章节标题路径 |
| `content` | chunk 正文 |
| `start_line` / `end_line` | 在源文件中的行号范围 |
| `metadata` | 切分策略、内容类型、是否含代码等元数据 |
| `chunk_type` | 文档顶层目录名 |

## 索引说明

`scripts.build_all_index` 以三个子进程并行构建 `scripts/index_builder/` 下的三个模块（三者互不依赖，全部成功才进入校验）：

- **Qdrant**：通过 SiliconFlow API 使用 `bge-m3` 生成 1024 维归一化向量，余弦距离，写入本地 `data/qdrant_data`，collection 名为 `godot_docs`。每个 point 含 `title` 和 `content` 两个命名向量：`title` 由 `header_path` 用 ` > ` 拼接后编码，`content` 为 chunk 正文编码；`header_path` 缺失或非法的 chunk 只写入 `content` 向量。chunk ID 会经 UUIDv5 哈希后作为 point ID。重建时会物理清空整个 `data/qdrant_data` 存储目录后重新创建（qdrant-client 本地模式的 `delete_collection` 在部分版本下不会真正清空旧 point），写完会对 collection 内 point 总数与本次写入数做对账校验；请确认其中没有需要保留的数据。
- **BM25**：使用 `rank_bm25` 的 `BM25Okapi`，分词逻辑见 `scripts/tokenizer.py`，仅保留英文标识符和数字并转小写，与检索阶段保持一致。产物为 `data/bm25_index/` 下的 `bm25.pkl`、`chunk_ids.json` 和 `meta.json`。
- **符号索引**：只从 `classes` chunk 提取 API 符号——类名取 `header_path[0]`，方法/属性/常量/信号成员声明取自正文（正则见 `scripts/libs/pattern_utils.py`，捕获组包含数字，可覆盖 `CharacterBody2D` 这类类名）；教程等非类文档 chunk 不产符号，检索覆盖交给向量与 BM25。产物为 `data/symbol_index/` 下的 `symbol_index.json`（原样符号映射）和 `symbol_lower_index.json`（小写符号映射）。

各阶段日志分别写入 `logs/`。

## 检索与问答

`scripts/retrieve.py` 的检索流程：

1. **符号检索**：用正则从查询中识别 API 符号（先查原样符号索引，再查小写索引并过滤停用词），命中的 chunk 按优先级排序（章节标题/来源路径/正文是否命中该符号，`classes` 优先于 `tutorials`），再经重排序模型打分，取前 `symbol_top_k`（默认 3）个 chunk。
2. **混合召回**：`title` 与 `content` 两路向量检索各取前 30 条，BM25 取前 30 条，三路结果用 RRF（`k=60`）融合。
3. **重排序**：融合结果经重排序模型打分并重新排序，取前 `fuse_top_k`（默认 6）个 chunk。
4. **合并**：符号命中置顶，与混合结果按 `chunk_id` 去重后返回，合计最多 3 + 6 = 9 个 chunk；每个结果带 `retrieve_method`（`symbol` / `hybrid`）标记。

`scripts/ask.py` 在检索结果为空时直接拒答，否则把 chunk 拼接为带来源的上下文，调用大模型生成回答，并返回去重后的来源列表。

命令行示例：

```bash
python -m scripts.ask "move_and_slide 的 delta 参数怎么传？"
```

生成部分使用 OpenAI 兼容接口，通过环境变量配置：`OPENAI_BASE_URL`（**必填**，服务地址；云端 DeepSeek 为 `https://api.deepseek.com`，本地 Ollama 为 `http://localhost:11434/v1`）、`OPENAI_API_KEY`（服务密钥；缺失时以占位值放行——本地部署不校验，云端会返回 401）、`OPENAI_MODEL`（模型名，**必填**）。生成使用低温度（`temperature=0.1`）与 `max_tokens=1500` 以减少自由发挥。提示词要求只依据检索片段回答、标注 `[来源: 章节路径]`，资料不足时明确说明。

## 评估

```bash
python -m scripts.evaluate
```

评估脚本会先校验 `evaluate/questions.jsonl`：每行必须是合法 JSON 对象且包含 `id`、`query`、`type`、`expected_source` 字段，`expected_source` / `expected_section` 需真实存在于 chunks 中，校验失败则终止。随后输出：

- **检索评估**：按语言统计 `Hit@9`（`symbol_top_k + fuse_top_k`，期望来源是否出现在返回结果中）和 `Symbol Hit@1`（API 类问题的首个结果是否来自符号检索）。
- **生成评估**：按语言统计来源引用率、API 名称保留率和不可回答问题的拒答准确率。

失败案例写入 `evaluate/misses.json`，分为 `retrieval_misses` 与 `generation_failures` 两部分。问题集每行为一个 JSON 对象，字段包括 `id`、`lang`、`query`、`type`（如 `api`、`mixed`、`unanswerable`）、`expected_symbol`、`expected_source` 和 `expected_section`。

## 说明

- 本项目是本地实验项目，未提供 Web 服务或前端，检索与问答均通过 Python 模块调用。
- 脚本依赖 `scripts/libs/utils.py` 中的 `PROJECT_DIR` 定位数据目录，因此仓库内部相对路径与目录结构需要保持一致。
- 检索优化的决策与评估数据见 `docs/RETRIEVAL_OPTIMIZATION.md`；`docs/GPU_TUNING.md` 是本地 GPU 编码时代的显存调优历史记录（现已改为 API 编码，文中环境变量已失效）。
- 文档内容版权归 Godot Engine 文档贡献者所有，使用时请遵循其许可证。
- 本项目采用 [MIT 许可证](LICENSE) 许可。
