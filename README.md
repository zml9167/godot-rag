# Godot 文档 RAG

[中文](README.md) | [English](README.en.md)

基于 Godot 官方文档构建本地检索索引的实验项目。流程为：将 RST 文档转换为 MyST Markdown，按结构切分为 chunks，构建 BM25、API 符号和 Qdrant 向量索引，再通过符号直达 + 向量 + BM25 的混合检索为大模型提供上下文，并附带一套检索/生成评估脚本。

## 项目结构

```text
.
├── godot-docs-master/         # 从 godotengine/godot-docs 克隆的文档
├── evaluate/
│   └── questions.jsonl        # 评估问题集
├── scripts/
│   ├── run_pipeline.py        # 一键执行转换、切分、建索引
│   ├── rst2md.py              # RST -> MyST Markdown
│   ├── chunk_documents.py     # Markdown -> data/chunks.jsonl
│   ├── build_all_index.py     # 并行构建向量、BM25、符号索引，随后校验
│   ├── retrieve.py            # 混合检索与 RRF 融合
│   ├── generate.py            # 调用大模型生成回答
│   ├── ask.py                 # 检索 + 生成的一体化入口
│   ├── evaluate.py            # 检索与生成评估
│   └── libs/
│       ├── utils.py           # 设备检测、模型加载、路径常量
│       ├── tokenizer.py       # BM25 分词
│       ├── pattern_utils.py   # API 符号提取正则
│       └── index_builder/
│           ├── qdrant.py      # Qdrant 向量索引
│           ├── bm25.py        # BM25 索引
│           ├── symbol.py      # API 符号索引
│           └── validate.py    # 索引完整性校验
├── data/
│   ├── chunks.jsonl
│   ├── bm25_index/
│   ├── symbol_index/
│   └── qdrant_data/
├── models_cache/              # 嵌入模型缓存
└── logs/                      # 各阶段日志
```

`godot-docs-master/`、`data/`、`models_cache/` 和 `logs/` 属于本地输入或生成数据，不纳入版本控制。

## 环境准备

建议在仓库根目录创建虚拟环境并安装依赖：

```bash
pip install -r requirements.txt
```

`requirements.txt` 不包含 PyTorch，请根据本机 CUDA 环境另行安装。建索引时默认只使用 GPU（`cuda`），并在加载模型后、生成向量前打印当前设备；GPU 不可用时直接报错，不会静默回退到 CPU。启动时会先在子进程中运行一次 GPU 自检；如果 ROCm 版 PyTorch 没有为当前 GPU 架构（例如 `gfx1031`）编译内核，会自动设置兼容的 `HSA_OVERRIDE_GFX_VERSION`。如果确实需要 CPU，必须显式设置 `GODOT_RAG_ALLOW_CPU=1`。长序列截断长度可通过 `GODOT_RAG_MAX_SEQ_LENGTH` 调整（默认 4096）。编码批大小由构建启动时的自动标定决定：两个预算闸门——注意力预算 `GODOT_RAG_ATTN_BUDGET` 与单批 token 预算 `GODOT_RAG_TOKEN_BUDGET`——默认 `auto`，启动时从保守值逐档翻倍真实编码试水、按显存峰值二分逼近安全线；显存保留比例 `GODOT_RAG_VRAM_MARGIN`（默认 0.20）、缓存释放间隔 `GODOT_RAG_EMPTY_CACHE_EVERY`（默认 256 批）、二分精标轮数 `GODOT_RAG_REFINE_ROUNDS`（默认 7）、写入端攒批条数 `GODOT_RAG_UPSERT_BATCH`（默认 512）均可覆盖。

嵌入模型为 `BAAI/bge-m3`，首次使用时会按当前 IP 所属国家选择下载源：中国大陆（`CN`）通过 ModelScope 下载，其他地区通过 Hugging Face 下载，缓存目录为 `models_cache/`。

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
- 转换和切分目录由 `scripts/rst2md.py`、`scripts/chunk_documents.py` 中的 `DOCS_DIRS` 配置，默认包含 `about`、`classes`、`community`、`engine_details`、`getting_started`、`tutorials`。运行前请确认这些目录确实存在于 `godot-docs-master/` 下。
- 切分使用 `chunkana` 的 `structural` 策略，`max_chunk_size` 为 8192。每次运行都会重建 `data/chunks.jsonl`。
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

- **Qdrant**：使用 `bge-m3` 生成 1024 维归一化向量，余弦距离，写入本地 `data/qdrant_data`，collection 名为 `godot_docs`。chunk ID 会经 UUIDv5 哈希后作为 point ID。重建时会物理清空整个 `data/qdrant_data` 存储目录后重新创建（qdrant-client 本地模式的 `delete_collection` 在部分版本下不会真正清空旧 point），写完会对 collection 内 point 总数与本次写入数做对账校验；请确认其中没有需要保留的数据。
- **BM25**：使用 `rank_bm25` 的 `BM25Okapi`，分词逻辑见 `scripts/tokenizer.py`，仅保留英文标识符和数字并转小写，与检索阶段保持一致。产物为 `data/bm25_index/` 下的 `bm25.pkl`、`chunk_ids.json` 和 `meta.json`。
- **符号索引**：从章节标题最后一级、反引号包裹的标识符和 CamelCase 名称中提取 API 符号，建立“符号 -> chunk 列表”的小写映射，产物为 `data/symbol_index/symbol_index.json`。

各阶段日志分别写入 `logs/`。调试批量标定可单独运行 `python -m scripts.libs.index_builder.qdrant --validate-calibration`：阶段一校验分批算法的不变量（无需 GPU、秒级完成），阶段二真机标定并试编码一批报告显存峰值；无 CUDA 环境会自动跳过显存探针。

## 检索与问答

`scripts/retrieve.py` 的检索流程：

1. 从查询中识别符号并查符号索引，每个符号最多取 3 个 chunk。
2. 向量检索取前 30 条，BM25 检索取前 30 条。
3. 两路结果用 RRF（`k=60`）融合，符号命中置顶，最终返回 `top_k`（默认 6）个 chunk。

`scripts/ask.py` 在检索结果为空时直接拒答，否则把 chunk 拼接为带来源的上下文，调用大模型生成回答，并返回去重后的来源列表。

命令行示例：

```bash
python -m scripts.ask "move_and_slide 的 delta 参数怎么传？"
```

生成部分使用 OpenAI 兼容接口，通过环境变量配置：`OPENAI_BASE_URL`（**必填**，服务地址；云端 DeepSeek 为 `https://api.deepseek.com`，本地 Ollama 为 `http://localhost:11434/v1`）、`OPENAI_API_KEY`（服务密钥；缺失时以占位值放行——本地部署不校验，云端会返回 401）、`OPENAI_MODEL`（模型名，默认 `deepseek-chat`）。`OPENAI_BASE_URL` 未设置时会在生成阶段直接报错提示。提示词要求只依据检索片段回答、标注 `[来源: 章节路径]`，资料不足时明确说明。

## 评估

```bash
python -m scripts.evaluate
```

评估脚本会先校验 `evaluate/questions.jsonl` 中的 `expected_source` 和 `expected_section` 是否真实存在，校验失败则终止。随后输出：

- **检索评估**：按语言统计 `Hit@6`（期望来源是否出现在返回结果中）和 `Symbol Hit@1`（API 类问题的首个结果是否为对应符号）。
- **生成评估**：按语言统计来源引用率、API 名称保留率和不可回答问题的拒答准确率。

失败案例写入 `evaluate/misses.json`。问题集每行为一个 JSON 对象，字段包括 `id`、`lang`、`query`、`type`（如 `api`、`mixed`、`unanswerable`）、`expected_symbol`、`expected_source` 和 `expected_section`。

## 说明

- 本项目是本地实验项目，未提供 Web 服务或前端，检索与问答均通过 Python 模块调用。
- 脚本依赖 `scripts/libs/utils.py` 中的 `PROJECT_DIR` 定位数据目录，因此仓库内部相对路径与目录结构需要保持一致。
- 文档内容版权归 Godot Engine 文档贡献者所有，使用时请遵循其许可证。
- 本项目采用 [MIT 许可证](LICENSE) 许可。
