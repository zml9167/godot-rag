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
│   ├── build_index.py         # 依次构建向量、BM25、符号索引
│   ├── index_builder/
│   │   ├── qdrant.py          # Qdrant 向量索引
│   │   ├── bm25.py            # BM25 索引
│   │   └── symbol.py          # API 符号索引
│   ├── retrieve.py            # 混合检索与 RRF 融合
│   ├── generate.py            # 调用大模型生成回答
│   ├── ask.py                 # 检索 + 生成的一体化入口
│   ├── evaluate.py            # 检索与生成评估
│   ├── ip.py                  # 获取当前 IP 所属国家
│   ├── tokenizer.py           # BM25 分词
│   └── paths.py               # 项目根目录定位
├── data/
│   ├── chunks.jsonl
│   ├── bm25_index/
│   ├── symbol_index/
│   └── qdrant_data/
├── models_cache/              # 嵌入模型缓存
└── logs/                      # 各阶段日志
```

`godot-docs-master/`、`data/`、`models_cache/`、`logs/` 和 `chunk_mismatch_reports/` 属于本地输入或生成数据，不纳入版本控制。

## 环境准备

建议在仓库根目录创建虚拟环境并安装依赖：

```bash
pip install -r requirements.txt
```

`requirements.txt` 不包含 PyTorch，请根据本机 CUDA 环境另行安装。建索引时会自动选择设备：有可用 GPU 时使用 `cuda`，否则回退到 `cpu`。

嵌入模型为 `BAAI/bge-m3`，首次使用时会按当前 IP 所属国家选择下载源：中国大陆（`CN`）通过 ModelScope 下载，其他地区通过 Hugging Face 下载，缓存目录为 `models_cache/`。

> 所有命令都需要在仓库根目录执行，并使用 `python -m scripts.<module>` 形式。脚本使用包内相对导入，直接 `python scripts/xxx.py` 会因相对导入失败。

## 构建索引

```bash
python -m scripts.run_pipeline
```

脚本会先校验各阶段脚本存在且非空，然后按顺序执行：

1. `scripts.rst2md`：若 `godot-docs-master/` 不存在，先浅克隆 `godotengine/godot-docs`；随后把 `DOCS_DIRS` 中的 RST 转换为同目录下的 `.md`。
2. `scripts.chunk_documents`：按结构切分 Markdown，写入 `data/chunks.jsonl`。
3. `scripts.build_index`：构建 Qdrant、BM25 和符号索引。

任一阶段失败会立即停止后续步骤。

注意：

- 首次运行才会克隆文档仓库，后续运行**不会**自动更新。需要更新文档时，请手动在 `godot-docs-master/` 中执行 `git pull`。
- 转换和切分目录由 `scripts/rst2md.py`、`scripts/chunk_documents.py` 中的 `DOCS_DIRS` 配置，默认包含 `about`、`classes`、`community`、`engine_details`、`getting_started`、`tutorials`。运行前请确认这些目录确实存在于 `godot-docs-master/` 下。
- 切分使用 `chunkana` 的 `structural` 策略，`max_chunk_size` 为 8192。每次运行都会重建 `data/chunks.jsonl`。
- 内容重建后若与原文不一致，会记录警告并把结果写入 `chunk_mismatch_reports/`，便于排查。

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

`scripts.build_index` 依次调用 `scripts/index_builder/` 下的三个模块：

- **Qdrant**：使用 `bge-m3` 生成 1024 维归一化向量，余弦距离，写入本地 `data/qdrant_data`，collection 名为 `godot_docs`。重建时会删除并重新创建该 collection，请确认其中没有需要保留的数据。chunk ID 会经 UUIDv5 哈希后作为 point ID。
- **BM25**：使用 `rank_bm25` 的 `BM25Okapi`，分词逻辑见 `scripts/tokenizer.py`，仅保留英文标识符和数字并转小写，与检索阶段保持一致。产物为 `data/bm25_index/` 下的 `bm25.pkl`、`chunk_ids.json` 和 `meta.json`。
- **符号索引**：从章节标题最后一级、反引号包裹的标识符和 CamelCase 名称中提取 API 符号，建立“符号 -> chunk 列表”的小写映射，产物为 `data/symbol_index/symbol_index.json`。

各阶段日志分别写入 `logs/`。

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

生成部分使用 OpenAI 兼容接口，配置位于 `scripts/generate.py`：`api_key` 目前是占位值，运行前需要替换；`base_url` 默认为 `https://api.deepseek.com`，模型默认为 `deepseek-chat`。提示词要求只依据检索片段回答、标注 `[来源: 章节路径]`，资料不足时明确说明。

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
- 脚本依赖 `scripts/paths.py` 中的 `PROJECT_DIR` 定位数据目录，因此仓库内部相对路径与目录结构需要保持一致。
- 文档内容版权归 Godot Engine 文档贡献者所有，使用时请遵循其许可证。
