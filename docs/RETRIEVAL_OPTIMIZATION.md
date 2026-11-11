# 检索优化记录：符号索引重构与评估修正

> 记录一次以 `python -m scripts.evaluate` 指标为驱动的检索优化：
> 检索 miss 从 8 条降到 4 条，Symbol Hit@1 从 75%/58%（en/zh）提升到双语言 100%。
> 背景前提：embedding 与 rerank 已切换 SiliconFlow API（BAAI/bge-m3 / BAAI/bge-reranker-v2-m3），
> chunk 为自适应切分（chunkana structural + adaptive，17867 条）。

## 基线（优化前）

| 指标 | 数值 |
|---|---|
| en Hit@9 | 95.0% (19/20) |
| en Symbol Hit@1 | 75.0% (9/12) |
| zh Hit@9 | 70.0% (14/20) |
| zh Symbol Hit@1 | 58.3% (7/12) |
| 检索 miss | 8 条 |

## 问题分诊（8 条 miss）

| 类别 | 条数 | 根因 |
|---|---|---|
| mixed 类名查询未命中类文档 | 2 | 符号索引缺类名符号 |
| signals vs callbacks 查询 | 2 | 评估标注错误（class_variant.md） |
| 跨语言语义查询 | 4 | BM25 对纯中文失效 + 跨语言语义匹配损耗 |

### 根因 1：符号索引缺类名符号

- 查询侧 `detect_symbols` 的 CamelCase 分支支持数字（`CharacterBody2D` 可提取），
  但索引侧 content pattern 的捕获组为 `([A-Za-z_]+)`，**不含数字**——
  `:ref:`CharacterBody2D<class_CharacterBody2D>`` 被截断成 `CharacterBody`
- 类名的天然来源 `header_path[0]`（类文档页首 header 即类名）在历史重构中丢失

### 根因 2：method 声明行格式覆盖不全

类文档的方法声明行存在多种前后缀形态：

```text
{{ void }} **queue_free**() {ref}`🔗<class_...>`            ← MyST 替换块前缀
{ref}`bool<class_bool>` **move_and_slide** ( ... )          ← 返回类型 ref 前缀 + 括号带空格/参数
{ref}`bool<class_bool>` **set_as_audio_listener_2d**(...)   ← setget 行
```

旧 pattern 只匹配「ref 前缀 + `**名字**()` 空括号紧跟」的 setget 形态，
导致 `queue_free`/`free`/`add_child` 等 9389 个真实方法声明缺席；
且旧捕获组在数字处截断，索引中残留 `set_as_audio_listener` 这类截断垃圾键。

### 根因 3：评估标注

`class_variant.md` 全文 0 次提及 signal/callback（实为 Variant 类型参考页），
真正覆盖该主题的是 `gdscript_basics.md`（"Signals are a Callback…" 章节），
且检索本就已返回该文档。

## 修复

### 1. 符号提取重构（symbol.py + pattern_utils.py）

- 类名：classes chunk 的 `header_path[0]`（过滤 `__preamble__` 等内部标记；
  **只从 classes chunk 提取**——教程等非类文档 chunk 的 header 是章节短语，
  查询侧无法以短语形态命中，属惰性键）
- 成员：类文档 content 的 method/property/constant/signals 声明；
  **enumerations（`:ref:` 类名链接文本）移出内容提取**——类名改由 header_path 提供，
  避免内容引用把海量 chunk 挂到核心类型符号上（曾导致 `bool` 挂 776 个 chunk）
- 反引号提取移除（曾把教程里的 `name`/`index`/`position` 等泛化词挂成 100+ chunk 大符号）
- 教程等非类文档 chunk 不产符号，检索覆盖交给 BM25/向量

### 2. 正则收紧与补数字（pattern_utils.py）

| pattern | 旧捕获组 | 新捕获组 |
|---|---|---|
| signals | `([A-Za-z_]+)` | `([a-z0-9_]+)` |
| enumerations | `([A-Za-z_]+)` | `([A-Z][A-Za-z0-9_]*)` |
| property | `([a-z_]+)` | `([a-z0-9_]+)` |
| constant | `([A-Z_]+)` | `([A-Z0-9_]+)` |
| method | `([A-Za-z_]+)` | `([a-z0-9_]+)` |

数据验证（全量 17867 条 diff）：数字放开后的新增捕获**全部**为含数字的合法名
（Area2D / Camera3D / ANISOTROPY_16X / ARRAY_CUSTOM0 / is_audio_listener_2d…），
不含数字的误增为 0。

### 3. method pattern 通用化

`**name**\s*\(` 不要求任何前缀（`{{ }}`、`{ref}`、🔗 链接等前后缀形态均覆盖），
方法符号 3911 → 13111（补入 `queue_free`、`free`、`add_child`、`_apply_impulse`…）。
顺带兼容文档中的转义下划线写法（`**\_xxx**` → 捕获为 `_xxx`，Godot 虚方法命名）。

### 4. 评估标注修正（questions.jsonl）

`sem_zh_006` / `sem_en_006` 的 `expected_source`：
`classes/class_variant.md` → `tutorials/scripting/gdscript/gdscript_basics.md`。

## 决策记录（探索过并否决/暂缓的方案）

| 方案 | 结论 |
|---|---|
| 类名来源：header_path vs content | 先否决 header_path（"类名不从 header_path 取"），后经仿真验证（符号池平均 -58%、mixed 查询类文档排位 1）改为采用；最终形态：只从 classes chunk 取 `header_path[0]` |
| 反引号提取（教程标识符入索引） | 把教程泛化词挂成大符号（name 173 / index 165），且类名已由 header_path 覆盖 → 移除 |
| 按覆盖文档数频率剪枝符号 | 覆盖文档数 299→30 平滑无悬崖，且剪掉 Dictionary 等合法类查询符号 → 否决 |
| 符号 rerank 漏斗（SYMBOL_RERANK_LIMIT） | 垃圾符号清除前会被垃圾占满；清除后最大池为 ProjectSettings 375（合法单页巨文档）→ 暂缓，作成本优化备选 |
| zh 查询术语扩展（glossary） | 已验证有效（见下），用户决定暂不实施 |
| 枚举成员符号 | constant pattern 已捕获（如 EXPAND_KEEP_SIZE），property 模式保持小写避免重复 |

## 结果

| 指标 | 基线 | 修复后 |
|---|---|---|
| en Hit@9 | 95.0% (19/20) | 95.0% (19/20) |
| en Symbol Hit@1 | 75.0% (9/12) | **100% (12/12)** |
| zh Hit@9 | 70.0% (14/20) | **85.0% (17/20)** |
| zh Symbol Hit@1 | 58.3% (7/12) | **100% (12/12)** |
| 检索 miss | 8 条 | **4 条** |

符号索引规模：22453 → 18843 符号；评估查询符号池平均 17.1 → 10.9、最大 188 → 101。
修复分两步生效：符号重构后 Symbol Hit@1 91.7%（`queue_free` 缺席），
method pattern 通用化后 100%。

## 剩余 4 条 miss（未处理）

均为语义/跨语言查询，返回来源语义相关但非单点标注文档：

| 查询 | 期望 | 实际返回（节选） |
|---|---|---|
| zh 角色老穿模、卡进墙里 | using_character_body_2d.md | troubleshooting_physics_issues.md 等 |
| zh 输入映射怎么配置 | input_examples.md | class_input.md 等 |
| en How do I configure input mappings | input_examples.md | controllers_gamepads_joysticks.md、inputevent.md 等 |
| zh 物理层和碰撞掩码 | physics_introduction.md | class_collisionobject2d.md 等 |

## 已验证未实施的优化方向

### zh 查询术语扩展（glossary）

实测：分词器只保留英文标识符，纯中文查询的 BM25 tokens 为**空**（BM25 路失效，
融合实际只剩 title/content 双向量）。注入英文 Godot 术语后，期望文档 BM25 排名：

| zh 查询 | 注入术语 | 期望文档 BM25 排名 |
|---|---|---|
| 输入映射怎么配置 | InputMap input mapping configure | None → 5 |
| 物理层和碰撞掩码 | physics layer collision layer collision mask | None → 11 |
| 角色穿模卡进墙 | collision clipping stuck in wall | None → 1 |

三条全部进入召回池。实现要点：zh→en 术语词典（独立文件，便于维护）+
`retrieve()` 入口检测 CJK 后拼接；**只注入 BM25 路**（向量保持原中文，避免稀释
bge-m3 跨语言语义）。预期 zh Hit@9 85% → 95%+。

### 符号 rerank 漏斗

剩余最大符号池 ProjectSettings 375 chunks（合法单页巨文档）——查询明确提到
ProjectSettings 时一次 rerank 375 条成本偏高。备选：类名查询跳过 rerank 直接按
`get_symbol_order` 粗排取 top_k（类名查询的粗排信号已验证足够强：排位 1），
或 `SYMBOL_RERANK_LIMIT` 兜底截断。

## 附：同期的构建链路健壮性修复

与检索质量无直接关系，但同属本轮迭代：

- embeddings API 400（code 20015）：超长 chunk 超 bge-m3 单条 8192 token 上限
  → `siliconflow.py` 发送前用 bge-m3 同款 tokenizer 按 token 精确截断
  （`GODOT_RAG_EMBEDDING_MAX_TOKENS`，tokenizer.json 首次自动下载/复用本地缓存）
- 分批三闸门：条数 ≤32（官方硬上限）+ 批内总字符 + 批内总 token
- 空白 chunk：切分源头跳过 + validate 报错 + 客户端占位兜底
- rerank/embeddings API 错误分类：429/503/504 指数退避重试，400/401/403/404 直接中止

## 验证命令

```bash
.venv/bin/python -m scripts.libs.index_builder.symbol
python -m scripts.evaluate
```
