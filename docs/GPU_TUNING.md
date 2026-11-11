# 构建索引阶段的 GPU 显存调优记录

> **历史文档**：向量编码与重排序均已改为调用 SiliconFlow API（见 `scripts/libs/siliconflow.py`），
> `scripts/libs/index_builder/qdrant.py` 中的本地 GPU 编码与显存预算机制已随之移除，
> 下文提到的环境变量（`GODOT_RAG_VRAM_BUDGET` 等）已不存在。本文仅作为调优过程的历史记录保留。

> 记录一次针对 `scripts/libs/index_builder/qdrant.py` 向量构建流程的完整调优过程：
> 从"越跑越慢、显存增长直至 OOM"到"显存平稳、速度平稳、30 分钟全量构建 0 OOM"。

## 环境

| 项 | 值 |
|---|---|
| GPU | AMD Radeon RX 6750 GRE 12GB（ROCm 10.1 / Windows，torch 2.14.0+rocm10.1.0） |
| 模型 | BAAI/bge-m3（本地），`max_seq_length=4096` 硬截断 |
| 数据 | 14814 chunks，中位 378 token，p99 3076，>4096 有 61 条（编码时截断），另有 1 条 41.6 万字符巨型 chunk |
| 输出 | Qdrant 本地库 `godot_docs`，1024 维，归一化向量 |

## 核心模型：显存由什么决定

```
显存 ≈ 权重(固定 ~2.2GB) + 注意力项(∝ T×L，二次) + 线性激活项(∝ T，一次) + 碎片/内核ε
```

- **T**：一批的实际 token 总量（Σ 每条 token 数）
- **L**：一批内最长那条的 token 数（批内所有序列 pad 到它）
- 注意力张量真实大小 ≈ `16头 × 4B × n × L_max²`，而 `T × L_max ≈ n × L_max²`（批内长度相近时），所以用 `T×L` 作注意力的度量，量纲 token²

**关键认知：显存有两个相互独立的增长维度**——二次项（注意力，随批内最长长度平方增长）和一次项（激活，随批 token 总量线性增长）。任何"单旋钮"方案都罩不住两个维度。

## 演进史（每一版都是一次失败换来的教训）

### v0：固定条数 + 升序排序 → 单调涨直到 OOM

原始方案按固定条数分批，数据按长度升序排列。后果链条：

```
升序 → L 逐批增大；T 被固定条数钉死 → T×L 随 L 线性涨
→ 显存涨、每批耗时涨（计算量 ∝ T×L）→ 最终撞穿 OOM
```

实测日志形态：前段批 2.9s，后段批 11s，最后 OOM。另一个隐患：继承的
`GODOT_RAG_TOKEN_BUDGET=16384` 环境变量曾悄悄覆盖代码默认值。

### v2：推导式 token 上限 → 短批瞬间 OOM

想法很美：`token_cap = ATTN_BUDGET // max_seq_length`，数学上保证 `T×L ≤ ATTN_BUDGET` 恒成立。

死因：它只罩住二次项，把一次项放出了笼子。token 上限被推导到 131072/批，
首批是全数据集最短的 chunk，T 冲到 13 万 token——注意力没超，但线性激活
∝ T 直接几十 GB 级申请，启动第一秒就 OOM。而探针只测了"注意力最重的批"，
炸的是"token 最重的批"——**argmax 方向不对，梯子量的是没人管的维度**。

**教训：每个被使用的预算，都必须有一根属于它的探针。**

### v3：双闸门 + 双探针（现行架构的核心）

两道独立闸门，对应两个增长维度：

```python
# 1) 注意力预算 ATTN_BUDGET（二次项）：批满足 T × L ≤ 预算
# 2) token 预算 TOKEN_BUDGET（一次项）：批 token 总量 ≤ 该值
```

分批算法（`_take_batch`）：贪心装填，每装一条就重算两个值，任一超限截断当前批；
单条自身超限时至少保留一条（bypass），保证循环推进。

标定算法（`calibrate_budgets`）：梯子每档两个预算同时 ×2，每档用真实数据按本档
闸门把**全量数据**真实分一次批，然后从批列表里挑两个极值批各自实测：

- **注意力探针** = `max(批列表, key=n × max(L)²)` —— 二次维度的最坏批（升序数据下通常落在列表尾部附近的长批）
- **token 探针** = `max(批列表, key=Σtoken)` —— 一次维度的最坏批（通常落在列表头部附近、被 token 上限顶满的短批）
- 两个 argmax 命中同一批时只测一次

测好两个极端 ≈ 测好全部几百批（其余批的两个分量必夹在两极之间）。判决规则：
**任一探针 OOM → 整对配置作废，回退上一通过档（配比不拆分）**。

### v3.5：巨型 chunk 破坏 argmax → 梯子瘫痪

那条 41.6 万字符的巨型 chunk（70205 token）因 bypass 规则每档都以同样的
singleton 形状出现，其 metric（1×70205²）永远碾压一切，同时霸占两个探针 →
`obs == last_obs` → 梯子第二档就收手，标定形同虚设。

**修复：闸门/探针一律用"截断后的计费长度"** `min(n, max_seq_length)`——编码时
超长序列本来就被截到 4096，真实峰值显存由截断后的长度决定。一行修复后梯子恢复。

### v4：二分精标——×2 粒度太粗

实测梯子：第 4 档剩余 5.30 GiB（占用 56%），第 5 档直接贴火线（~100%）。
翻倍粒度让标定只能在两个极端里二选一。

**修复：越过峰值上限的档位，与上一通过档之间做二分精标**（`GODOT_RAG_REFINE_ROUNDS=7`，
解析精度 ~1.5%）。每个中点都是真实配置的真实探测。

### v5：仪表坏了——驱动层读数失真

最隐蔽的一坑。日志显示**每一档都报"剩余显存 11.98/11.98 GiB"**（连权重加载后
都不变），而同刻的 HIPCachingAllocator 警告显示驱动层实际 `free: 0`——
**`torch.cuda.mem_get_info()` 在这套 ROCm/Windows 上返回失真读数**。

后果链条：读数恒满 → 目标线永远打不到 → 梯子一路爬到真 OOM 才停 →
返回 OOM 前一档的激进配置 (67M, 131072) → 试编码 2285 条爬行 178 秒。

**修复：仪表从"司机视角"换到"车自身视角"**——不再读驱动层，改用 torch 分配器
自身记账：

- 度量：每个探针 encode 前 `reset_peak_memory_stats()`，encode 后取
  `max(max_memory_allocated(), memory_reserved())`（峰值分配量与含碎片的缓存驻留取大者，对驱动层足迹是保守代理）
- 总量：`get_device_properties(0).total_memory`（硬件属性，恒定可靠）
- 判决改为**事前封顶**：峰值 ≥ 上限 `(总显存−外部占用)×(1-VRAM_MARGIN)` 就收手二分——
  探针从此永远不会把驱动层打到 free:0，桌面卡顿和分配器濒死重试同步消失

### v6：碎片驻留——"越跑越慢"的真正元凶

修复仪表后试编码暴露：峰值读数 **13.57 GiB > 物理显存 11.98 GiB**，且 1501 条
短文本要 47 秒。物理解释只有一种：分配器持有的显存超过 VRAM，多余部分被
Windows WDDM **溢出到系统内存**，此后每个显存操作都要过 PCIe 来回搬数据。

机制链条：标定的十几轮探针用十几种批形状交替训练 → 固定式 segment 分配器记住
每种形状，`empty_cache` 只能释放空闲段，被碎片占着的段永远不还 → reserved 水位
只涨不降到超物理 → 全程溢出 → 慢。这与 OOM 是两个独立的病。

**修复：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**（在 torch 首次初始化
分配器之前于模块顶部设置）。展开式 segment 按小块映射/卸载显存，实测 2GB 分配
释放后 reserved 精确归零。修复后试编码：47s → 13.66s，峰值 13.57 → 10.12 GiB。

## 最终架构

```
标定（一次，约 4-10 分钟）
  floor (2M, 4096) 起步 → 每档 (attn, token) 同时 ×2
  → 每档真实分批 + 双 argmax 探针实测峰值
  → 峰值越上限 → 与上一通过档二分 7 轮 → 收敛
  停止条件：探针 OOM / 峰值越上限(二分后) / 探针形状无增益 / 双轴显式上限

运行
  每批现场贪心装填（双闸门）→ encode_batch OOM 半批重试
  → OOM 后双闸门各自 ×0.75 → 每 256 批 empty_cache 压平水位
```

层级关系：**标定管可预期，运行期自适应管不可预期**（如数据分布怪到中间批比极值
批更重时，argmax 启发式失效，运行期 OOM 半批重试兜底）。

## 标定结果与验收数据

标定值（本机，两次独立运行完全复现）：

```
注意力预算 = 28966912 (27.6M token²)    token 上限 = 56576/批    保留 20% (VRAM_MARGIN=0.20)
```

全量构建验收（14814 chunks，564 批，墙钟 30.3 分钟）：

```
OOM 记录 = 0
批次耗时：p50=2.72s  p90=5.85s  p95=7.50s  去冷启动后 max=8.62s
逐十分位无单调漂移（首批 16.46s 为内核编译冷启动，一次性）
```

## 环境变量速查

| 变量 | 默认 | 说明 |
|---|---|---|
| `GODOT_RAG_ATTN_BUDGET` | `auto` | 注意力预算（token²），显式数字则固定该轴 |
| `GODOT_RAG_TOKEN_BUDGET` | `auto` | token 上限/批，显式数字则固定该轴；两轴都固定则跳过标定 |
| `GODOT_RAG_VRAM_MARGIN` | `0.20` | 标定目标线：峰值上限 = 总显存 × (1−margin) |
| `GODOT_RAG_REFINE_ROUNDS` | `7` | 二分精标轮数 |
| `GODOT_RAG_EMPTY_CACHE_EVERY` | `256` | 每 N 批 empty_cache 一次，0 关闭 |
| `GODOT_RAG_UPSERT_BATCH` | `512` | 写入子进程攒批大小 |
| `GODOT_RAG_MAX_SEQ_LENGTH` | `4096` | 序列截断上限（在 utils.get_model，唯一限制点） |

已实测复现的标定值可直接固定（跳过标定约省 4 分钟）：

```powershell
$env:GODOT_RAG_ATTN_BUDGET = "28966912"
$env:GODOT_RAG_TOKEN_BUDGET = "56576"
python -m scripts.build_all_index
```

## 踩坑清单（可复用的经验）

1. **升序排序 + 静态批参数 = T×L 单调失控**。批参数必须随批内最长长度反向联动。
2. **显存有两个独立增长维度（二次注意力项 + 一次激活项），单旋钮罩不住**。
   每个被使用的预算都必须有属于它的探针。
3. **超长序列的计费长度必须用截断后的值**，否则一条巨型 chunk 会霸占所有探针
   的 argmax，让标定梯子整体失效。
4. **ROCm/Windows 的 `torch.cuda.mem_get_info()` 读数会失真**（free==total 假读数，
   与驱动层实际 free:0 并存）。显存度量要用 torch 分配器自身记账
   （`max_memory_allocated` / `memory_reserved`），总量用 `get_device_properties`。
5. **多种批形状交替训练会让固定式 segment 分配器产生无法释放的碎片驻留**，
   reserved 超过物理显存后 WDDM 溢出到系统内存，表现为"越跑越慢"。
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 从根上解决（须在 torch
   初始化前设置）。
6. **贴满载的探针 forward + empty_cache 驱动往返会导致桌面卡顿**（标定阶段固有
   成本，几分钟）；事前封顶式判决（峰值越线就停）可显著缓和。
7. **环境变量继承陷阱**：shell 里遗留的 `GODOT_RAG_*` 会悄悄覆盖默认值，启动前
   用 `Get-ChildItem Env:GODOT_*` 清点。
8. **PowerShell 会把 stderr 日志（loguru 缺省写 stderr）包装成错误记录**，
   造成"exit 1"假象；判断真实退出码用 `$LASTEXITCODE`。

## 附带修复（同一次全量构建暴露，非 GPU 调优本体）

- `validate.py`：删除 payload 逐字段比对（写入端只存 `chunk_id`，正文经 CHUNK_MAP
  从 chunks.jsonl 取），保留全部其余校验。
- `symbol.py`：符号索引 key 统一小写——检索端一律按小写 key 查询，此前混合大小写
  的 7813 个 key 是永远查不到的死条目，CamelCase 类名的符号直达从未生效。
