import json
from pathlib import Path
from collections import defaultdict

from .retrieve import retrieve
from .ask import ask
from .libs.utils import PROJECT_DIR, CHUNKS_PATH

QUESTIONS_PATH = PROJECT_DIR / "evaluate/questions.jsonl"
MISSES_PATH = PROJECT_DIR / "evaluate/misses.json"
TOP_K = 6


# ---------- 1. 校验 expected_source ----------
def validate_questions():
    sources = set()
    sections = set()
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            sources.add(c["source"])
            path = "/".join(c["header_path"]) if c["header_path"] else ""
            sections.add((c["source"], path))

    problems = []
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            if q["expected_source"] is None:
                continue
            if q["expected_source"] not in sources:
                problems.append((q["id"], "source 不存在", q["expected_source"]))
                continue
            if q.get("expected_section"):
                matched = any(
                    src == q["expected_source"]
                    and q["expected_section"].lower() in path.lower()
                    for src, path in sections
                )
                if not matched:
                    problems.append((q["id"], "section 未匹配", q["expected_section"]))

    if problems:
        print(f"⚠️  校验发现 {len(problems)} 个问题：")
        for pid, reason, value in problems:
            print(f"  {pid}: {reason} -> {value}")
        print("请先修正 questions.jsonl 再继续。\n")
    else:
        print("✅ 所有 expected_source / expected_section 校验通过\n")
    return problems


# ---------- 2. 检索评估（按语言分组） ----------
def eval_retrieval(questions, top_k=TOP_K):
    stats = defaultdict(lambda: {"total": 0, "hit": 0, "symbol_total": 0, "symbol_hit": 0})
    misses = []

    for q in questions:
        if q["type"] == "unanswerable":
            continue
        lang = q.get("lang", "unknown")
        s = stats[lang]
        s["total"] += 1

        chunks = retrieve(q["query"], top_k=top_k)
        sources = {c["source_file"] for c in chunks}

        hit = q["expected_source"] in sources if q["expected_source"] else False
        if hit:
            s["hit"] += 1
        else:
            misses.append({
                "id": q["id"], "lang": lang, "type": q["type"],
                "query": q["query"],
                "expected": q["expected_source"],
                "got": list(sources),
            })

        if q["type"] in ("api", "mixed") and q.get("expected_symbol"):
            s["symbol_total"] += 1
            if chunks and chunks[0].get("source") == "symbol" \
               and q["expected_symbol"].lower() in chunks[0]["content"].lower():
                s["symbol_hit"] += 1

    print("=== 检索评估 ===")
    for lang in sorted(stats):
        s = stats[lang]
        hit_rate = s["hit"] / s["total"] if s["total"] else 0
        print(f"[{lang}] Hit@{top_k}: {hit_rate:.1%} ({s['hit']}/{s['total']})")
        if s["symbol_total"]:
            sym_rate = s["symbol_hit"] / s["symbol_total"]
            print(f"[{lang}] Symbol Hit@1: {sym_rate:.1%} "
                  f"({s['symbol_hit']}/{s['symbol_total']})")
    print()
    return misses


# ---------- 3. 生成评估（按语言分组） ----------
def eval_generation(questions):
    stats = defaultdict(lambda: {"total": 0, "cited": 0, "api_total": 0, "api_kept": 0, "unans_total": 0, "unans_ok": 0})
    failures = []

    for q in questions:
        lang = q.get("lang", "unknown")
        s = stats[lang]
        s["total"] += 1

        result = ask(q["query"])
        answer = result["answer"]

        if "[来源:" in answer or "[来源：" in answer:
            s["cited"] += 1

        if q.get("expected_symbol"):
            s["api_total"] += 1
            if q["expected_symbol"] in answer:
                s["api_kept"] += 1

        if q["type"] == "unanswerable":
            s["unans_total"] += 1
            if "没有找到" in answer or "未找到" in answer \
               or "not found" in answer.lower() or "no relevant" in answer.lower():
                s["unans_ok"] += 1
            else:
                failures.append({
                    "id": q["id"], "lang": lang, "type": q["type"],
                    "query": q["query"],
                    "answer_preview": answer[:300],
                })

    print("=== 生成评估 ===")
    for lang in sorted(stats):
        s = stats[lang]
        print(f"[{lang}] 来源引用率: {s['cited']/s['total']:.1%} "
              f"({s['cited']}/{s['total']})")
        if s["api_total"]:
            print(f"[{lang}] API 名保留率: {s['api_kept']/s['api_total']:.1%}")
        if s["unans_total"]:
            print(f"[{lang}] 拒答准确率: {s['unans_ok']/s['unans_total']:.1%}")
    print()
    return failures


# ---------- 4. 主流程 ----------
if __name__ == "__main__":
    problems = validate_questions()
    if problems:
        raise SystemExit("先修正问题集再跑评估")

    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    retrieval_misses = eval_retrieval(questions)
    generation_failures = eval_generation(questions)

    Path(MISSES_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(MISSES_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "retrieval_misses": retrieval_misses,
            "generation_failures": generation_failures,
        }, f, ensure_ascii=False, indent=2)

    print(f"失败案例已写入 {MISSES_PATH}")
    print(f"  检索未命中: {len(retrieval_misses)} 条")
    print(f"  生成失败:   {len(generation_failures)} 条")