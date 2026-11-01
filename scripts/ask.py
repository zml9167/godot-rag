from retrieve import retrieve
from generate import generate_answer

def ask(query: str, top_k: int = 6) -> dict:
    # 1. 检索
    chunks = retrieve(query, top_k=top_k)

    # 2. 如果检索结果为空或分数过低，直接拒答
    if not chunks:
        return {
            "answer": "文档中没有找到相关内容。",
            "sources": [],
        }

    # 3. 生成
    answer = generate_answer(query, chunks)

    # 4. 整理来源列表（供前端展示可点击链接）
    sources = []
    seen = set()
    for c in chunks:
        key = c["source_file"]
        if key not in seen:
            seen.add(key)
            sources.append({
                "file": c["source_file"],
                "section": " > ".join(c["section_path"]),
                "chunk_id": c["chunk_id"],
            })

    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "move_and_slide 的 delta 参数怎么传？"
    result = ask(q)
    print("回答:\n", result["answer"])
    print("\n来源:")
    for s in result["sources"]:
        print(f"  - {s['file']} > {s['section']}")