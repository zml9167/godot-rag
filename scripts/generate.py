from openai import OpenAI

SYSTEM_PROMPT = """你是一个 Godot 引擎文档助手。请严格根据下面提供的文档片段回答问题。

规则：
1. 只使用提供的文档片段中的信息，不要依赖你自己的知识。
2. 如果文档片段中没有足够的信息回答问题，直接说“文档中没有找到相关内容”，不要编造。
3. 回答时用中文，但 API 名称、类名、方法名保留英文原文。
4. 每个关键结论后面用 [来源: section_path] 标注出处。
5. 如果涉及代码，给出完整的 GDScript 示例。"""

def build_prompt(query: str, chunks: list[dict]) -> list[dict]:
    # 把检索到的 chunk 拼成带编号的参考资料
    context_parts = []
    for i, c in enumerate(chunks, 1):
        source = " > ".join(c["section_path"]) if c["section_path"] else c["source_file"]
        context_parts.append(
            f"[片段 {i}] 来源: {source}\n"
            f"文件: {c['source_file']}\n"
            f"内容:\n{c['content']}\n"
        )
    context = "\n---\n".join(context_parts)

    user_message = f"""请根据以下文档片段回答问题。

## 文档片段
{context}

## 问题
{query}

## 回答要求
- 用中文回答，API 名保留英文
- 每个结论标注 [来源: 章节路径]
- 资料不足时明确说明"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

client = OpenAI(
    api_key="your-api-key",
    base_url="https://api.deepseek.com",  # 或其他兼容接口
)


def generate_answer(query: str, chunks: list[dict]) -> str:
    messages = build_prompt(query, chunks)
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.1,   # 低温度，减少自由发挥
        max_tokens=1500,
    )
    return response.choices[0].message.content
