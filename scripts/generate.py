import os

from openai import OpenAI

SYSTEM_PROMPT = """你是一个 Godot 引擎文档助手。请严格根据下面提供的文档片段回答问题。

规则：
1. 只使用提供的文档片段中的信息，不要依赖你自己的知识。
2. 如果文档片段中没有足够的信息回答问题，直接说“文档中没有找到相关内容”，不要编造。
3. 回答时用中文，但 API 名称、类名、方法名保留英文原文。
4. 每个关键结论后面用 [来源: header_path] 标注出处。
5. 如果涉及代码，给出完整的 GDScript 示例。"""

def build_prompt(query: str, chunks: list[dict]) -> list[dict]:
    # 把检索到的 chunk 拼成带编号的参考资料
    context_parts = []
    for i, c in enumerate(chunks, 1):
        source = " > ".join(c["header_path"]) if c["header_path"] else c["source"]
        context_parts.append(
            f"[片段 {i}] 来源: {source}\n"
            f"文件: {c['source']}\n"
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

def _get_client() -> OpenAI:
    """惰性构造客户端：import 本模块不需要 key，走到生成这一步才构造。

    统一使用 OpenAI 兼容的通用变量名（DeepSeek 本身就是 OpenAI 兼容端点）：
    - OPENAI_BASE_URL（必填）：服务地址，
      云端 DeepSeek 如 https://api.deepseek.com，
      本地部署如 http://localhost:11434/v1（Ollama）；
    - OPENAI_API_KEY（可选）：服务密钥；缺失时用占位值 "EMPTY" 放行
      （本地部署不校验 key；云端服务会对占位 key 返回 401，报错原样透出）；
    - OPENAI_MODEL（可选）：模型名，默认 deepseek-chat。
    """
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not base_url:
        raise RuntimeError(
            "未配置服务地址：请先设置环境变量 OPENAI_BASE_URL，"
            "例如云端 DeepSeek：'https://api.deepseek.com'，"
            "本地 Ollama：'http://localhost:11434/v1'"
        )
    return OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY",
        base_url=base_url,
    )


def generate_answer(query: str, chunks: list[dict]) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "deepseek-chat"),
        messages=build_prompt(query, chunks),
        temperature=0.1,   # 低温度，减少自由发挥
        max_tokens=1500,
    )
    return response.choices[0].message.content
