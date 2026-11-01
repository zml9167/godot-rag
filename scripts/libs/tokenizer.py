import re


def tokenize(text):
    # 保留 API 符号整体：字母数字下划线组成的 token 不拆，丢弃中文
    tokens = re.findall(r'[A-Za-z_][A-Za-z0-9_]*|\d+', text.lower())
    return tokens


if __name__ == "__main__":
    print(tokenize("tokenizer测试"))