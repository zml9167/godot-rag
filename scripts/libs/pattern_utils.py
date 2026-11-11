import re


def pattern_of_signals(pattern: str = r'([a-z0-9_]+)') -> str:
    return r'\*\*' + pattern + r'\*\*\([^:]+:\s+\{ref\}`[^<]+<[^>]+>`\)'


def pattern_of_enumerations(pattern: str = r'([A-Z][A-Za-z0-9_]*)') -> str:
    # 大写开头、无下划线：严格匹配 Godot 类名形态（Node2D/ExpandMode/GLTFCamera）
    return r':ref:`' + pattern + r'<[A-Za-z0-9_]+>`'


def pattern_of_property(pattern: str = r'([a-z0-9_]+)') -> str:
    return r'\{ref\}`[A-Za-z0-9_]+<[A-Za-z0-9_]+>`\s+\*\*' + pattern + r'\*\*'


def pattern_of_constant(pattern: str = r'([A-Z0-9_]+)') -> str:
    return r'\*\*' + pattern + r'\*\*\s+=\s+'


def pattern_of_method(pattern: str = r'([a-z0-9_]+)') -> str:
    # 通用方法声明：**name**( / **name** ( ... )，不要求任何前缀
    # （classref-method 声明行有 {{ }} / {ref}`类型<类>` / 🔗 等多种前后缀形态）
    return r'\*\*[\\]*' + pattern + r'\*\*\s*\('


def get_symbol_from_clases_content(content: str):
    """从 chunk 的 content 中提取成员符号（方法/属性/常量/信号）。

    不含 enumerations：:ref:`类名<...>` 的链接文本是类名来源，类名改由
    header_path 提取（见 symbol.extract_symbols），避免内容引用把海量
    chunk 挂到核心类型符号上（曾导致 bool 挂 776 个 chunk）。"""
    symbols = set()
    for m in re.findall(pattern_of_property(), content):
        symbols.add(m)
    for m in re.findall(pattern_of_signals(), content):
        symbols.add(m)
    for m in re.findall(pattern_of_constant(), content):
        symbols.add(m)
    for m in re.findall(pattern_of_method(), content):
        symbols.add(m)
    return symbols


def match_sybol_from_classes_content(sym: str, content: str):
    """从 chunk 的 content 中匹配符号"""
    # sym 来自用户查询，先转义再拼接，防止元字符被正则引擎当成规则（正则注入）
    sym = re.escape(sym)
    symbols = set()
    for m in re.findall(pattern_of_enumerations(sym), content):
        symbols.add(m)
    for m in re.findall(pattern_of_property(sym), content):
        symbols.add(m)
    for m in re.findall(pattern_of_signals(sym), content):
        symbols.add(m)
    for m in re.findall(pattern_of_constant(sym), content):
        symbols.add(m)
    for m in re.findall(pattern_of_method(sym), content):
        symbols.add(m)
    return symbols