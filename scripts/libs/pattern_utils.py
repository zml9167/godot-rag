import re


def pattern_of_signals(pattern: str = r'([A-Za-z_]+)') -> str:
    return r'\*\*' + pattern + r'\*\*\([^:]+:\s+\{ref\}`[^<]+<[^>]+>`\)'


def pattern_of_enumerations(pattern: str = r'([A-Za-z_]+)') -> str:
    # 类名槽 [A-Za-z0-9_]+：Godot 类名普遍含数字（Node2D/Vector3/CharacterBody2D）
    return r':ref:`' + pattern + r'<[A-Za-z0-9_]+>`'


def pattern_of_property(pattern: str = r'([a-z_]+)') -> str:
    return r'\{ref\}`[A-Za-z0-9_]+<[A-Za-z0-9_]+>`\s+\*\*' + pattern + r'\*\*'


def pattern_of_constant(pattern: str = r'([A-Z_]+)') -> str:
    return r'\*\*' + pattern + r'\*\*\s+=\s+'


def pattern_of_method(pattern: str = r'([A-Za-z_]+)') -> str:
    return r'\{ref\}`[A-Za-z0-9_]+<[A-Za-z0-9_]+>`\s+\*\*[\\]*'+ pattern + r'\*\*\(\)'


def get_symbol_from_clases_content(content: str):
    """从 chunk 的 content 中提取符号"""
    symbols = set()
    for m in re.findall(pattern_of_enumerations(), content):
        symbols.add(m)
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