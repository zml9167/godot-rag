import re


def pattern_of_signals(pattern: str = r'([A-Za-z_]+)') -> str:
    return r'\*\*' + pattern + r'\*\*\([^:]+:\s+\{ref\}`[^<]+<[^>]+>`\)'


def pattern_of_enumerations(pattern: str = r'([A-Za-z_]+)') -> str:
    return r':ref:`' + pattern + r'<[A-Za-z_]+>`'


def pattern_of_property(pattern: str = r'([a-z_]+)') -> str:
    return r'\{ref\}`[A-Za-z]+<[A-Za-z_]+>`\s+\*\*' + pattern + r'\*\*'


def pattern_of_constant(pattern: str = r'([A-Z_]+)') -> str:
    return r'\*\*' + pattern + r'\*\*\s+=\s+'


def pattern_of_method(pattern: str = r'([A-Za-z_]+)') -> str:
    return r'\{ref\}`[A-Za-z]+<[A-Za-z_]+>`\s+\*\*[\\]*'+ pattern + r'\*\*\(\)'


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