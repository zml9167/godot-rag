import re


def ref_with_target_re(pattern: str = r'([^`<>\s]+)') -> str:
    return r'\{ref\}`\s*' + pattern + r'\s*<[^`>]*>`'


def ref_target_bold_ident_re(pattern: str = r'([A-Za-z_][A-Za-z0-9_.]*)') -> str:
    return r'\{ref\}`\s*[^`<]+?<[^`>]*>`\s*\*\*'+ pattern + r'\*\*'


def void_mark_bold_ident_re(pattern: str = r'([A-Za-z_][A-Za-z0-9_.]*)') -> str:
    return r'\{\{ void \}\} \*\*' + pattern + r'\*\*'


def bold_text_pattern_in_signal(pattern: str = r'([^\*]+)') -> str:
    return r'\*\*' + pattern + '\*\*\([^:]+:\s+\{ref\}`[^<]+<[^>]+>`\)'


def get_symbol_from_clases_content(chunk: str):
    """从 chunk 的 content 中提取符号"""
    symbols = set()
    for m in re.findall(ref_with_target_re(), chunk):
        symbols.add(m)
    for m in re.findall(ref_target_bold_ident_re(), chunk):
        symbols.add(m)
    for m in re.findall(void_mark_bold_ident_re(), chunk):
        symbols.add(m)
    for m in re.findall(bold_text_pattern_in_signal(), chunk):
        symbols.add(m)
    return symbols