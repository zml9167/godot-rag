def ref_with_target_re(pattern: str = r'([^`<>\s]+)') -> str:
    return r'\{ref\}`\s*' + pattern + r'\s*<[^`>]*>`'


def ref_target_bold_ident_re(pattern: str = r'([A-Za-z_][A-Za-z0-9_.]*)') -> str:
    return r'\{ref\}`\s*[^`<]+?<[^`>]*>`\s*\*\*'+ pattern + r'\*\*'


def void_mark_bold_ident_re(pattern: str = r'([A-Za-z_][A-Za-z0-9_.]*)') -> str:
    return r'\{\{ void \}\} \*\*' + pattern + r'\*\*'
