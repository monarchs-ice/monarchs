"""
Functions for rendering catalogue-derived Markdown reference tables.

This just handles the formatting. The actual content is defined via the
relevant `docs.py` in the `variables`, `config` and `met_data` submodules.
"""


def group_order(items, key):
    """Distinct values of ``key(item)``, in first-seen order."""
    order = []
    for item in items:
        value = key(item)
        if value not in order:
            order.append(value)
    return order


def render_reference(title, intro, columns, sections):
    """
    Render a Markdown reference page.

    Parameters
    ----------
    title : str
        Page title, rendered as the top-level ``#`` heading.
    intro : list of str
        Introduction lines placed directly under the title.
    columns : list of str
        Table column headers, shared by every section.
    sections : list of (str or None, list of list of str)
        ``(section_title, rows)`` pairs. ``section_title`` is rendered as a
        ``##`` heading (omitted when ``None``); each row is a list of
        pre-formatted cell strings, one per column.
    """
    out = [f"# {title}", ""]
    out += list(intro)
    if intro:
        out.append("")
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join(["---"] * len(columns)) + " |"
    for section_title, rows in sections:
        if section_title is not None:
            out += [f"## {section_title}", ""]
        out += [header, rule]
        for row in rows:
            out.append("| " + " | ".join(row) + " |")
        out.append("")
    return "\n".join(out).rstrip() + "\n"
