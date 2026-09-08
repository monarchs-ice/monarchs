"""
Functions for rendering catalogue-derived Markdown reference pages.

This just handles the formatting. The actual content is defined via the
relevant `docs.py` in the `variables`, `config` and `met_data` submodules.
"""

# group names that are acronyms - define these separately so that we don't
# format these to "Dem" rather than "DEM" etc.
_ACRONYMS = {"dem": "DEM", "io": "IO"}


def heading(group):
    """Section heading for a group name, keeping acronyms uppercase."""
    return _ACRONYMS.get(group, group.capitalize())


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
        Labels for each field - e.g. heading/default value etc.
    sections : list of (str or None, list of list of str)
        ``(section_title, rows)`` pairs. ``section_title`` is rendered as a
        ``##`` heading (omitted when ``None``), one value per column.
    """
    out = [f"# {title}", ""]
    out += list(intro)
    if intro:
        out.append("")
    for section_title, rows in sections:
        # without a section heading the entries sit directly under the title,
        # so promote them a level rather than skipping one
        entry_level = "###" if section_title is not None else "##"
        if section_title is not None:
            out += [f"## {section_title}", ""]
        for row in rows:
            out += [f"{entry_level} {row[0]}", ""]
            # short fields on one line, skipping any this entry leaves blank
            fields = [
                f"**{label}:** {value}"
                for label, value in zip(columns[1:-1], row[1:-1])
                if value
            ]
            if fields:
                out += [" &nbsp;·&nbsp; ".join(fields), ""]
            if row[-1]:
                out += [row[-1], ""]
    return "\n".join(out).rstrip() + "\n"
