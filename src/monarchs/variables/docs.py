"""
Render the variable catalogue as a Markdown reference.

Reads the columns (name, dimension, units, default, description) from
the ``Variable`` rows in ``catalogue.py``, and organise them by ``group``.

Run ``scripts/gen_variable_docs.py`` to write the reference out.
"""

from monarchs.variables.catalogue import CATALOGUE
from monarchs.variables.definitions import INPUT


def _default_cell(value):
    """Map default value descriptors to human-readable descriptions."""
    if value is INPUT:
        return "from input"
    if callable(value):
        return f"computed (`{value.__name__}`)"
    return f"`{value!r}`"


def _groups_in_order(catalogue):
    """Group names in order as determined by the catalogue."""
    order = []
    for var in catalogue:
        if var.group not in order:
            order.append(var.group)
    return order


def to_markdown(catalogue=CATALOGUE):
    """Return the whole catalogue as a Markdown reference."""
    out = [
        "# MONARCHS grid variables",
        "",
        "List of MONARCHS model grid variables. Generated from the "
        "variable catalogue (`monarchs.variables`). Don't edit this manually, "
        "run `python scripts/gen_variable_docs.py` to regenerate!"
        "",
        "Units broadly follow the CF conventions - see "
        "https://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/build/ch03.html for details.",
        "",
    ]
    header = "| Variable | Long name | Dim | Units | Default | Description |"
    rule = "| --- | --- | --- | --- | --- | --- |"
    # iterate over all the groups we can find in the catalogue
    for group in _groups_in_order(catalogue):
        # make a header for that group
        out += [f"## {group.capitalize()}", "", header, rule]
        for var in catalogue:
            # ignore if not in the current group
            if var.group != group:
                continue
            out.append(
                f"| `{var.name}` | {var.long_name} | {var.dim.value} "
                f"| {var.units} | {_default_cell(var.default_value)} "
                f"| {var.description} |"
            )
        out.append("")
    return "\n".join(out).rstrip() + "\n"


if __name__ == "__main__":
    print(to_markdown(), end="")
