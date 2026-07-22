"""
Render the variable catalogue as a Markdown reference.

Reads the columns (name, dimension, units, default, description) from the
``Variable`` rows in ``catalogue.py``, organised by ``group``, using the shared
`monarchs.docs_tables` helpers. The met-forcing catalogue has its own reference
(`monarchs.met_data.docs`).

Run ``scripts/gen_docs.py`` to write the reference out.
"""

from monarchs.docs_tables import group_order, render_reference
from monarchs.variables.catalogue import CATALOGUE
from monarchs.variables.definitions import INPUT

_COLUMNS = ["Variable", "Long name", "Dim", "Units", "Default", "Description"]

_INTRO = [
    "List of MONARCHS model grid variables. Generated automatically from the "
    "variable catalogue (`monarchs.variables`).  "
    "run `python scripts/gen_docs.py` to regenerate!",
    "",
    "Units broadly follow the CF conventions - see "
    "https://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/build/ch03.html for details.",
]


def _default_cell(value):
    """Map default value descriptors to human-readable descriptions."""
    if value is INPUT:
        return "from input"
    if callable(value):
        return f"computed (`{value.__name__}`)"
    return f"`{value!r}`"


def _row(var):
    return [
        f"`{var.name}`",
        var.long_name,
        var.dim.value,
        var.units,
        _default_cell(var.default_value),
        var.description,
    ]


def to_markdown(catalogue=CATALOGUE):
    """Return the variable catalogue as a Markdown reference."""
    sections = [
        (group.capitalize(), [_row(v) for v in catalogue if v.group == group])
        for group in group_order(catalogue, lambda v: v.group)
    ]
    return render_reference("MONARCHS grid variables", _INTRO, _COLUMNS, sections)
