"""
Render the met-forcing catalogue (`monarchs.met_data.catalogue`) as a Markdown
reference, using the shared `monarchs.docs_tables` helpers like the settings and
grid-variable references.

Regenerated during the docs build (see docs/source/conf.py).
"""

from monarchs.docs_tables import render_reference
from monarchs.met_data.catalogue import MET_CATALOGUE

_COLUMNS = ["Variable", "Long name", "Units", "ERA5 name", "Description"]

_INTRO = [
    "Fields of the met data grid (one record per timestep per cell), from the "
    "met catalogue (`monarchs.met_data.catalogue`). The ERA5 name is the short "
    "name(s) the field is read from when using ERA5 input data.",
]


def _row(var):
    return [
        f"`{var.name}`",
        var.long_name,
        var.units,
        f"`{var.era5_name}`",
        var.description,
    ]


def to_markdown(met_catalogue=MET_CATALOGUE):
    """Return the met-forcing catalogue as a Markdown reference."""
    sections = [(None, [_row(var) for var in met_catalogue])]
    return render_reference(
        "MONARCHS met forcing variables", _INTRO, _COLUMNS, sections
    )
