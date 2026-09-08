"""
Render the met-forcing catalogue (`monarchs.met_data.catalogue`) as a Markdown
reference.
"""

from monarchs.docs_tables import render_reference
from monarchs.met_data.catalogue import MET_CATALOGUE

# define table columns
_COLUMNS = ["Variable", "Long name", "Units", "ERA5 name", "Description"]

# define an intro paragraph that is displayed above the schema table
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


def to_markdown():
    """Return the met-forcing catalogue as a Markdown reference."""
    sections = [(None, [_row(var) for var in MET_CATALOGUE])]
    return render_reference(
        "MONARCHS met forcing variables", _INTRO, _COLUMNS, sections
    )
