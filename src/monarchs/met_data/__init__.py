"""
MONARCHS met-forcing schema.

Like ``monarchs.config`` and ``monarchs.variables``, the met-forcing fields live
in a single catalogue that everything else is generated from:

* ``catalogue.py``     - **the data**: one ``MetVariable`` per met field
                         (``MET_CATALOGUE``), plus ``met_dtype`` (the structured
                         -array dtype for the met grid).
* ``docs.py``          - renders the catalogue as a Markdown reference.
* ``met_data_grid.py`` - builds the per-run met structured array.
* ``import_ERA5.py``   - reads an ERA5 netCDF into MONARCHS fields, driven by the
                         catalogue's ``era5_name`` / ``derived_from`` / ``convert``.
* ``setup_met_data.py`` / ``load.py`` - interpolate and load met data for a run.

The catalogue and its Markdown reference are re-exported here so they can be
used the same way as the other schemas, e.g.::

    from monarchs.met_data import to_markdown, MET_CATALOGUE
"""

from monarchs.met_data.catalogue import MET_CATALOGUE, MetVariable, met_dtype
from monarchs.met_data.docs import to_markdown

__all__ = [
    "MET_CATALOGUE",
    "MetVariable",
    "met_dtype",
    "to_markdown",
]
