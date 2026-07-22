"""
Render the settings catalogue as a Markdown reference.

Reads the columns (name, type, default, choices, description) from the settings
catalogue (`monarchs.config.catalogue`) and organises them by ``group``, using
the shared `monarchs.docs_tables` helpers.

Regenerated during the docs build (see docs/source/conf.py).
"""

from monarchs.config.catalogue import SETTINGS
from monarchs.config.definitions import REQUIRED, UNSET, dtype_name
from monarchs.docs_tables import group_order, render_reference

_COLUMNS = ["Setting", "Type", "Default", "Allowed", "Description"]

_INTRO = [
    "This provides a reference for all of the possible settings",
    "available in the model.",
    "",
    "It is automatically generated from the variables "
    "in the settings catalogue (`monarchs.config.catalogue`).",
    "",
    "If you add more settings, these will be automatically added here "
    "provided you have added them to the catalogue itself.",
]


def _default_cell(setting):
    """formatting for default values"""
    if setting.default is REQUIRED:
        return "**required**"
    if setting.default is UNSET:
        return "optional (unset)"
    if callable(setting.default):
        return setting.default_doc or f"computed (`{setting.default.__name__}`)"
    return f"`{setting.default!r}`"


def _constraint_cell(setting):
    """formatting for values that have rules/constraints about their use"""
    if setting.choices:
        return ", ".join(f"`{c}`" for c in setting.choices)
    if setting.validator is not None:
        return setting.validator_doc
    return ""


def _row(setting):
    return [
        f"`{setting.name}`",
        dtype_name(setting.dtype),
        _default_cell(setting),
        _constraint_cell(setting),
        setting.description,
    ]


def to_markdown(settings=SETTINGS):
    """Return the settings catalogue as Markdown."""
    sections = [
        (group.capitalize(), [_row(s) for s in settings if s.group == group])
        for group in group_order(settings, lambda s: s.group)
    ]
    return render_reference("MONARCHS run settings", _INTRO, _COLUMNS, sections)
