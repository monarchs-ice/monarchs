"""
MONARCHS run-settings schema.

Files:

``catalogue.py`` - contains all ``model_setup`` variable definitions.
``rules.py`` - rules that apply to settings or combinations of settings
``definitions.py`` -
``computed_defaults.py`` - calculate default values for settings that are
                           dependent on other settings
``apply.py`` - apply the config at runtime, run validation, fill default values
``docs.py`` - converts the catalogue to documentation
"""

from monarchs.config.apply import (
    Config,
    check_required,
    check_rules,
    check_settings,
    configure,
    fill_defaults,
)
from monarchs.config.catalogue import SETTINGS
from monarchs.config.rules import RULES
from monarchs.config.definitions import REQUIRED, UNSET, Rule, Setting
from monarchs.config.docs import to_markdown

__all__ = [
    "Config",
    "check_required",
    "check_rules",
    "check_settings",
    "configure",
    "fill_defaults",
    "to_markdown",
    "SETTINGS",
    "RULES",
    "Setting",
    "Rule",
    "REQUIRED",
    "UNSET",
]
