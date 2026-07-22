"""
MONARCHS run-settings schema - a single source of truth for every
``model_setup`` setting.

The package mirrors ``monarchs.variables``, one file per job:

* ``catalogue.py``   - **the data**: one ``Setting`` row per model_setup
                       setting. This is the file you edit to add/change one.
* ``rules.py``       - the cross-setting consistency checks (``RULES``).
* ``definitions.py`` - what a ``Setting``/``Rule`` is (+ ``type_matches``).
* ``computed_defaults.py`` - defaults computed from other settings.
* ``apply.py``       - fills defaults and validates a loaded model_setup.
* ``docs.py``        - renders the catalogue as a Markdown reference.

The single entry point is ``configure`` - it validates a loaded model_setup
and returns the frozen ``Config``::

    from monarchs.config import configure
    config = configure(model_setup)
"""

from monarchs.config.apply import (
    Config,
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
