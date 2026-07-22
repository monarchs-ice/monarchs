"""
Apply the settings catalogue to a loaded ModelSetup object
(see monarchs.core.load_model_setup).

The defined functions here operate on a ModelSetup instance,
which is mutable and allows for its parameters to be changed
based on the rules we have defined. This is then at the end
converted into an immutable Config object, which cannot be
changed during runtime.
"""

import dataclasses
import warnings
from typing import Any
from monarchs.config.catalogue import SETTINGS
from monarchs.config.rules import RULES
from monarchs.config.definitions import REQUIRED, UNSET, dtype_name, type_matches

# Create a dataclass based on the parameters in the catalogue.
# frozen=True makes it immutable, i.e. the configuration at
# the start of a run cant be changed later on
Config = dataclasses.make_dataclass(
    "Config",
    [(setting.name, Any) for setting in SETTINGS],
    frozen=True,
)


def check_rules(model_setup, rules=RULES):
    """Run the cross-setting rules, raise (or warn) on the first violation."""
    func_name = "monarchs.config.apply.check_rules"
    for rule in rules:
        # failed_when is the condition we are testing
        # against, defined by the lambda function
        # at the start of the Rule
        if rule.failed_when(model_setup):
            # some rules just raise warnings, not errors
            if issubclass(rule.error, Warning):
                warnings.warn(f"{func_name}: {rule.message}", rule.error)
            # others prevent the model from working properly, so raise an error
            else:
                raise rule.error(f"{func_name}: {rule.message}")


def check_settings(model_setup, settings=SETTINGS):
    """Validate the settings the user provided against choices/validators."""
    func_name = "monarchs.config.apply.check_settings"

    for setting in settings:
        # ignore unpresent settings
        if not hasattr(model_setup, setting.name):
            continue
        value = getattr(model_setup, setting.name)
        # check the type matches, e.g. lateral_movement_toggle can be
        # Bool or np.bool (see definitions.type_matches for the rules)
        if not type_matches(value, setting.dtype):
            raise TypeError(
                f"{func_name}: {setting.name} should be a {dtype_name(setting.dtype)},"
                f" got {type(value).__name__} ({value!r})"
            )
        # if a setting has a limited number of choices, flag if it is specified
        # as something outside of those choices
        if setting.choices and value not in setting.choices:
            raise ValueError(
                f"{func_name}: {setting.name} must be one of"
                f" {list(setting.choices)}, not {value}"
            )
        # run validation rules on settings
        if setting.validator is not None and not setting.validator(value):
            requirement = setting.validator_doc or "valid"
            raise ValueError(
                f"{func_name}: {setting.name} must be {requirement}, not {value}"
            )


def fill_defaults(model_setup, settings=SETTINGS):
    """
    Set catalogue defaults for any missing settings, resolving computed
    defaults in catalogue order. Raises if a REQUIRED setting is missing.
    """
    func_name = "monarchs.config.apply.fill_defaults"
    missing_required = []
    filled = []  # (name, reported default) collected for a single summary below
    for setting in settings:
        # if it is present in the setup then just move on
        if hasattr(model_setup, setting.name):
            continue
        # if a missing setting is required, then don't fill it
        # in, instead add it to a list of things to return to the
        # user via an error at the end
        if setting.default is REQUIRED:
            missing_required.append(setting.name)
            continue
        # if a missing parameter has UNSET as default, then don't fill it in,
        # but also don't raise an error as it is optional
        if setting.default is UNSET:
            continue
        # if the default value is defined by a function, then
        # run that function to generate the default value.
        # e.g. any parameters that are derived from the model grid
        # size will be callables
        if callable(setting.default):
            value = setting.default(model_setup)
            if value is UNSET:
                continue
            default_message = setting.default_doc or value

        else:
            value = setting.default
            default_message = value
        setattr(model_setup, setting.name, value)
        filled.append((setting.name, default_message))
    # report every filled default once, as a block at the top of the run, rather
    # than a line per setting scattered through the log
    if filled:
        summary = "\n".join(f"    {name} = {msg}" for name, msg in filled)
        print(
            f"{func_name}: filled {len(filled)} missing setting(s) with default"
            f" values:\n{summary}"
        )
    # raise those errors if we got any. does this for all settings issues
    # not just the first
    if missing_required:
        raise AttributeError(
            f"{func_name}: model_setup is missing required setting(s)"
            f" {missing_required} - these have no defaults and must be"
            " specified in your runscript."
        )


def configure(model_setup):
    """
    Run the setup pipeline. Check for issues with conflicting settings,
    check they are valid, fill in default values into the settings list
    if missing, and then build the dataclass from the result.
    """
    check_rules(model_setup)
    check_settings(model_setup)
    fill_defaults(model_setup)
    # TODO - describe syntax here
    config = Config(
        **{
            setting.name: getattr(model_setup, setting.name, None)
            for setting in SETTINGS
        }
    )
    return config
