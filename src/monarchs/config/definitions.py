"""
Definitions for MONARCHS model setup settings.

We define here two dataclasses - a Setting (which has some attributes that
tell the model how it should be used later, and whether it is required or
optional), and a Rule (which determines how Settings interact, e.g. not letting
you specify conflicting settings).
"""

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

# sentinel for settings that *have* to be included
REQUIRED = object()

# sentinel for variables that can remain unset if certain conditions
# are met, e.g. not having a DEM means that some settings can be left alone
UNSET = object()


@dataclass(frozen=True, kw_only=True)
class Setting:
    """
    Defines a MONARCHS model_setup setting.

    ``default`` is a constant, REQUIRED, or a callable
    i.e. a function for defaults computed from other settings
    (which may return UNSET to leave the attribute missing).
    """

    name: str
    # dtype: bool, int, float, str, tuple, etc.
    # May be a tuple of types for settings that accept more than one, e.g.
    # cores=(str, int) or rho_init=(str, float).
    dtype: type | tuple
    default: Any = REQUIRED
    group: str = "run"  # organisational tag, used to group the docs
    # default_doc: how the default is reported in the docs/console for computed defaults,
    # e.g. "t_steps_per_day * 3600"
    default_doc: str = ""
    choices: tuple = ()  # allowed values, checked if the setting is present
    # TODO - check this
    validator: Callable = None
    # message describing what the validator requires, for the error/docs
    validator_doc: str = ""
    # switches read inside Numba kernels should set this to True so they are
    # added to the dict passed into the physics kernels
    kernel_toggle: bool = False
    # full description of what the setting does. this is what you would want
    # to see if reading a user manual!
    description: str = ""


@dataclass(frozen=True, kw_only=True)
class Rule:
    """
    A cross-setting consistency check.

    For example, we don't want to let a user specify both a DEM path and a
    firn column height - these two settings conflict, and we don't want the
    model to silently make choices about *which* of those to use.

    ``failed_when`` is a callable that returns True when the rule is violated.
    It is written as a ``lambda`` so the check lives with the rule,
    and because it is a function it is only evaluated *when the rules are run*
    (in ``apply.check_rules``), not when the catalogue is imported.
    This lets a rule refer to other settings without worrying about the order
    in which they are loaded (e.g. a rule can set a value based on another value).

    To add a rule, append one to ``rules.RULES``::

        Rule(
            failed_when=lambda ms: ms.dump_data and not hasattr(ms, "dump_filepath"),
            message="dump_data needs a dump_filepath to write to.",
            error=NameError,   # omit for a plain ValueError
        )

    ``failed_when`` is a lambda function - equivalent to doing
    def failed_when_func(ms):
        return ms.dump_data and not hasattr(ms, "dump_filepath")
    Use ``getattr(ms, "x", default)`` / ``hasattr`` for settings that may not
    be present yet (rules run before defaults are filled).
    """

    failed_when: Callable
    # message raised on error
    message: str
    # error type (a Warning subclass warns instead of raising)
    error: type = ValueError


def type_matches(value, dtype):
    """
    Ensure that a given config flag `value` has the expected `dtype`.

    `dtype` may be a single type or a tuple of types (a union), in which case
    the value need only match one of them.
    """
    if value is None or value is False:
        return True
    # a union of allowed types - value need only match one of them
    if isinstance(dtype, tuple):
        return any(type_matches(value, t) for t in dtype)
    # booleans/ints can be Numpy types also
    if dtype is bool:
        return isinstance(value, (bool, np.bool_))
    if dtype is int:
        return isinstance(value, (int, np.integer)) and not isinstance(value, bool)

    # floats could be defined in arrays or scalars
    if dtype is float:
        return isinstance(
            value, (int, float, np.integer, np.floating, np.ndarray)
        ) and not isinstance(value, bool)

    # lists/tuples can also be arrays
    if dtype in (list, tuple):
        return isinstance(value, (list, tuple, np.ndarray))
    # dicts have to be dicts, no numpy dtype to compare with
    if dtype is dict:
        return isinstance(value, dict)
    return True  # for any other datatypes we don't care so much about checking


def dtype_name(dtype):
    """Human-readable name for a dtype (or union tuple), for errors and docs."""
    if isinstance(dtype, tuple):
        return " or ".join(t.__name__ for t in dtype)
    return dtype.__name__
