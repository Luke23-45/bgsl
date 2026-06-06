"""
studies/runner/commons/ablation_conditions.py
---------------------------------------------
Strict condition-to-override translation for study runners.

The helpers here intentionally fail fast if a condition is missing a
`class_path`, contains an unexpected key, or tries to use an unmapped
structure. The baseline study depends on exact run definitions, not
best-effort inference.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict


_TOP_LEVEL_KEYS = {"name", "loss", "data"}
_LOSS_KEYS = {"class_path", "init_args"}


def build_condition_overrides(condition: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Convert a study condition into dot-path CLI overrides.

    Supported schema
    ----------------
    condition:
      name: "..."
      loss:
        class_path: "package.ClassName"
        init_args:
          some_arg: 1
      data:
        tau: 2.0
    """
    if not isinstance(condition, Mapping):
        raise TypeError(f"Condition must be a mapping, got {type(condition)!r}")

    unknown_top_level = set(condition.keys()) - _TOP_LEVEL_KEYS
    if unknown_top_level:
        raise ValueError(f"Unsupported condition keys: {sorted(unknown_top_level)!r}")

    overrides: Dict[str, Any] = {}

    loss = condition.get("loss")
    if loss is not None:
        if not isinstance(loss, Mapping):
            raise TypeError(f"'loss' must be a mapping, got {type(loss)!r}")
        unknown_loss_keys = set(loss.keys()) - _LOSS_KEYS
        if unknown_loss_keys:
            raise ValueError(f"Unsupported loss keys: {sorted(unknown_loss_keys)!r}")

        class_path = loss.get("class_path")
        if not class_path:
            raise ValueError("Loss condition is missing required 'class_path'.")
        # LightningCLI subclass_mode_model=True nests all constructor args under
        # model.init_args — the loss_fn key lives at model.init_args.loss_fn.
        overrides["model.init_args.loss_fn"] = str(class_path)

        init_args = loss.get("init_args", {})
        if init_args:
            if not isinstance(init_args, Mapping):
                raise TypeError(f"'loss.init_args' must be a mapping, got {type(init_args)!r}")
            for key, value in init_args.items():
                if isinstance(value, Mapping):
                    raise ValueError(
                        f"Nested mappings are not supported for loss.init_args.{key!s}."
                    )
                overrides[f"model.init_args.loss_fn.init_args.{key}"] = value

    data = condition.get("data")
    if data is not None:
        if not isinstance(data, Mapping):
            raise TypeError(f"'data' must be a mapping, got {type(data)!r}")
        for key, value in data.items():
            if isinstance(value, Mapping):
                raise ValueError(f"Nested mappings are not supported for data.{key!s}.")
            overrides[f"data.{key}"] = value

    if not overrides:
        raise ValueError(
            f"Condition {condition.get('name', '<unnamed>')!r} does not define any overrides."
        )

    return overrides
