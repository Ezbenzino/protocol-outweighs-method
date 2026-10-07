"""Configuration loading with deep-merge for ablation configs."""
import re
from copy import deepcopy

import yaml

_NUM_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def _coerce(value):
    """Convert numeric-looking strings (e.g. '1e-4') into numbers."""
    if isinstance(value, dict):
        return {k: _coerce(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_coerce(v) for v in value]
    if isinstance(value, str) and _NUM_RE.match(value.strip()):
        s = value.strip()
        try:
            return int(s) if ("." not in s and "e" not in s.lower()) else float(s)
        except ValueError:
            return value
    return value


class AttrDict(dict):
    """dict that also supports attribute-style access (cfg.foo.bar)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = AttrDict(v)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)

    def __setattr__(self, key, value):
        self[key] = value


def _deep_merge(base, override):
    """Recursively merge `override` into a copy of `base`."""
    result = deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result


def load_config(path, base_path=None):
    """Load a YAML config; optionally merge it on top of a base config."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if base_path is not None:
        with open(base_path, "r", encoding="utf-8") as f:
            base = yaml.safe_load(f)
        cfg = _deep_merge(base, cfg)
    return AttrDict(_coerce(cfg))
