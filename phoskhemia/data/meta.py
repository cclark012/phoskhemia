from __future__ import annotations
from typing import Any, Mapping
import numpy as np
from numpy.typing import NDArray

class MetaDict(dict[str, Any]):
    """
    dict-compatible metadata with optional typed attribute access.

    - Works with existing code: meta.get("noise_t0"), "noise_t0" in meta, JSON dumping.
    - Adds ergonomic access: meta.noise_t0, meta.scope_ppm, etc.
    """

    schema_id: str = "phoskhemia.ta.meta"
    meta_version: int = 1

    # ---- typed-ish properties for autocomplete ----
    @property
    def noise_t0(self) -> NDArray[np.floating] | float | None:
        v = self.get("noise_t0", None)
        if v is None:
            return None
        # preserve arrays; coerce sequences to ndarray only if you want
        if isinstance(v, NDArray[np.floating]):
            return v
        # allow scalar numeric
        if np.isscalar(v):
            try:
                return float(v)
            except Exception:
                return None
        # allow list/tuple -> ndarray (optional)
        try:
            return np.asarray(v, dtype=float)
        except Exception:
            return None

    @noise_t0.setter
    def noise_t0(self, value):
        if value is None:
            self.pop("noise_t0", None)
        else:
            self["noise_t0"] = value

    @property
    def scope_ppm(self) -> float | None:
        v = self.get("scope_ppm", None)
        try:
            return None if v is None else float(v)
        except Exception:
            return None

    @scope_ppm.setter
    def scope_ppm(self, value: float | None) -> None:
        if value is None:
            self.pop("scope_ppm", None)
        else:
            self["scope_ppm"] = float(value)

    @property
    def scope_jitter(self) -> float | None:
        v = self.get("scope_jitter", None)
        try:
            return None if v is None else float(v)
        except Exception:
            return None

    @scope_jitter.setter
    def scope_jitter(self, value: float | None) -> None:
        if value is None:
            self.pop("scope_jitter", None)
        else:
            self["scope_jitter"] = float(value)

    def normalized(self) -> "MetaDict":
        """
        Ensure schema markers exist and return self (for chaining).
        """
        self.setdefault("schema_id", self.schema_id)
        self.setdefault("meta_version", self.meta_version)
        return self

    @staticmethod
    def coerce(meta: Mapping[str, Any] | None) -> "MetaDict":
        if isinstance(meta, MetaDict):
            return meta.normalized()
        out = MetaDict(meta or {})
        return out.normalized()

def meta_copy_update(
        meta: Mapping[str, Any] | None,
        updates: Mapping[str, Any] | None = None,
    ) -> MetaDict:
    """
    Coerce to MetaDict, shallow-copy, apply updates, and normalize schema markers.
    """
    m = MetaDict.coerce(meta)
    out = MetaDict(dict(m))  # shallow copy; preserves nested arrays by ref
    if updates:
        out.update(dict(updates))
    return out.normalized()
