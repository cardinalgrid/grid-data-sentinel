"""Serialisation of detector state: plain types, nested dicts and lists, numpy arrays and datetimes."""

from __future__ import annotations

import json
from typing import Any

import numpy as np


def _encode(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode(v) for v in obj]
    if isinstance(obj, np.ndarray):
        if np.issubdtype(obj.dtype, np.datetime64):
            return {"__ndarray__": obj.astype("datetime64[ns]").astype("int64").tolist(), "dtype": "datetime64[ns]"}
        return {"__ndarray__": obj.tolist(), "dtype": str(obj.dtype)}
    if isinstance(obj, np.datetime64):
        return {"__datetime64__": int(obj.astype("datetime64[ns]").astype("int64"))}
    if isinstance(obj, np.generic):
        return obj.item()
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    raise TypeError(f"state contains an unsupported type: {type(obj).__name__}")


def _decode(obj: Any) -> Any:
    if isinstance(obj, dict):
        if "__ndarray__" in obj:
            if obj["dtype"] == "datetime64[ns]":
                return np.array(obj["__ndarray__"], dtype="int64").astype("datetime64[ns]")
            return np.array(obj["__ndarray__"], dtype=obj["dtype"])
        if "__datetime64__" in obj:
            return np.int64(obj["__datetime64__"]).astype("datetime64[ns]")
        return {k: _decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode(v) for v in obj]
    return obj


def state_to_json(state: dict[str, Any]) -> str:
    return json.dumps(_encode(state))


def state_from_json(text: str) -> dict[str, Any]:
    return _decode(json.loads(text))
