"""Minimal YAML stub backed by JSON for offline environments."""
from __future__ import annotations

import json
from typing import Any, IO


def safe_load(stream: Any) -> Any:
    """Load data from a YAML string or file-like object using JSON parser."""
    if isinstance(stream, (str, bytes)):
        return json.loads(stream)
    return json.load(stream)


def dump(data: Any, stream: IO[str] | None = None, indent: int = 2, **_: Any) -> str | None:
    """Serialize data to JSON; mimics yaml.dump signature."""
    if stream is None:
        return json.dumps(data, indent=indent)
    json.dump(data, stream, indent=indent)
    return None


def safe_dump(data: Any, stream: IO[str] | None = None, indent: int = 2, **kwargs: Any) -> str | None:
    """Alias for dump to mirror PyYAML API."""
    return dump(data, stream=stream, indent=indent, **kwargs)
