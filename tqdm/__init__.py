"""Minimal tqdm stub for offline progress loops."""
from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")


def tqdm(iterable: Iterable[T], *_, **__) -> Iterator[T]:
    for item in iterable:
        yield item
