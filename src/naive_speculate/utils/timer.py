from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import TracebackType


class Timer:
    name: str
    start: float
    end: float
    elapsed: float

    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self) -> Timer:
        self.start = perf_counter()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> bool:
        self.end = perf_counter()
        self.elapsed = self.end - self.start
        return False
