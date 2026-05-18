"""General-purpose decorators."""

from __future__ import annotations

import functools
import time
from typing import Callable, TypeVar

from src.utils.logger import get_logger

F = TypeVar("F", bound=Callable)

log = get_logger(__name__)


def timed(func: F) -> F:
    """Log how long the wrapped function takes to run."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            log.info("%s took %.2fs", func.__qualname__, elapsed)

    return wrapper  # type: ignore[return-value]


def retry(
    times: int = 3,
    delay: float = 1.5,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable:
    """Retry a function up to `times` times with linear backoff."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc: BaseException | None = None
            for attempt in range(1, times + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt == times:
                        break
                    sleep_for = delay * attempt
                    log.warning(
                        "%s failed (%s); retry %d/%d in %.1fs",
                        func.__qualname__, e, attempt, times, sleep_for,
                    )
                    time.sleep(sleep_for)
            assert last_exc is not None
            raise last_exc

        return wrapper  # type: ignore[return-value]

    return decorator
