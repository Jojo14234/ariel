import time
from contextlib import contextmanager


@contextmanager
def timeit(name: str, silent: bool = False):
    try:
        t = time.perf_counter()
        yield
    finally:
        elapsed = time.perf_counter() - t
        silent or print(f"{name} took {elapsed:.2f}s" + [f" ({elapsed * 1000:.2f}ms)", ''][elapsed > 1])
