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


class DummyResult:
    def __init__(self, f, *args, **kwargs):
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def result(self):
        return self.f(*self.args, **self.kwargs)


class DummyPool:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ...

    def submit(self, __f, *args, **kwargs):
        _ = self # to avoid 'can be static' hint
        return DummyResult(__f, *args, **kwargs)
