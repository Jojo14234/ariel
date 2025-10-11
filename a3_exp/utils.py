import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Dict


@contextmanager
def timeit(name: str, silent: bool = False):
    try:
        t = time.perf_counter()
        yield
    finally:
        elapsed = time.perf_counter() - t
        silent or print(f"{name} took {elapsed:.2f}s" + [f" ({elapsed * 1000:.2f}ms)", ''][elapsed > 1])


def argmax(x: List):
    return max(range(len(x)), key=x.__getitem__)


def repr_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def repr_kw(d: Dict):
    return " ".join(f"{k}_{v}" for k, v in d.items())


def fd_count():
    return len(os.listdir(p)) if (p := Path(f"/proc/{os.getpid()}/fd")).exists() else -1


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
