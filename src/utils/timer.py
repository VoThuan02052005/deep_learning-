import time
from contextlib import contextmanager

@contextmanager
def timer(name: str = "Block"):
    start = time.time()
    yield
    end = time.time()
    print(f"[{name}] Time elapsed: {end - start:.2f}s")
