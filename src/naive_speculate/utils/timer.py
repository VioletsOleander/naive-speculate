from time import perf_counter


class Timer:
    name: str
    start: float
    end: float
    elapsed: float

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = perf_counter()
        self.elapsed = self.end - self.start
        return False
