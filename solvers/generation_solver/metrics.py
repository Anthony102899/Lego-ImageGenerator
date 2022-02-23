import time


class Metrics:
    def __init__(self):
        self._start_time = None
        self._end_time = None

    def measure_without_return(self, function, *args, **kwargs):
        self._start_time = time.time()
        function(*args, **kwargs)
        self._end_time = time.time()
        print(f"{'#' * 5} Metric Measure Cost -> function {function.__name__} uses {self._end_time - self._start_time}s")

    def measure_with_return(self, function, *args, **kwargs):
        self._start_time = time.time()
        result = function(*args, **kwargs)
        self._end_time = time.time()
        print(f"{'#' * 5} Metric Measure Cost -> function {function.__name__} uses {self._end_time - self._start_time}s")
        return result


class TestMetrics:
    def __init__(self):
        self._name = "Test"

    def add(self, a, b):
        return a + b


if __name__ == "__main__":
    test_metric = TestMetrics()
    metrics = Metrics()
    result = metrics.measure_with_return(test_metric.add, 1, 2)
    print(result)
