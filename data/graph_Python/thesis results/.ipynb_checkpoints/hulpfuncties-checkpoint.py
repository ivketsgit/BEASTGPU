from dataclasses import dataclass, field
from typing import List
import re
import statistics

@dataclass
class BenchmarkResult:
    duration: float
    runs: int
    min: float
    mean: float
    max: float
    std: float
    median: float
    percentile_75: float

@dataclass
class BenchmarkSuite:
    results: List[BenchmarkResult] = field(default_factory=list)

    def add_result(self, result: BenchmarkResult):
        self.results.append(result)

    def average(self) -> BenchmarkResult:
        count = len(self.results)
        if count == 0:
            raise ValueError("No results to average.")
        
        return BenchmarkResult(
            duration=sum(r.duration for r in self.results) / count,
            runs=sum(r.runs for r in self.results) // count,
            min=min(r.min for r in self.results),
            mean=statistics.mean(r.mean for r in self.results),
            max=max(r.max for r in self.results),
            std=statistics.mean(r.std for r in self.results),
            median=statistics.mean(r.median for r in self.results),
            percentile_75=statistics.mean(r.percentile_75 for r in self.results),
        )

def parse_benchmark_file(text: str) -> BenchmarkSuite:
    suite = BenchmarkSuite()

    # Pattern to match each benchmark block
    block_pattern = re.compile(
        r"Manual Benchmark of duration ([\d.]+) over (\d+) runs:\s*"
        r"Min: ([\d.]+) s\s*"
        r"Mean: ([\d.]+) s\s*"
        r"Max: ([\d.]+) s\s*"
        r"Std: ([\d.]+) s\s*"
        r"2nd Quartile \(Median\): ([\d.]+) s\s*"
        r"3rd Quartile \(75th percentile\): ([\d.]+) s",
        re.MULTILINE
    )

    for match in block_pattern.finditer(text):
        result = BenchmarkResult(
            duration=float(match.group(1)),
            runs=int(match.group(2)),
            min=float(match.group(3)),
            mean=float(match.group(4)),
            max=float(match.group(5)),
            std=float(match.group(6)),
            median=float(match.group(7)),
            percentile_75=float(match.group(8)),
        )
        suite.add_result(result)

    return suite

def read_benchmark_file(content):
    match = re.search(r'\[(.*?)\]', content)
    if not match:
        raise ValueError("No array found in file.")
    data_str = match.group(1)
    times = [float(x.strip()) for x in data_str.split(',')]
    return times
