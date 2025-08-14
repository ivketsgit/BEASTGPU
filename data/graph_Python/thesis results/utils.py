import re
import statistics
from dataclasses import dataclass, field
from typing import List
from pathlib import Path

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

def read_benchmark_file(content: str) -> List[float]:
    match = re.search(r'\[(.*?)\]', content)
    if not match:
        raise ValueError("No array found in file.")
    data_str = match.group(1)
    times = [float(x.strip()) for x in data_str.split(',')]
    return times

def read_benchmark_file_(content: str) -> List[List[float]]:
    matches = re.findall(r'\[([^\]]+)\]', content)
    if not matches:
        raise ValueError("No arrays found in file.")

    all_times = []
    for data_str in matches:
        times = [float(x.strip()) for x in data_str.split(',')]
        all_times.append(times)
    
    return all_times

def read_benchmark_file_multi(content: str) -> List[float]:
    matches = re.findall(r'\[([^\]]+)\]', content)
    if not matches:
        raise ValueError("No arrays found in file.")

    all_times = []
    for data_str in matches:
        all_times.extend(float(x.strip()) for x in data_str.split(','))

    return all_times



import numpy as np
import os

def get_density_list():
    with open("BEASTGPU/data/graph_data.jl", "r") as f:
        content = f.read()
    density_values = re.search(r'density_values\s*=\s*\[([^\]]+)\]', content)
    intgral_amount = re.search(r'intgral_amount\s*=\s*\[([^\]]+)\]', content)
    system_matrix_size = re.search(r'system_matrix_size\s*=\s*\[([^\]]+)\]', content)
    
    if density_values and intgral_amount:
        density_list = np.array(list(map(int, density_values.group(1).split(','))))
        intgral_list = np.array(list(map(int, intgral_amount.group(1).split(','))))
        system_matrix_size = np.array(list(map(int, system_matrix_size.group(1).split(','))))
    else:
        print("One or both arrays not found.")
    return density_list, intgral_list, system_matrix_size

def get_pStore_data():
    with open("BEASTGPU/data/graph_data.jl", "r") as f:
        content = f.read()
    partial_store_threads = re.search(r'partial_store_threads\s*=\s*\[([^\]]+)\]', content)
    partial_store_mem = re.search(r'partial_store_mem\s*=\s*\[([^\]]+)\]', content)
    
    if partial_store_threads and partial_store_mem:
        partial_store_threads = np.array(list(map(int, partial_store_threads.group(1).split(','))))
        partial_store_mem = np.array(list(map(int, partial_store_mem.group(1).split(','))))
    else:
        print("One or both arrays not found.")
    return partial_store_threads, partial_store_mem


def merge_paths(*paths):
    return Path(paths[0]).joinpath(*paths[1:])

import os

def load_cpu_stats_from_files_multi(path, density_values, filename_suffix):
    median_values = []
    min_values = []
    max_values = []
    std_values = []

    for density_value in density_values:
        file_path = merge_paths("BEASTGPU/data/", path, "/" + str(density_value), "/" + filename_suffix)
        file_path = os.path.join("BEASTGPU", "data", path, str(density_value), filename_suffix)
        with open(file_path, "r") as f:
            content = f.read()

        times = np.array(read_benchmark_file_multi(content))

        median_values.append(np.median(times))
        min_values.append(np.min(times))
        max_values.append(np.max(times))
        std_values.append(np.std(times))

    return (
        np.array(median_values), 
        np.array(min_values), 
        np.array(max_values), 
        np.array(std_values)
    )
    # return {
    #     "median": np.array(median_values),
    #     "min": np.array(min_values),
    #     "max": np.array(max_values),
    #     "std": np.array(std_values),
    # }

def load_cpu_stats_from_files(path, density_values, filename_suffix):
    median_values = []
    min_values = []
    max_values = []
    std_values = []

    for density_value in density_values:
        file_path = merge_paths("BEASTGPU/data", path, str(density_value), filename_suffix)
        with open(file_path, "r") as f:
            content = f.read()

        times = np.array(read_benchmark_file(content))

        median_values.append(np.median(times))
        min_values.append(np.min(times))
        max_values.append(np.max(times))
        std_values.append(np.std(times))

    return (
        np.array(median_values), 
        np.array(min_values), 
        np.array(max_values), 
        np.array(std_values)
    )


def read_benchmark_file_multi_sautswab(content: str) -> List[float]:
    # Remove whitespace and trailing commas
    content = content.strip().rstrip(',')

    if not content:
        raise ValueError("File is empty or contains no numbers.")

    try:
        # Split by commas and convert to floats
        return [float(x) for x in content.split(',') if x.strip()]
    except ValueError as e:
        raise ValueError(f"Invalid number found in file: {e}")





def load_cpu_stats_from_files_sautswab(path, density_values, filename_suffix):
    median_values = []
    min_values = []
    max_values = []
    std_values = []
    avg_worst_mean = []

    for density_value in density_values:
        file_path = merge_paths("BEASTGPU/data", path, str(density_value), filename_suffix)
        with open(file_path, "r") as f:
            content = f.read()

        times = np.array(read_benchmark_file_multi_sautswab(content))
        # print(times)
        chunk_means = [
            np.mean(times[i:i+16])
            for i in range(0, len(times), 16)
        ]
        
        avg_worst_mean.append(max(chunk_means))

        median_values.append(np.median(times))
        # print(median_values)
        min_values.append(np.min(times))
        max_values.append(np.max(times))
        std_values.append(np.std(times))
        # print("\n")

    return (
        np.array(median_values), 
        np.array(min_values), 
        np.array(max_values), 
        np.array(std_values),
        np.array(avg_worst_mean)
    )



def load_cpu_stats_from_files_pînned(path, density_values):
    median_values = []
    min_values = []
    max_values = []
    std_values = []

    for density_value in density_values:
        file_path = merge_paths("BEASTGPU/data", path, str(density_value) + ".txt")
        with open(file_path, "r") as f:
            content = f.read()

        times = np.array(read_benchmark_file(content))

        median_values.append(np.median(times))
        min_values.append(np.min(times))
        max_values.append(np.max(times))
        std_values.append(np.std(times))

    return (
        np.array(median_values), 
        np.array(min_values), 
        np.array(max_values), 
        np.array(std_values)
    )


def load_cpu_stats_from_files_pînned_2(path, density_values, name):
    median_values = []
    min_values = []
    max_values = []
    std_values = []

    for density_value in density_values:
        file_path = merge_paths("BEASTGPU/data", path, str(density_value) + "_" + name + ".txt")
        with open(file_path, "r") as f:
            content = f.read()

        times = np.array(read_benchmark_file(content))

        median_values.append(np.median(times))
        min_values.append(np.min(times))
        max_values.append(np.max(times))
        std_values.append(np.std(times))

    return (
        np.array(median_values), 
        np.array(min_values), 
        np.array(max_values), 
        np.array(std_values)
    )




















import ast  
def calculate_avg_and_worst_times(path, filename, thread_counts):

    average_total_times_2 = []
    average_worst_times_2 = []
    average_total_times_2_std = []
    average_worst_times_2_std = []

    for thread_count in thread_counts:
        full_path = merge_paths("BEASTGPU","data", "GPU", path, filename + "_" +  str(thread_count) + ".txt")
        with open(full_path, "r") as f:
            content = f.read()
        
            try:
                start = content.find('[[')
                end = content.rfind(']]') + 2
        
                if start == -1 or end == -1:
                    raise ValueError("No valid data list found in the file.")
        
                data_str = content[start:end]
        
                benchmark_data = ast.literal_eval(data_str)
        
                if not all(isinstance(entry, list) and len(entry) == 4 for entry in benchmark_data):
                    raise ValueError("Each entry must be a list of 4 values.")
            except Exception as e:
                raise ValueError(f"Failed to parse benchmark data: {e}")
    
        
    
        
        benchmark_data = np.array(benchmark_data)
        total_entries = len(benchmark_data)
        num_chunks = total_entries // thread_count
    
        average_total_times = []
        average_worst_times = []
    
        for i in range(num_chunks):
            chunk = benchmark_data[i * thread_count : (i + 1) * thread_count]
    
            total_times = chunk[:, 0] + chunk[:, 1] + chunk[:, 2]
    
            avg_total = np.mean(total_times)
            average_total_times.append(avg_total)
    
            worst_time = np.max(total_times)
            average_worst_times.append(worst_time)


        
        average_total_times_2.append(np.mean(average_total_times))
        average_worst_times_2.append(np.mean(average_worst_times))
        
        average_total_times_2_std.append(np.std(average_total_times))
        average_worst_times_2_std.append(np.std(average_worst_times))
        
    return (
        np.array(average_total_times_2),
        np.array(average_worst_times_2),
        np.array(average_total_times_2_std),
        np.array(average_worst_times_2_std)
    )



def calculate_avg_and_worst_times_components(path, filename, thread_counts):
    # Each of these will become lists of 3-element arrays (one for each component)
    average_component_times = []
    worst_component_times = []
    average_component_times_std = []
    worst_component_times_std = []

    for thread_count in thread_counts:
        full_path = merge_paths("BEASTGPU", "data", "GPU", path, filename + "_" + str(thread_count) + ".txt")
        
        with open(full_path, "r") as f:
            content = f.read()
            try:
                start = content.find('[[')
                end = content.rfind(']]') + 2
        
                if start == -1 or end == -1:
                    raise ValueError("No valid data list found in the file.")
        
                data_str = content[start:end]
                benchmark_data = ast.literal_eval(data_str)
        
                if not all(isinstance(entry, list) and len(entry) == 4 for entry in benchmark_data):
                    raise ValueError("Each entry must be a list of 4 values.")
            except Exception as e:
                raise ValueError(f"Failed to parse benchmark data: {e}")

        benchmark_data = np.array(benchmark_data)
        total_entries = len(benchmark_data)
        num_chunks = total_entries // thread_count

        component_avg_times = [[], [], []]  # For chunk[:, 0], chunk[:, 1], chunk[:, 2]
        component_worst_times = [[], [], []]

        for i in range(num_chunks):
            chunk = benchmark_data[i * thread_count : (i + 1) * thread_count]

            for j in range(3):  # 0, 1, 2 components
                comp_times = chunk[:, j]
                component_avg_times[j].append(np.mean(comp_times))
                component_worst_times[j].append(np.max(comp_times))

        # Compute mean and std across chunks for each component
        avg_times = [np.mean(component_avg_times[j]) for j in range(3)]
        worst_times = [np.mean(component_worst_times[j]) for j in range(3)]
        avg_times_std = [np.std(component_avg_times[j]) for j in range(3)]
        worst_times_std = [np.std(component_worst_times[j]) for j in range(3)]

        average_component_times.append(avg_times)
        worst_component_times.append(worst_times)
        average_component_times_std.append(avg_times_std)
        worst_component_times_std.append(worst_times_std)

    return (
        np.array(average_component_times),       # shape: (len(thread_counts), 3)
        np.array(worst_component_times),         # shape: (len(thread_counts), 3)
        np.array(average_component_times_std),   # shape: (len(thread_counts), 3)
        np.array(worst_component_times_std)      # shape: (len(thread_counts), 3)
    )








































from statistics import mean
from typing import Dict, Any, List, Optional



def parse_julia_timings(text: str) -> Dict[str, Any]:
    # Scalar patterns: single float values
    scalar_patterns = {
        "time_all": r"time_all\s*=\s*([\d.]+)",
        "calculate_the_double_int": r"calculate the double int\s+([\d.]+)",
        "calculate_SauterSchwab": r"calculate SauterSchwab\s+([\d.]+)",
        "time_to_determin_quadrule": r"time to determin the quadrule\s+([\d.]+)",
        "create_results_complex": r"create results as complex numbers\s+([\d.]+)",
        "calc_sauter_schwab_2": r"calc_sauter_schwab 2\s+([\d.]+)",
        "calc_sauter_schwab_3": r"calc_sauter_schwab 3\s+([\d.]+)",
        "calc_sauter_schwab_4": r"calc_sauter_schwab 4\s+([\d.]+)",
        "time_to_store": r"time_to_store\s+([\d.]+)",
        "time_sauter_schwab_overhead_2": r"time_sauter_schwab_overhead_and_test_toll 2\s+([\d.]+)",
        "time_sauter_schwab_overhead_3": r"time_sauter_schwab_overhead_and_test_toll 3\s+([\d.]+)",
        "time_sauter_schwab_overhead_4": r"time_sauter_schwab_overhead_and_test_toll 4\s+([\d.]+)",
        "time_overhead": r"time overhead\s+([\d.]+)",
        "transfer_results_to_CPU": r"transfer results to CPU\s+([\d.]+)",
    }

    # Vector patterns: 4-element lists
    vector_patterns = {
        "time_table_1": r"time_table\[1,:\]\s+\[\[([\d.,\s]+)\]\]",
        "time_table_2": r"time_table\[2,:\]\s+\[\[([\d.,\s]+)\]\]",
    }

    result: Dict[str, Any] = {}

    # Process scalars
    for label, pattern in scalar_patterns.items():
        matches = re.findall(pattern, text)
        values = [float(m) for m in matches]
        result[label] = mean(values) if values else None

    # Process vectors
    for label, pattern in vector_patterns.items():
        matches = re.findall(pattern, text)
        vectors: List[List[float]] = []
        for match in matches:
            nums = [float(v.strip()) for v in match.split(",")]
            if len(nums) == 4:
                vectors.append(nums)

        if vectors:
            # Transpose and compute element-wise averages
            cols = list(zip(*vectors))
            result[label] = [mean(col) for col in cols]
        else:
            result[label] = None

    return result


def read_and_avg_pipeline(filename):
    file_path = os.path.join("BEASTGPU", "data", "GPU_full", filename)
    with open(file_path, "r") as f:
        text = f.read()
    
    results = parse_julia_timings(text)
    
    def fmt(val):
        if isinstance(val, list):
            return "[" + " ".join(f"{x:.3f}" for x in val) + "]"
        elif val is not None:
            return f"{val:.3f}"
        else:
            return "N/A"

    rows = [
        ("overhead", results.get("time_overhead")),
        ("calculate quadrule", results.get("time_to_determin_quadrule")),
        ("calculate double integral", results.get("calculate_the_double_int")),
        ("store to CPU", results.get("time_to_store")),
        ("transfer results to CPU", results.get("transfer_results_to_CPU")),
        ("create complex results", results.get("create_results_complex")),
        ("calculate Sauter-Schwab", [
            results.get("calc_sauter_schwab_2"),
            results.get("calc_sauter_schwab_3"),
            results.get("calc_sauter_schwab_4")
        ]),
        ("Sauter-Schwab overhead", [
            results.get("time_sauter_schwab_overhead_2"),
            results.get("time_sauter_schwab_overhead_3"),
            results.get("time_sauter_schwab_overhead_4")
        ]),
        ("time_table[1,:]", results.get("time_table_1")),
        ("time_table[2,:]", results.get("time_table_2")),
        ("calculate SauterSchwab", results.get("calculate_SauterSchwab")),
        ("total time", results.get("time_all")),
    ]

    print(r"\begin{table}[H]")
    print(r"  \centering")
    print(r"  \caption{Breakdown of steps of fully storing on GPU}")
    print(r"  \begin{tabular}{l r}")
    print(r"    \toprule")
    print(r"    Step & Time (s) \\")
    print(r"    \midrule")

    for name, value in rows:
        print(f"    {name} & {fmt(value)} \\\\")

    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"  \label{tab:Breakdown of steps of fully storing on GPU}")
    print(r"\end{table}")







