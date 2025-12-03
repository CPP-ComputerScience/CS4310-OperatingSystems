"""
Project 1: Simulating Job Scheduler and Performance Analysis
Author: Devin Khun
Date: October 30, 2025
Description:
    Implements and tests four scheduling algorithms:
        1. First-Come-First-Serve (FCFS)
        2. Shortest-Job-First (SJF)
        3. Round-Robin with Time Slice = 2 (RR-2)
        4. Round-Robin with Time Slice = 5 (RR-5)
"""

# --------------------
# Utility Functions
# --------------------
import os
import random
from statistics import mean
from collections import deque
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import pandas as pd

# Define Job type
Job = Tuple[str, int]

def calc_avg_turnaround(finish_times: Dict[str, int], jobs: List[Tuple[str, int]]) -> float:
    """
    Calculate average turnaround time given finish times and job list
    """
    total_turnaround = 0
    for job, burst in jobs:
        total_turnaround += finish_times[job]
    return total_turnaround / len(jobs)

def parse_jobs_file(path: str) -> List[Job]:
    """
    Parses a jobs file and returns a list of (job_name, burst_time) as tuples.
    """
    jobs: List[Job] = []
    pending_name: str | None = None

    with open(path, "r", encoding="utf-8") as f:
        for ln, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.replace(",", " ").split()]

            try:
                if pending_name is not None:
                    # Expect a burst on this line
                    burst = int(parts[0])
                    jobs.append((pending_name, burst))
                    pending_name = None
                    continue

                if len(parts) == 1:
                    # Could be just a burst OR just a name
                    try:
                        burst = int(parts[0])  # it's a burst
                        jobs.append((f"Job{len(jobs) + 1}", burst))
                    except ValueError:
                        # it's a name; take burst from the next non-empty line
                        pending_name = parts[0]
                else:
                    # name + burst on one line
                    name, burst_str = parts[0], parts[1]
                    burst = int(burst_str)
                    jobs.append((name, burst))

            except ValueError as e:
                raise ValueError(f"Parse error in {path} at line {ln}: {e}") from e

    if pending_name is not None:
        raise ValueError(f"Incomplete job at end of file {path}: got name '{pending_name}' but no burst.")
    
    if not jobs:
        raise ValueError(f"No jobs found in file: {path}")

    return jobs


# --------------------
# First-Come-First-Serve (FCFS)
# --------------------
def fcfs(jobs: List[Tuple[str, int]]) -> Tuple[List[Tuple[str, int, int, int]], float]:
    """
    First-Come-First-Serve Scheduling Algorithm
    """
    time = 0
    log = []
    finish_times = {}

    for job, burst in jobs:
        start = time
        time += burst
        finish_times[job] = time
        log.append((job, start, time, 0))
    
    # Calculate average turnaround time
    avg_turnaround = calc_avg_turnaround(finish_times, jobs)
    return log, avg_turnaround


# --------------------
# Shortest-Job-First (SJF)
# --------------------
def sjf(jobs: List[Tuple[str, int]]) -> Tuple[List[Tuple[str, int, int, int]], float]:
    """
    Shortest-Job-First Scheduling Algorithm
    """
    time = 0
    log = []
    finish_times = {}
    remaining = {job: burst for job, burst in jobs}
    index = {job: i for i, (job, _) in enumerate(jobs)} # file order index
    undone = set(job for job, _ in jobs)
    
    while undone:
        # Pick the job with the smallest remaining time,
        # if tie, smaller file order index
        job = min(undone, key=lambda j: (remaining[j], index[j]))
        burst = remaining[job]
        start = time
        time += burst
        finish_times[job] = time
        undone.remove(job)
        log.append((job, start, time, 0))
    
    # Calculate average turnaround time
    avg_turnaround = calc_avg_turnaround(finish_times, jobs)
    return log, avg_turnaround


# --------------------
# Round Robin (RR)
# --------------------
def round_robin(jobs: List[Tuple[str, int]], time_slice: int) -> Tuple[List[Tuple[str, int, int, int]], float]:
    """
    Round Robin Scheduling Algorithm with given time slice
    """
    time = 0
    log = []
    finish_times = {}
    remaining = {job: burst for job, burst in jobs}
    queue = deque(job for job, _ in jobs)

    while queue:
        job = queue.popleft()
        if remaining[job] == 0:
            continue
        ran = min(time_slice, remaining[job])
        start = time
        time += ran
        remaining[job] -= ran
        log.append((job, start, time, remaining[job]))

        if remaining[job] > 0:
            queue.append(job)
        else:
            finish_times[job] = time
    
    # Calculate average turnaround time
    avg_turnaround = calc_avg_turnaround(finish_times, jobs)
    return log, avg_turnaround

# Round Robin with Time Slice = 2 (RR-2)
def rr2(jobs: List[Tuple[str, int]]) -> Tuple[List[Tuple[str, int, int, int]], float]:
    """
    Round Robin Scheduling Algorithm with time slice = 2
    """
    return round_robin(jobs, 2)

# Round Robin with Time Slice = 5 (RR-5)
def rr5(jobs: List[Tuple[str, int]]) -> Tuple[List[Tuple[str, int, int, int]], float]:
    """
    Round Robin Scheduling Algorithm with time slice = 5
    """
    return round_robin(jobs, 5)


# --------------------
# Testing
# --------------------

# ---------- Generators ----------
def generate_jobs(n: int, max_burst: int = 30, rng: random.Random | None = None) -> List[Job]:
    """
    Creates a random set of n jobs with burst times in [1, max_burst].
    """
    rng = rng or random
    return [(f"Job{i+1}", rng.randint(1, max_burst)) for i in range(n)]

def generate_job_files(sizes=(5, 10, 15), trials=20, max_burst=30, base_seed=1000, out_dir="jobs") -> None:
    """
    Generates random job files for each input size n and trial t.
    Each file will be named job{n}-trial{t}.txt and saved in the output directory.
    """
    os.makedirs(out_dir, exist_ok=True)
    for i, n in enumerate(sizes):
        rng = random.Random(base_seed + i)  # reproducible seed per size
        for t in range(1, trials + 1):
            jobs = generate_jobs(n, max_burst=max_burst, rng=rng)
            filename = f"job{n}-trial{t}.txt"
            path = os.path.join(out_dir, filename)
            with open(path, "w", encoding="utf-8") as f:
                for job_name, burst in jobs:
                    f.write(f"{job_name}\n{burst}\n")

def run_one_trial_all_algorithms(jobs: List[Job]) -> Dict[str, float]:
    """
    Runs all four scheduling algorithms on the same trial set of jobs
    Returns their average turnaround times
    """
    _, a_fcfs = fcfs(jobs)
    _, a_sjf  = sjf(jobs)
    _, a_rr2  = rr2(jobs)
    _, a_rr5  = rr5(jobs)
    return {"FCFS": a_fcfs, "SJF": a_sjf, "RR-2": a_rr2, "RR-5": a_rr5}

# ---------- Test Trials ----------
def run_trials_for_size(n: int, trials: int = 20, max_burst: int = 30, seed: int = 42):
    """
    Generate 20 trials for a given input size n, run all algorithms,
    and return both detailed per-trial results and the summary averages.
    """
    rng = random.Random(seed) # reproducible for this size
    detailed = [] # list of dicts per trial
    for t in range(1, trials + 1):
        jobs = generate_jobs(n, max_burst=max_burst, rng=rng)
        res = run_one_trial_all_algorithms(jobs)
        detailed.append({"trial": t, **res})
    
    summary = {
        "size": n,
        "FCFS": mean([r["FCFS"] for r in detailed]),
        "SJF":  mean([r["SJF"]  for r in detailed]),
        "RR-2": mean([r["RR-2"] for r in detailed]),
        "RR-5": mean([r["RR-5"] for r in detailed]),
        "trials": trials,
        "burst_range": f"1..{max_burst}",
    }
    return summary, detailed

def run_trials_for_size_from_files(n: int, trials: int, dir_path: str, filename_pattern: str = "job{n}-trial{t}.txt"):
    """
    For a given input size n, read trial job files from directory using filename,
    run all algorithms per trial, and return summary results.
    """
    detailed = []  # list of dicts per trial
    for t in range(1, trials + 1):
        path = os.path.join(dir_path, filename_pattern.format(n=n, t=t))
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing file for size {n}, trial {t}: {path}"
            )
        jobs = parse_jobs_file(path)
        # Check the file has exactly n jobs
        if len(jobs) != n:
            raise ValueError(
                f"{os.path.basename(path)} has {len(jobs)} jobs, expected {n}"
            )
        res = run_one_trial_all_algorithms(jobs)
        detailed.append({"trial": t, **res})
    
    summary = {
        "size": n,
        "FCFS": mean([r["FCFS"] for r in detailed]),
        "SJF":  mean([r["SJF"]  for r in detailed]),
        "RR-2": mean([r["RR-2"] for r in detailed]),
        "RR-5": mean([r["RR-5"] for r in detailed]),
        "trials": trials,
        "burst_range": f"from files",
    }
    return summary, detailed

def run_all_sizes(sizes=(5, 10, 15), trials=20, max_burst=30, base_seed=100):
    """
    Runs the experiment for each input size. Uses a different RNG stream per size
    by offsetting the base seed, keeping things reproducible.
    Returns (summaries, details_by_size).
    """
    summaries = []
    details_by_size = {}
    for i, n in enumerate(sizes):
        summary, detailed = run_trials_for_size(
            n=n,
            trials=trials,
            max_burst=max_burst,
            seed=base_seed + i # distinct stream per size
        )
        summaries.append(summary)
        details_by_size[n] = detailed
    return summaries, details_by_size

def run_all_sizes_from_files(sizes=(5, 10, 15), trials=20, dir_path=".", filename_pattern: str = "job{n}-trial{t}.txt"):
    """
    Reads every (size, trial) jobs file and runs all algorithms.
    Returns (summaries, details_by_size).
    """
    summaries = []
    details_by_size = {}
    for n in sizes:
        summary, detailed = run_trials_for_size_from_files(
            n=n,
            trials=trials,
            dir_path=dir_path,
            filename_pattern=filename_pattern
        )
        summaries.append(summary)
        details_by_size[n] = detailed
    return summaries, details_by_size

# ---------- Summary Table of Trials ----------
def print_summary_table(summaries: List[Dict[str, float]]) -> None:
    """
    Prints a summary table of the average turnaround times for each algorithm
    """
    header = (
        f"{'Input Size n jobs':<16}"
        f"{'FCFS (avg of avgs)':>22}"
        f"{'SJF (avg of avgs)':>20}"
        f"{'RR-2 (avg of avgs)':>22}"
        f"{'RR-5 (avg of avgs)':>22}"
    )
    print(header)
    print("-" * len(header))
    for s in sorted(summaries, key=lambda x: x["size"]):
        print(
            f"{str(s['size']) + ' jobs':<16}"
            f"{s['FCFS']:>22.3f}"
            f"{s['SJF']:>20.3f}"
            f"{s['RR-2']:>22.3f}"
            f"{s['RR-5']:>22.3f}"
        )


# --------------------
# Graph Plots
# --------------------
def collect_results_df(sizes=(5, 10, 15), trials=20, max_burst=30, base_seed=2025) -> pd.DataFrame:
    """
    Runs all 20 trials for each job size input and organizes results into a DataFrame.
    columns: size, FCFS, SJF, RR-2, RR-5
    """
    summaries, _ = run_all_sizes(
        sizes=sizes, trials=trials, max_burst=max_burst, base_seed=base_seed
    )
    df = pd.DataFrame(summaries)[["size", "FCFS", "SJF", "RR-2", "RR-5"]]
    return df.sort_values("size")

def plot_per_algorithm(df: pd.DataFrame, filename: str = "alg"):
    """
    Plots average turnaround times vs. input size for each algorithm.
    """
    sizes = df["size"].tolist()
    for alg in ["FCFS", "SJF", "RR-2", "RR-5"]:
        plt.figure()
        plt.plot(sizes, df[alg].tolist(), marker="o")
        plt.title(f"{alg}: Average Turnaround vs. Input Size")
        plt.xlabel("Number of Jobs")
        plt.ylabel("Average Turnaround Time")
        plt.grid(True, alpha=0.25)
        plt.savefig(f"{filename}-{alg.lower()}.png", bbox_inches="tight")
        plt.close()

def plot_all_algorithms(df: pd.DataFrame, filename: str = "alg-all.png"):
    """
    Plots average turnaround times vs. input size for all algorithms.
    """
    sizes = df["size"].tolist()
    plt.figure()
    for alg in ["FCFS", "SJF", "RR-2", "RR-5"]:
        plt.plot(sizes, df[alg].tolist(), marker="o", label=alg)
    plt.title("Average Turnaround vs. Input Size (20-trial averages)")
    plt.xlabel("Number of Jobs")
    plt.ylabel("Average Turnaround Time")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

def rank_algorithms(df: pd.DataFrame) -> list[tuple[str, float]]:
    """
    Finds the mean average turnaround time across all input sizes for each algorithm.
    Returns a sorted list of algorithms from best to worst.
    """
    means = {alg: df[alg].mean() for alg in ["FCFS", "SJF", "RR-2", "RR-5"]}
    return sorted(means.items(), key=lambda kv: kv[1])

def run_and_plot(df: pd.DataFrame, sizes=(5, 10, 15), trials=20, max_burst=30, base_seed=2025) -> None:
    """
    Combines all functions above: runs trials, collects results, graphs plots, and prints ranking.
    """
    print("\n20-trial averages per size (Dataframe):")
    print(df.to_string(index=False))

    plot_per_algorithm(df, filename="alg")
    plot_all_algorithms(df, filename="alg-all.png")

    ranking = rank_algorithms(df)
    print("\nOverall Ranking (mean of average turnaround times across sizes):")
    for i, (alg, val) in enumerate(ranking, 1):
        print(f"{i}. {alg}: {val:.3f}")


# --------------------
# Example Testing
# --------------------
if __name__ == "__main__":
    # Simple test case for each job size
    jobs = {5: [("Job1", 7), ("Job2", 18), ("Job3", 10), ("Job4", 4), ("Job5", 12)],
            10: [("Job1", 3), ("Job2", 8), ("Job3", 2), ("Job4", 14), ("Job5", 7),
                 ("Job6", 5), ("Job7", 9), ("Job8", 4), ("Job9", 12), ("Job10", 6)],
            15: [("Job1", 10), ("Job2", 1), ("Job3", 8), ("Job4", 20), ("Job5", 5),
                 ("Job6", 3), ("Job7", 12), ("Job8", 7), ("Job9", 15), ("Job10", 4),
                 ("Job11", 9), ("Job12", 11), ("Job13", 6), ("Job14", 2), ("Job15", 13)]}
    
    # Run each algorithm on each job size and print results
    scheduling_algorithms = [
        ("FCFS", fcfs),
        ("SJF", sjf),
        ("RR-2", rr2),
        ("RR-5", rr5)
    ]
    
    for size in jobs:
        for name, func in scheduling_algorithms:
            print(f"\n--- Testing {name} Scheduling Algorithm with {size} jobs ---")
            log, avg = func(jobs[size])
            for entry in log:
                print(f" {entry}")
            print(f"Average Turnaround Time: {avg:.2f}")
    
    # Run full 20 trials and print summary table
    summaries, details = run_all_sizes(
        sizes=(5, 10, 15),
        trials=20,
        max_burst=30,
        base_seed=2025
    )
    print_summary_table(summaries)

    # Generate plots of average turnaround times for each algorithm
    df = collect_results_df(sizes=(5, 10, 15), trials=20, max_burst=30, base_seed=2025)
    run_and_plot(df, sizes=(5, 10, 15), trials=20, max_burst=30, base_seed=2025)

    # ---------- Performance Analysis ----------

    # Generate job files for each size and trial
    generate_job_files(
        sizes=(5, 10, 15),
        trials=20,
        max_burst=30,
        base_seed=1000,
        out_dir="./jobs"
    )
    
    # Run trials from generated files
    summaries, details = run_all_sizes_from_files(
        sizes=(5, 10, 15),
        trials=20,
        dir_path="./jobs",
        filename_pattern="job{n}-trial{t}.txt"
    )
    
    df = pd.DataFrame(summaries)[["size", "FCFS", "SJF", "RR-2", "RR-5"]]
    run_and_plot(df, sizes=(5, 10, 15), trials=20, max_burst=30, base_seed=1000)