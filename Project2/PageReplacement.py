"""
Project 2 Simulating Page Replacement Manager and Performance Analysis
Author: Devin Khun
Date: December 2, 2025
Description:
    Implements and tests three page replacement algorithms:
        1. First-In-First-Out (FIFO)
        2. Least Recently Used (LRU)
        3. Optimal Page Replacement (OPT)
"""

# --------------------
# Utility Functions
# --------------------
import random
import pandas as pd
import matplotlib.pyplot as plt
from queue import Queue
from collections import defaultdict
from pathlib import Path

# --------------------
# First-In-First-Out (FIFO)
# --------------------
def FIFO(pages, n, capacity):
    """
    FIFO page replacement algorithm.
    Args:
        - pages: List of page references.
        - n: Number of pages.
        - capacity: Number of frames.
    Returns:
        - Number of page faults.
    """
    # To represent the set of current pages
    s = set()
    # To store the pages in FIFO order
    indexes = Queue()
    # Start from initial page
    page_faults = 0
    for i in range(n):
        # Check if the set can hold more pages
        if len(s) < capacity:
            # Insert the page into the set if not present (page fault)
            if (pages[i] not in s):
                s.add(pages[i])
                page_faults += 1
                indexes.put(pages[i])
        # If the set is full then need to perform FIFO, remove the first page of the queue and insert current page
        else:
            # Check if the page is not already present in the set
            if (pages[i] not in s):
                # Pop the first page from the queue
                val = indexes.queue[0]
                indexes.get()
                # Remove the indexes page
                s.remove(val)
                # Insert the current page
                s.add(pages[i])
                # Push the current page into the queue
                indexes.put(pages[i])
                page_faults += 1
    return page_faults

# --------------------
# Least Recently Used (LRU)
# --------------------
def LRU(pages, n, capacity):
    """
    LRU page replacement algorithm.
    Args:
        - pages: List of page references.
        - n: Number of pages.
        - capacity: Number of frames.
    Returns:
        - Number of page faults.
    """
    # To represent the set of current pages
    s = set()
    # To store the least recently used indexes
    indexes = {}
    # Start from initial page
    page_faults = 0
    for i in range(n):
        # Check if the set can hold more pages
        if len(s) < capacity:
            # Insert the page into the set if not present (page fault)
            if (pages[i] not in s):
                s.add(pages[i])
                page_faults += 1
            # Store the recently used index of each page
            indexes[pages[i]] = i
        # If the set is full then need to perform LRU, remove the least recently used and insert current page
        else:
            # Check if the page is not already present in the set
            if (pages[i] not in s):
                # Find the least recently used pages in the set
                lru = float('inf')
                for page in s:
                    if indexes[page] < lru:
                        lru = indexes[page]
                        val = page
                # Remove the indexes page
                s.remove(val)
                # Insert the current page
                s.add(pages[i])
                page_faults += 1
            # Update the current page index
            indexes[pages[i]] = i
    return page_faults

# --------------------
# Optimal Page Replacement (OPT)
# --------------------
def search(key, fr):
    """
    Helper function to check whether a page exists in a frame.
    """
    for i in range(len(fr)):
        if (fr[i] == key):
            return True
    return False

def predict(pages, frames, n, index):
    """
    Helper function to find the pages to be replaced.
    """
    res = -1
    farthest = index
    for i in range(len(frames)):
        j = 0
        for j in range(index, n):
            if (frames[i] == pages[j]):
                if (j > farthest):
                    farthest = j
                    res = i
                break
        # If a page is never referenced in future, return it
        if (j == n):
            return i
    # If all the frames were not in the future, return 0. Otherwise, return res.
    return 0 if (res == -1) else res

def optimalPage(pages, n, capacity):
    """
    Optimal page replacement algorithm.
    Args:
        - pages: List of page references.
        - n: Number of pages.
        - capacity: Number of frames.
    Returns:
        - Number of page faults.
    """
    # Array for given number of frames
    frames = []
    # Start from initial page
    page_faults = 0
    for i in range(n):
        # Page found in a frame
        if search(pages[i], frames):
            continue
        # Page not found in a frame
        if len(frames) < capacity:
            frames.append(pages[i])
        # Find the page to be replaced
        else:
            j = predict(pages, frames, n, i+1)
            frames[j] = pages[i]
        page_faults += 1
    return page_faults

# --------------------
# Performance Testing and Analysis
# --------------------
def generate_testing_file(path="TestingData.txt", num_strings=50, length=30, pages=range(8), seed=None):
    """
    Generates a testing data file with random page reference strings.
    """
    rng = random.Random(seed)
    pages = list(pages)

    if not pages or any(not isinstance(p, int) or p < 0 for p in pages):
        raise ValueError("Pages must be a non-empty range of non-negative integers.")
    if any(p > 9 for p in pages):
        raise ValueError("This file format is digit-only; pages must be 0..9 (project uses 0..7).")\
    
    lines = []
    for _ in range(num_strings):
        seq = "".join(str(rng.choice(pages)) for _ in range(length))
        lines.append(seq)
    
    with open(f"{path}", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def run_one_test():
    """
    Run a single test sequence to verify correctness of algorithms.
    """
    replacement_algorithms = [
        ("FIFO", FIFO),
        ("LRU", LRU),
        ("OPT", optimalPage)
    ]
    
    frame_sizes = [3, 4, 5, 6]
    
    pages = [2, 3, 6, 5, 7, 1, 3, 4, 5, 1, 7, 2, 4, 5]
    n = len(pages)
    print(f"--- Reference String ---")
    print("Sequence:", "".join(map(str, pages)))

    for name, algo in replacement_algorithms:
        for capacity in frame_sizes:
            page_faults = algo(pages, n, capacity)
            print(f"{name} with frame size {capacity}: {page_faults} page faults")

def run_all_tests(file_path="TestingData.txt"):
    """
    Run all sequences from the testing data file and collect number of page faults.
    """
    replacement_algorithms = [
        ("FIFO", FIFO),
        ("LRU", LRU),
        ("OPT", optimalPage)
    ]
    
    frame_sizes = [3, 4, 5, 6]

    with open(file_path, "r", encoding="utf-8") as f:
        reference_strings = [list(map(int, line.strip())) for line in f if line.strip()]
    
    totals = {name: defaultdict(int) for name, _ in replacement_algorithms}
    num_sequences = len(reference_strings)
    
    for seq_index, pages in enumerate(reference_strings):
        n = len(pages)
        print(f"\n--- Reference String {seq_index+1} ---")
        print("Sequence:", "".join(map(str, pages)))

        for name, algo in replacement_algorithms:
            for capacity in frame_sizes:
                page_faults = algo(pages, n, capacity)
                totals[name][capacity] += page_faults
                print(f"{name} with frame size {capacity}: {page_faults} page faults")
    
    summary = {
        algo: {size: {
            "total": totals[algo][size],
            "average": totals[algo][size] / num_sequences
        } for size in frame_sizes}
        for algo, _ in replacement_algorithms
    }

    print("\n--- Summary of Total and Average Page Faults ---")
    for algo, sizes in summary.items():
        print(f"\n{algo}:")
        for size, stats in sizes.items():
            print(f"  Frames={size} | Total={stats['total']} | Avg={stats['average']:.2f}")
    
    return summary

def collect_results_df() -> pd.DataFrame:
    """
    Collects results from all tests into a pandas DataFrame.
    """
    summary = run_all_tests("TestingData.txt")
    df = pd.DataFrame([
        {"Algorithm": algo, "Frames": size, **stats}
        for algo, sizes in summary.items()
        for size, stats in sizes.items()
    ])
    return df

def plot_per_algorithm(df: pd.DataFrame, filename_prefix: str = "Page-alg"):
    """
    Plots average page faults vs. frame size for each algorithm separately.
    """
    for alg, sub in df.groupby("Algorithm"):
        sub = sub.sort_values("Frames")
        x = sub["Frames"].to_numpy()
        y = sub["average"].to_numpy()
        plt.figure()
        plt.plot(x, y, marker="o")
        plt.title(f"{alg}: Average Page Faults vs. Frame Size")
        plt.xlabel("Frame Size")
        plt.ylabel("Average Page Faults")
        plt.grid(True, alpha=0.25)
        plt.savefig(f"{filename_prefix}-{alg.lower()}.png", bbox_inches="tight")
        plt.close()

def plot_all_algorithms(df: pd.DataFrame, filename: str = "Page-alg-all.png"):
    """
    Plots average page faults vs. frame size for all algorithms.
    """
    plt.figure()
    for alg, sub in df.groupby("Algorithm"):
        sub = sub.sort_values("Frames")
        x = sub["Frames"].to_numpy()
        y = sub["average"].to_numpy()
        plt.plot(x, y, marker="o", label=alg)
    plt.title("Average Page Faults vs. Frame Size (50-sequence averages)")
    plt.xlabel("Frame Size")
    plt.ylabel("Average Page Faults")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

def rank_algorithms(df: pd.DataFrame) -> list[tuple[str, float]]:
    """
    Finds the mean average page faults across all frame sizes for each algorithm.
    Returns a sorted list of algorithms from best to worst.
    """
    means = df.groupby("Algorithm")["average"].mean()
    return list(means.sort_values().items())

def run_and_plot(df: pd.DataFrame) -> None:
    """
    Combines all functions above: runs trials, collects results, graphs plots, and prints ranking.
    """
    print("\n50-sequence averages per frame size (Dataframe):")
    print(df.to_string(index=False))

    plot_per_algorithm(df, filename_prefix="Page-alg")
    plot_all_algorithms(df, filename="Page-alg-all.png")

    ranking = rank_algorithms(df)
    print("\nOverall Ranking (mean of average page faults across sizes):")
    for i, (alg, val) in enumerate(ranking, 1):
        print(f"{i}. {alg}: {val:.3f}")

if __name__ == "__main__":
    # Run single test sequence for correctness verification
    run_one_test()

    # Generate testing data file, reproducible set of 50 strings, length 30, pages 0–7
    generate_testing_file("TestingData.txt", num_strings=50, length=30, pages=range(8), seed=42)
    df = collect_results_df()
    
    run_and_plot(df)