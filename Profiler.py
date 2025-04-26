#====================================================
# Project: Racing Line AI
# Authors: Spencer Epp, Samuel Trepac
# Date:    March 23rd - April 28th
#
# Description:
#     A toolset for extracting and parsing track metadata, AI racing lines, 
#     and surface geometry from Assetto Corsa track files (.kn5, .ai).
#
# File Overview:
#     This file provides a simple performance Profiler class for measuring 
#     execution time of labeled code blocks and functions. Includes a 
#     decorator for easy profiling of any function call.
#
# Functions Included:
#     - Profiler: Class for timing code sections manually.
#     - profiled(): Decorator to automatically time function execution.
#====================================================


# === Imports ===
import time


# === Profiler Class Definition ===
"""
    Simple profiling tool to measure code execution time by labeled sections.

    Attributes:
        times (dict): Stores total time, call count, and start time for each label.

    Methods:
        start(label): Start timing for a labeled section.
        stop(label): Stop timing for the labeled section and record elapsed time.
        report(): Print a summary of total time, call count, and average time for each label.
"""
class Profiler:
    def __init__(self):
        self.times = {}

    def start(self, label):
        self.times[label] = self.times.get(label, {"total": 0, "count": 0, "start": 0})
        self.times[label]["start"] = time.perf_counter()

    def stop(self, label):
        elapsed = time.perf_counter() - self.times[label]["start"]
        self.times[label]["total"] += elapsed
        self.times[label]["count"] += 1

    def report(self):
        print("\n=== Profiler Summary ===")
        for label, t in self.times.items():
            avg = t["total"] / max(1, t["count"])
            print(f"{label:25} | Total: {t['total']:.2f}s | Calls: {t['count']} | Avg: {avg:.4f}s")


# === Profiler Decorator ===
"""
    Decorator that wraps a function and automatically profiles its execution.

    Args:
        profiler (Profiler): Instance of the Profiler class to use.
        name (str): Label to associate with the function timing.

    Returns:
        function: Wrapped function with profiling enabled.
"""
def profiled(profiler, name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            profiler.start(name)
            result = func(*args, **kwargs)
            profiler.stop(name)
            return result
        return wrapper
    return decorator