import time
from multiprocessing import Pool, cpu_count

def cpu_work(n):
    """CPU-bound work: sum of squares"""
    total = 0
    for i in range(n):
        total += i * i
    return total

if __name__ == '__main__':
    print(f"CPU count: {cpu_count()}")

    iterations = 10_000_000
    tasks = [iterations] * 8

    # Sequential
    print("\nRunning sequential...")
    start = time.time()
    seq_results = [cpu_work(t) for t in tasks]
    seq_time = time.time() - start

    # Parallel
    print("Running parallel with 4 workers...")
    start = time.time()
    with Pool(4) as p:
        par_results = p.map(cpu_work, tasks)
    par_time = time.time() - start

    # Verify results match
    results_match = seq_results == par_results

    print(f"\nSequential: {seq_time:.2f}s")
    print(f"Parallel (4 workers): {par_time:.2f}s")
    print(f"Speedup: {seq_time/par_time:.2f}x")
    print(f"Results match: {results_match}")

    if par_time < seq_time and results_match:
        print("\n✓ MULTIPROCESSING IS WORKING - parallel is faster!")
    else:
        print("\n✗ MULTIPROCESSING MAY NOT BE WORKING - parallel is not faster")
