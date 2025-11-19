"""
Parallel channel extraction with real-time multi-progress bars.
Each worker has its own progress bar that updates independently.
"""

import multiprocessing as mp
from multiprocessing import Process, Queue, Manager
import time
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import config  # For precision configuration

# Rich imports for beautiful progress bars
try:
    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn, TaskID
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.layout import Layout
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def channel_worker_with_progress(task_queue: Queue, result_queue: Queue, progress_queue: Queue, worker_id: int):
    """
    Worker process that calculates channels and reports detailed progress.

    Args:
        task_queue: Queue containing tasks to process
        result_queue: Queue for results
        progress_queue: Queue for progress updates
        worker_id: Unique ID for this worker
    """
    # Add parent directory to path for imports
    parent_dir = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(parent_dir))

    from src.linear_regression import LinearRegressionChannel

    # Get process ID for debugging
    import os
    pid = os.getpid()

    try:
        empty_count = 0  # Track consecutive empty queue checks

        while True:
            try:
                # Get next task
                task = task_queue.get(timeout=0.5)
                empty_count = 0  # Reset on successful get
            except:
                # Timeout - check if queue is truly empty
                empty_count += 1
                if empty_count > 10:  # ~5 seconds of empty queue → exit
                    print(f"Worker {worker_id}: Exiting after {empty_count} empty queue checks")
                    break
                continue

            if task is None:  # Sentinel value
                break

            # Unpack task
            task_idx, (ohlcv_data, timestamps, tf_name, tf_rule, symbol) = task
            task_name = f"{symbol}_{tf_name}"

            # Send start signal
            progress_queue.put({
                'worker_id': worker_id,
                'task_name': task_name,
                'task_idx': task_idx,
                'status': 'start',
                'current': 0,
                'total': 100,
                'symbol': symbol,
                'timeframe': tf_name
            })

            # Create channel calculator
            channel_calc = LinearRegressionChannel()

            try:
                # Create DataFrame for resampling with explicit column names
                df_minimal = pd.DataFrame(
                    ohlcv_data,
                    index=pd.DatetimeIndex(timestamps),
                    columns=['open', 'high', 'low', 'close', 'volume']
                )

                # Resample to target timeframe
                resampled = df_minimal.resample(tf_rule).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()

                n = len(timestamps)

                # Use centralized window sizes for ALL timeframes (21 windows, no filtering)
                import config
                window_candidates = config.CHANNEL_WINDOW_SIZES

                # Initialize result arrays for EACH window size
                results = {}

                for window in window_candidates:
                    w_prefix = f'{symbol}_channel_{tf_name}_w{window}'

                    # Position features
                    results[f'{w_prefix}_position'] = np.zeros(n, dtype=config.NUMPY_DTYPE)
                    results[f'{w_prefix}_upper_dist'] = np.zeros(n, dtype=config.NUMPY_DTYPE)
                    results[f'{w_prefix}_lower_dist'] = np.zeros(n, dtype=config.NUMPY_DTYPE)

                    # Slope features (OHLC)
                    results[f'{w_prefix}_close_slope'] = np.zeros(n, dtype=config.NUMPY_DTYPE)
                    results[f'{w_prefix}_close_slope_pct'] = np.zeros(n, dtype=config.NUMPY_DTYPE)
                    results[f'{w_prefix}_high_slope'] = np.zeros(n, dtype=config.NUMPY_DTYPE)
                    results[f'{w_prefix}_high_slope_pct'] = np.zeros(n, dtype=config.NUMPY_DTYPE)
                    results[f'{w_prefix}_low_slope'] = np.zeros(n, dtype=config.NUMPY_DTYPE)
                    results[f'{w_prefix}_low_slope_pct'] = np.zeros(n, dtype=config.NUMPY_DTYPE)

                    # R-squared features (OHLC)
                    results[f'{w_prefix}_close_r_squared'] = np.zeros(n, dtype=config.NUMPY_DTYPE)
                    results[f'{w_prefix}_high_r_squared'] = np.zeros(n, dtype=config.NUMPY_DTYPE)
                    results[f'{w_prefix}_low_r_squared'] = np.zeros(n, dtype=config.NUMPY_DTYPE)
                    results[f'{w_prefix}_r_squared_avg'] = np.zeros(n, dtype=config.NUMPY_DTYPE)

                    # Channel metrics
                    results[f'{w_prefix}_channel_width_pct'] = np.zeros(n, dtype=config.NUMPY_DTYPE)
                    results[f'{w_prefix}_slope_convergence'] = np.zeros(n, dtype=config.NUMPY_DTYPE)
                    results[f'{w_prefix}_stability'] = np.zeros(n, dtype=config.NUMPY_DTYPE)

                    # Ping-pongs (multi-threshold)
                    results[f'{w_prefix}_ping_pongs'] = np.zeros(n, dtype=config.NUMPY_DTYPE)
                    results[f'{w_prefix}_ping_pongs_0_5pct'] = np.zeros(n, dtype=config.NUMPY_DTYPE)
                    results[f'{w_prefix}_ping_pongs_1_0pct'] = np.zeros(n, dtype=config.NUMPY_DTYPE)
                    results[f'{w_prefix}_ping_pongs_3_0pct'] = np.zeros(n, dtype=config.NUMPY_DTYPE)

                    # Direction flags
                    results[f'{w_prefix}_is_bull'] = np.zeros(n, dtype=config.NUMPY_DTYPE)
                    results[f'{w_prefix}_is_bear'] = np.zeros(n, dtype=config.NUMPY_DTYPE)
                    results[f'{w_prefix}_is_sideways'] = np.zeros(n, dtype=config.NUMPY_DTYPE)

                    # Quality indicators
                    results[f'{w_prefix}_quality_score'] = np.zeros(n, dtype=config.NUMPY_DTYPE)
                    results[f'{w_prefix}_is_valid'] = np.zeros(n, dtype=config.NUMPY_DTYPE)
                    results[f'{w_prefix}_insufficient_data'] = np.zeros(n, dtype=config.NUMPY_DTYPE)
                    results[f'{w_prefix}_duration'] = np.zeros(n, dtype=config.NUMPY_DTYPE)

                # Total: 21 windows × 28 features = 588 features per (symbol, timeframe) pair

                # Handle insufficient data
                if len(resampled) < 20:
                    progress_queue.put({
                        'worker_id': worker_id,
                        'task_name': task_name,
                        'task_idx': task_idx,
                        'status': 'complete',
                        'current': 100,
                        'total': 100,
                        'message': 'Insufficient data'
                    })
                    result_queue.put((task_idx, results))
                    continue

                # Use OPTIMIZED rolling calculation (45x faster!)
                total_bars = len(resampled)

                progress_queue.put({
                    'worker_id': worker_id,
                    'task_name': task_name,
                    'task_idx': task_idx,
                    'status': 'update',
                    'current': 0,
                    'total': total_bars,
                    'message': f'Processing {total_bars} bars with rolling stats'
                })

                # Calculate all channels for ALL window sizes using rolling statistics
                all_windows_channels = channel_calc.calculate_multi_window_rolling(resampled, tf_name)

                # Debug: Check what windows were calculated
                num_windows = len(all_windows_channels)
                window_sizes = list(all_windows_channels.keys())

                progress_queue.put({
                    'worker_id': worker_id,
                    'task_name': task_name,
                    'task_idx': task_idx,
                    'status': 'update',
                    'current': 0,
                    'total': total_bars,
                    'message': f'Calculated {num_windows} windows: {window_sizes}'
                })

                # Map results to original timestamps with progress updates
                original_timestamps = pd.DatetimeIndex(timestamps)

                # Process each bar and each window
                for i in range(len(resampled)):
                    # Send progress update every 50 bars
                    if i % 50 == 0:
                        progress_queue.put({
                            'worker_id': worker_id,
                            'task_name': task_name,
                            'task_idx': task_idx,
                            'status': 'update',
                            'current': i,
                            'total': total_bars
                        })

                    # Map this resampled bar to original 1-min timestamps
                    timestamp = resampled.index[i]

                    if i < len(resampled) - 1:
                        next_timestamp = resampled.index[i + 1]
                        mask = (original_timestamps >= timestamp) & (original_timestamps < next_timestamp)
                    else:
                        mask = original_timestamps >= timestamp

                    indices = np.where(mask)[0]
                    if len(indices) == 0:
                        continue

                    # Get current price for position calculation
                    current_price = resampled['close'].iloc[i]

                    # Iterate through ALL window sizes
                    for window, channels_list in all_windows_channels.items():
                        channel = channels_list[i]

                        if channel is None:
                            # No data for this window at this bar - features stay as zeros
                            continue

                        # Window-specific prefix
                        w_prefix = f'{symbol}_channel_{tf_name}_w{window}'

                        # Calculate position for this window's channel
                        position_data = channel_calc.get_channel_position(current_price, channel)

                        # Calculate slope percentages
                        close_slope_pct = (channel.close_slope / current_price) * 100 if current_price > 0 else 0.0
                        high_slope_pct = (channel.high_slope / current_price) * 100 if current_price > 0 else 0.0
                        low_slope_pct = (channel.low_slope / current_price) * 100 if current_price > 0 else 0.0

                        # Store ALL features for this window (vectorized - no loop)
                        # Position features
                        results[f'{w_prefix}_position'][indices] = position_data['position']
                        results[f'{w_prefix}_upper_dist'][indices] = position_data['distance_to_upper_pct']
                        results[f'{w_prefix}_lower_dist'][indices] = position_data['distance_to_lower_pct']

                        # Slope features (OHLC)
                        results[f'{w_prefix}_close_slope'][indices] = channel.close_slope
                        results[f'{w_prefix}_close_slope_pct'][indices] = close_slope_pct
                        results[f'{w_prefix}_high_slope'][indices] = channel.high_slope
                        results[f'{w_prefix}_high_slope_pct'][indices] = high_slope_pct
                        results[f'{w_prefix}_low_slope'][indices] = channel.low_slope
                        results[f'{w_prefix}_low_slope_pct'][indices] = low_slope_pct

                        # R-squared features
                        results[f'{w_prefix}_close_r_squared'][indices] = channel.close_r_squared
                        results[f'{w_prefix}_high_r_squared'][indices] = channel.high_r_squared
                        results[f'{w_prefix}_low_r_squared'][indices] = channel.low_r_squared
                        results[f'{w_prefix}_r_squared_avg'][indices] = channel.r_squared

                        # Channel metrics
                        results[f'{w_prefix}_channel_width_pct'][indices] = channel.channel_width_pct
                        results[f'{w_prefix}_slope_convergence'][indices] = channel.slope_convergence
                        results[f'{w_prefix}_stability'][indices] = channel.stability_score

                        # Ping-pongs
                        results[f'{w_prefix}_ping_pongs'][indices] = channel.ping_pongs
                        results[f'{w_prefix}_ping_pongs_0_5pct'][indices] = channel.ping_pongs_0_5pct
                        results[f'{w_prefix}_ping_pongs_1_0pct'][indices] = channel.ping_pongs_1_0pct
                        results[f'{w_prefix}_ping_pongs_3_0pct'][indices] = channel.ping_pongs_3_0pct

                        # Direction flags
                        results[f'{w_prefix}_is_bull'][indices] = float(close_slope_pct > 0.1)
                        results[f'{w_prefix}_is_bear'][indices] = float(close_slope_pct < -0.1)
                        results[f'{w_prefix}_is_sideways'][indices] = float(abs(close_slope_pct) <= 0.1)

                        # Quality indicators
                        results[f'{w_prefix}_quality_score'][indices] = channel.quality_score
                        results[f'{w_prefix}_is_valid'][indices] = channel.is_valid
                        results[f'{w_prefix}_insufficient_data'][indices] = channel.insufficient_data
                        results[f'{w_prefix}_duration'][indices] = channel.actual_duration

                # Send completion
                progress_queue.put({
                    'worker_id': worker_id,
                    'task_name': task_name,
                    'task_idx': task_idx,
                    'status': 'complete',
                    'current': total_bars,
                    'total': total_bars
                })

                # Send results
                result_queue.put((task_idx, results))

                # Critical: Clear memory to prevent accumulation across tasks
                del results
                if 'all_windows_channels' in locals():
                    del all_windows_channels
                if 'resampled' in locals():
                    del resampled
                if 'df_minimal' in locals():
                    del df_minimal
                import gc
                gc.collect()

            except Exception as e:
                # Send error status
                import traceback
                error_trace = traceback.format_exc()
                progress_queue.put({
                    'worker_id': worker_id,
                    'task_name': task_name,
                    'task_idx': task_idx,
                    'status': 'error',
                    'message': f"{str(e)}\n{error_trace}"
                })
                # Return empty results on error
                result_queue.put((task_idx, results if 'results' in locals() else {}))

                # Clear memory even on error
                if 'results' in locals():
                    del results
                if 'all_windows_channels' in locals():
                    del all_windows_channels
                if 'resampled' in locals():
                    del resampled
                if 'df_minimal' in locals():
                    del df_minimal
                import gc
                gc.collect()
    finally:
        # ALWAYS send worker_done signal, even if worker crashes
        progress_queue.put({
            'worker_id': worker_id,
            'status': 'worker_done'
        })


def parallel_channel_extraction_with_multi_progress(tasks: List[Tuple], n_jobs: int = -1) -> List[Dict]:
    """
    Execute channel extraction in parallel with individual progress bars for each task.

    Args:
        tasks: List of (ohlcv_data, timestamps, tf_name, tf_rule, symbol) tuples
        n_jobs: Number of parallel workers (-1 for all cores)

    Returns:
        List of result dictionaries in original task order
    """
    if n_jobs == -1:
        n_jobs = mp.cpu_count()

    # Create queues
    task_queue = Queue()
    result_queue = Queue()
    progress_queue = Queue()

    # Add tasks to queue
    for idx, task in enumerate(tasks):
        task_queue.put((idx, task))

    # Add sentinel values
    for _ in range(n_jobs):
        task_queue.put(None)

    # Start worker processes
    print(f"   🚀 Starting {n_jobs} worker processes...")
    workers = []
    for i in range(n_jobs):
        p = Process(target=channel_worker_with_progress,
                   args=(task_queue, result_queue, progress_queue, i))
        p.start()
        workers.append(p)
    print(f"   ✓ All {n_jobs} workers started")

    if RICH_AVAILABLE:
        # Use rich for beautiful multi-progress display
        console = Console()

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeRemainingColumn(),
            console=console,
            expand=True,
            refresh_per_second=10
        ) as progress:

            # Track tasks
            task_ids = {}  # task_name -> TaskID
            completed_tasks = set()
            workers_done = 0

            # Create overall progress bar
            overall_task = progress.add_task(
                "[bold green]Overall Progress",
                total=len(tasks)
            )

            # Process progress updates with safety timeout
            import time
            start_time = time.time()
            max_wait_seconds = 600  # 10 minutes max

            while workers_done < n_jobs or len(completed_tasks) < len(tasks):
                # Safety timeout
                elapsed = time.time() - start_time
                if elapsed > max_wait_seconds:
                    console.print(f"[red]⚠️ Timeout after {elapsed:.0f}s - {workers_done}/{n_jobs} workers done, {len(completed_tasks)}/{len(tasks)} tasks complete[/red]")
                    break

                try:
                    # Get progress update (with timeout to update display)
                    update = progress_queue.get(timeout=0.1)

                    if update['status'] == 'worker_done':
                        workers_done += 1
                        continue

                    task_name = update.get('task_name', '')

                    if update['status'] == 'start':
                        # Create new progress bar for this task
                        if task_name not in task_ids:
                            task_ids[task_name] = progress.add_task(
                                f"[cyan]{task_name}",
                                total=update['total']
                            )

                    elif update['status'] == 'update':
                        # Update existing progress bar
                        if task_name in task_ids:
                            progress.update(
                                task_ids[task_name],
                                completed=update['current'],
                                total=update['total']
                            )

                    elif update['status'] == 'complete':
                        # Mark task as complete
                        if task_name in task_ids:
                            progress.update(
                                task_ids[task_name],
                                completed=update['total'],
                                description=f"[green]✓ {task_name}"
                            )
                        completed_tasks.add(update.get('task_idx'))
                        progress.update(overall_task, completed=len(completed_tasks))

                    elif update['status'] == 'error':
                        # Mark task as errored and PRINT ERROR
                        error_msg = update.get('message', 'Unknown error')
                        console.print(f"[red]✗ Worker error in {task_name}:[/red]")
                        console.print(f"[red]{error_msg}[/red]")

                        if task_name in task_ids:
                            progress.update(
                                task_ids[task_name],
                                description=f"[red]✗ {task_name} (ERROR)"
                            )
                        completed_tasks.add(update.get('task_idx'))
                        progress.update(overall_task, completed=len(completed_tasks))

                except:
                    # Timeout - just continue to keep display updating
                    continue

    else:
        # Fallback without rich - just print updates
        print(f"Processing {len(tasks)} tasks with {n_jobs} workers...")
        completed = 0
        while completed < len(tasks):
            try:
                update = progress_queue.get(timeout=1.0)
                if update['status'] == 'complete':
                    completed += 1
                    print(f"  ✓ Completed: {update.get('task_name', 'task')} ({completed}/{len(tasks)})")
            except:
                continue

    print(f"\n   🔄 Progress complete. Waiting for {n_jobs} workers to finish...")

    # FIX 5: Better Worker Shutdown - Give more time, collect results BEFORE killing
    for i, worker in enumerate(workers):
        print(f"   ⏳ Waiting for worker {i}...")
        worker.join(timeout=30)  # Increased from 10s to 30s - give workers time to flush queue
        if worker.is_alive():
            print(f"   ⚠️  Worker {i} still alive after 30s - terminating...")
            worker.terminate()  # Force kill stuck worker
            worker.join(timeout=5)  # Give 5s to terminate gracefully
            if worker.is_alive():
                print(f"   ⚠️  Worker {i} won't die - killing...")
                worker.kill()
        else:
            print(f"   ✓ Worker {i} joined successfully")

    print(f"   📦 Collecting results from {len(tasks)} tasks...")

    # FIX 4: Check Queue Size First
    print(f"   🔍 Checking result queue state...")
    try:
        queue_size = result_queue.qsize()
        print(f"   ℹ️  Queue reports ~{queue_size} items available (expected {len(tasks)})")
        if queue_size < len(tasks):
            print(f"   ⚠️  Queue has fewer items than expected! May be missing {len(tasks) - queue_size} results")
    except NotImplementedError:
        print(f"   ℹ️  Queue size checking not available on this platform (macOS limitation)")
    except Exception as e:
        print(f"   ⚠️  Could not check queue size: {e}")

    # FIX 2 & FIX 3: Non-blocking collection with hard timeout
    import time
    from queue import Empty

    results_dict = {}
    timeout_count = 0
    max_consecutive_timeouts = 10  # Allow more retries before giving up
    collection_start_time = time.time()
    HARD_TIMEOUT_SECONDS = 60  # Maximum 60 seconds for entire collection

    print(f"   ⏰ Starting collection with {HARD_TIMEOUT_SECONDS}s hard timeout...")

    collected = 0
    while collected < len(tasks):
        # FIX 3: Check hard timeout
        elapsed = time.time() - collection_start_time
        if elapsed > HARD_TIMEOUT_SECONDS:
            print(f"\n   ⚠️  HARD TIMEOUT after {elapsed:.1f}s! Only collected {collected}/{len(tasks)} results")
            print(f"   ⚠️  Stopping collection to avoid infinite hang")
            break

        # FIX 1: Debug output with timestamp
        print(f"   ⏳ [{elapsed:.1f}s] Waiting for result {collected + 1}/{len(tasks)}...", end='', flush=True)

        try:
            # FIX 2: Non-blocking get
            task_idx, result = result_queue.get_nowait()  # Don't block, return immediately
            results_dict[task_idx] = result
            num_features = len(result) if result else 0
            print(f" ✓ Got task {task_idx} ({num_features} features)")
            collected += 1
            timeout_count = 0  # Reset on success

        except Empty:
            # Queue is empty right now
            timeout_count += 1
            print(f" ⚠️  Empty #{timeout_count}", flush=True)

            if timeout_count >= max_consecutive_timeouts:
                print(f"   ⚠️  {max_consecutive_timeouts} consecutive empties - queue likely exhausted")
                print(f"   ℹ️  Collected {collected}/{len(tasks)} results so far")
                break

            # FIX 1: Debug - show we're still trying
            if timeout_count % 5 == 0:
                print(f"   🔄 Still trying... ({timeout_count} attempts, {elapsed:.1f}s elapsed)")

            # Small sleep to avoid CPU spinning
            time.sleep(0.1)
            continue

        except Exception as e:
            # FIX 1: Log unexpected exceptions with details
            print(f" ❌ Exception: {type(e).__name__}: {e}")
            timeout_count += 1
            if timeout_count >= max_consecutive_timeouts:
                print(f"   ⚠️  Too many exceptions - stopping collection")
                break
            time.sleep(0.1)
            continue

    print(f"   ✓ Collected {len(results_dict)} results")

    # Check for missing results
    missing = [i for i in range(len(tasks)) if i not in results_dict]
    if missing:
        print(f"   ⚠️  Missing {len(missing)} task results: {missing}")
        for idx in missing:
            if idx < len(tasks):
                symbol = tasks[idx][4]
                tf_name = tasks[idx][2]
                print(f"      Missing: Task {idx} = {symbol}_{tf_name}")

    print(f"   🔄 Sorting results...")

    # Return results in original order, using empty dict for missing
    sorted_results = []
    for i in range(len(tasks)):
        if i in results_dict:
            sorted_results.append(results_dict[i])
        else:
            print(f"   ⚠️  Task {i} missing - using empty result")
            sorted_results.append({})  # Empty dict for missing task

    print(f"   ✅ Returning {len(sorted_results)} results")
    return sorted_results