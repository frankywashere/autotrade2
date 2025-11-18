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

    while True:
        try:
            # Get next task
            task = task_queue.get(timeout=0.5)
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
                # Create DataFrame for resampling
                df_minimal = pd.DataFrame(ohlcv_data, index=pd.DatetimeIndex(timestamps))

                # Resample to target timeframe
                resampled = df_minimal.resample(tf_rule).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()

                n = len(timestamps)
                prefix = f'{symbol}_channel_{tf_name}'

                # Initialize result arrays
                results = {
                    f'{prefix}_position': np.zeros(n, dtype=np.float32),
                    f'{prefix}_upper_dist': np.zeros(n, dtype=np.float32),
                    f'{prefix}_lower_dist': np.zeros(n, dtype=np.float32),
                    f'{prefix}_slope': np.zeros(n, dtype=np.float32),
                    f'{prefix}_slope_pct': np.zeros(n, dtype=np.float32),
                    f'{prefix}_stability': np.zeros(n, dtype=np.float32),
                    f'{prefix}_ping_pongs': np.zeros(n, dtype=np.float32),
                    f'{prefix}_ping_pongs_0_5pct': np.zeros(n, dtype=np.float32),
                    f'{prefix}_ping_pongs_1_0pct': np.zeros(n, dtype=np.float32),
                    f'{prefix}_ping_pongs_3_0pct': np.zeros(n, dtype=np.float32),
                    f'{prefix}_r_squared': np.zeros(n, dtype=np.float32),
                    f'{prefix}_is_bull': np.zeros(n, dtype=np.float32),
                    f'{prefix}_is_bear': np.zeros(n, dtype=np.float32),
                    f'{prefix}_is_sideways': np.zeros(n, dtype=np.float32),
                    f'{prefix}_duration': np.zeros(n, dtype=np.float32)
                }

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

                # Calculate channels with progress updates
                base_lookback = min(168, len(resampled) // 2)
                total_bars = len(resampled) - base_lookback

                # Send total bars count
                progress_queue.put({
                    'worker_id': worker_id,
                    'task_name': task_name,
                    'task_idx': task_idx,
                    'status': 'update',
                    'current': 0,
                    'total': total_bars,
                    'message': f'Processing {total_bars} bars'
                })

                # Process each bar with progress updates
                for idx, i in enumerate(range(base_lookback, len(resampled))):
                    # Send progress update every 5 bars or at milestones
                    if idx % 5 == 0 or idx == total_bars - 1:
                        progress_queue.put({
                            'worker_id': worker_id,
                            'task_name': task_name,
                            'task_idx': task_idx,
                            'status': 'update',
                            'current': idx,
                            'total': total_bars
                        })

                    try:
                        # Get available data
                        available_window = resampled.iloc[:i]

                        # Calculate dynamic lookback
                        from src.ml.features import TradingFeatureExtractor
                        extractor = TradingFeatureExtractor()
                        dynamic_lookback = extractor._calculate_dynamic_window(
                            available_window['close'],
                            base_window=min(168, len(available_window) // 2),
                            min_window=30,
                            max_window=min(300, len(available_window) // 2)
                        )

                        # Find optimal channel
                        channel = channel_calc.find_optimal_channel_window(
                            available_window,
                            timeframe=tf_name,
                            max_lookback=dynamic_lookback,
                            min_ping_pongs=3
                        )

                        if channel is None:
                            continue

                        # Calculate features (simplified for brevity)
                        current_price = resampled['close'].iloc[i]
                        position_data = channel_calc.get_channel_position(current_price, channel)

                        # Get the actual window used
                        actual_window = resampled.iloc[i-channel.actual_duration:i]

                        # Calculate multi-threshold ping-pongs
                        window_prices = actual_window['close'].values
                        multi_pp = channel_calc._detect_ping_pongs_multi_threshold(
                            window_prices,
                            channel.upper_line,
                            channel.lower_line,
                            thresholds=[0.005, 0.01, 0.02, 0.03]
                        )

                        # Map to original timestamps
                        timestamp = resampled.index[i]
                        original_timestamps = pd.DatetimeIndex(timestamps)

                        if i < len(resampled) - 1:
                            next_timestamp = resampled.index[i + 1]
                            mask = (original_timestamps >= timestamp) & (original_timestamps < next_timestamp)
                        else:
                            mask = original_timestamps >= timestamp

                        # Store results
                        indices = np.where(mask)[0]
                        for idx_orig in indices:
                            results[f'{prefix}_position'][idx_orig] = position_data['position']
                            results[f'{prefix}_upper_dist'][idx_orig] = position_data['distance_to_upper_pct']
                            results[f'{prefix}_lower_dist'][idx_orig] = position_data['distance_to_lower_pct']
                            results[f'{prefix}_slope'][idx_orig] = channel.slope

                            slope_pct = (channel.slope / current_price) * 100 if current_price > 0 else 0.0
                            results[f'{prefix}_slope_pct'][idx_orig] = slope_pct
                            results[f'{prefix}_is_bull'][idx_orig] = float(slope_pct > 0.1)
                            results[f'{prefix}_is_bear'][idx_orig] = float(slope_pct < -0.1)
                            results[f'{prefix}_is_sideways'][idx_orig] = float(abs(slope_pct) <= 0.1)

                            results[f'{prefix}_stability'][idx_orig] = channel.stability_score if hasattr(channel, 'stability_score') else 0.0
                            results[f'{prefix}_r_squared'][idx_orig] = channel.r_squared
                            results[f'{prefix}_duration'][idx_orig] = channel.actual_duration

                            results[f'{prefix}_ping_pongs_0_5pct'][idx_orig] = multi_pp[0.005]
                            results[f'{prefix}_ping_pongs_1_0pct'][idx_orig] = multi_pp[0.01]
                            results[f'{prefix}_ping_pongs'][idx_orig] = multi_pp[0.02]
                            results[f'{prefix}_ping_pongs_3_0pct'][idx_orig] = multi_pp[0.03]

                    except Exception:
                        continue

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

            except Exception as e:
                # Send error status
                progress_queue.put({
                    'worker_id': worker_id,
                    'task_name': task_name,
                    'task_idx': task_idx,
                    'status': 'error',
                    'message': str(e)
                })
                result_queue.put((task_idx, results))

        except:
            # Timeout - check again
            continue

    # Worker done
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
    workers = []
    for i in range(n_jobs):
        p = Process(target=channel_worker_with_progress,
                   args=(task_queue, result_queue, progress_queue, i))
        p.start()
        workers.append(p)

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

            # Process progress updates
            while workers_done < n_jobs or len(completed_tasks) < len(tasks):
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
                        # Mark task as errored
                        if task_name in task_ids:
                            progress.update(
                                task_ids[task_name],
                                description=f"[red]✗ {task_name}: {update.get('message', 'Error')}"
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

    # Wait for all workers
    for worker in workers:
        worker.join()

    # Collect and sort results
    results_dict = {}
    while not result_queue.empty():
        task_idx, result = result_queue.get()
        results_dict[task_idx] = result

    # Return results in original order
    return [results_dict[i] for i in range(len(tasks))]