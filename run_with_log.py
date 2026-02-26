"""
Windows-compatible job launcher: runs a Python module and writes all output to a log file.
Usage: python run_with_log.py <log_file> <module> [args...]
Example: python run_with_log.py logs/job.log v15.validation.combined_backtest --config mtf_conflict
"""
import sys
import runpy
import os

if len(sys.argv) < 3:
    print("Usage: run_with_log.py <log_file> <module> [args...]")
    sys.exit(1)

log_file = sys.argv[1]
module   = sys.argv[2]
new_argv = [module] + sys.argv[3:]

os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)

with open(log_file, 'w', encoding='utf-8', buffering=1) as f:
    sys.stdout = f
    sys.stderr = f
    sys.argv = new_argv
    try:
        runpy.run_module(module, run_name='__main__', alter_sys=True)
    except SystemExit as e:
        pass
    except Exception as e:
        import traceback
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
