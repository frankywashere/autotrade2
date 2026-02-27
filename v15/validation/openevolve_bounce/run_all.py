"""
Launcher for OpenEvolve bounce signal evolution.
Starts the Claude proxy, waits for it, then runs evolution.

Usage:
    python run_all.py                  # default port 5560
    python run_all.py --port 5560
"""

import argparse
import os
import sys
import threading
import time

import requests


def start_proxy(port):
    """Start the Claude proxy in a daemon thread."""
    from claude_proxy import app
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)


def wait_for_proxy(port, timeout=30):
    """Wait until proxy is responding."""
    url = f"http://localhost:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                print(f"  Proxy healthy on port {port}")
                return True
        except Exception:
            pass
        time.sleep(1)
    print(f"  WARNING: Proxy not responding after {timeout}s")
    return False


def main():
    parser = argparse.ArgumentParser(description='OpenEvolve Bounce Signal Launcher')
    parser.add_argument('--port', type=int, default=5560)
    parser.add_argument('--iterations', type=int, default=200)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"OpenEvolve Bounce Signal Evolution")
    print(f"Port: {args.port}, Iterations: {args.iterations}")
    print(f"{'='*60}")

    # Start proxy
    print("\nStarting Claude proxy...")
    proxy_thread = threading.Thread(target=start_proxy, args=(args.port,), daemon=True)
    proxy_thread.start()

    if not wait_for_proxy(args.port):
        print("Failed to start proxy. Exiting.")
        sys.exit(1)

    # Run evolution
    print("\nStarting OpenEvolve...")
    import openevolve.api

    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, 'config.yaml')
    initial_program = os.path.join(base_dir, 'initial_program.py')
    evaluator_path = os.path.join(base_dir, 'evaluator.py')
    output_dir = os.path.join(base_dir, 'output')

    result = openevolve.api.run_evolution(
        initial_program=initial_program,
        evaluator=evaluator_path,
        config=config_path,
        iterations=args.iterations,
        output_dir=output_dir,
    )
    print(f"\nBest score: {result}")

    print("\n\nEvolution complete!")


if __name__ == '__main__':
    main()
