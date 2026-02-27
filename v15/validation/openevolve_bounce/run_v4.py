"""
Launcher for OpenEvolve v4: directional signal (long + short).
Uses port 5562.
"""
import argparse, os, sys, threading, time
import requests

def start_proxy(port):
    from claude_proxy import app
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

def wait_for_proxy(port, timeout=30):
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
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5562)
    parser.add_argument('--iterations', type=int, default=300)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"OpenEvolve v4: Directional Signal (Long + Short)")
    print(f"Port: {args.port}, Iterations: {args.iterations}")
    print(f"{'='*60}")

    print("\nStarting Claude proxy...")
    proxy_thread = threading.Thread(target=start_proxy, args=(args.port,), daemon=True)
    proxy_thread.start()
    if not wait_for_proxy(args.port):
        print("Failed to start proxy. Exiting.")
        sys.exit(1)

    print("\nStarting OpenEvolve...")
    import openevolve.api
    base_dir = os.path.dirname(os.path.abspath(__file__))
    result = openevolve.api.run_evolution(
        initial_program=os.path.join(base_dir, 'initial_program_v4.py'),
        evaluator=os.path.join(base_dir, 'evaluator_v4.py'),
        config=os.path.join(base_dir, 'config_v4.yaml'),
        iterations=args.iterations,
        output_dir=os.path.join(base_dir, 'output_v4'),
    )
    print(f"\nBest score: {result}")
    print("\nEvolution complete!")

if __name__ == '__main__':
    main()
