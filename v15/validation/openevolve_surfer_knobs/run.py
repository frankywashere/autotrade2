#!/usr/bin/env python3
"""Launch OpenEvolve Phase A: Surfer-ML knob tuning with honest fills."""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

BASE = os.path.dirname(os.path.abspath(__file__))


def start_proxy(port):
    """Start Claude API proxy on given port."""
    from v15.validation.openevolve_surfer_knobs.claude_proxy import app
    app.run(host='0.0.0.0', port=port, debug=False)


def main():
    port = 5563

    print(f"Starting Claude proxy on port {port} (model: sonnet)...")
    proxy_thread = threading.Thread(target=start_proxy, args=(port,), daemon=True)
    proxy_thread.start()
    time.sleep(3)

    import openevolve.api
    openevolve.api.run_evolution(
        initial_program=os.path.join(BASE, 'initial_program.py'),
        evaluator=os.path.join(BASE, 'evaluator.py'),
        config=os.path.join(BASE, 'config.yaml'),
        output_dir=os.path.join(BASE, 'output'),
    )


if __name__ == '__main__':
    main()
