#!/usr/bin/env python3
"""Launch OpenEvolve V5: OOS-validated bounce signal with drawdown + indicators."""
import os, sys, threading, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

BASE = os.path.dirname(os.path.abspath(__file__))

def start_proxy(port):
    """Start Claude API proxy on given port."""
    from v15.validation.openevolve_bounce.claude_proxy import app
    app.run(host='0.0.0.0', port=port, debug=False)

def main():
    port = 5563

    print(f"Starting Claude proxy on port {port}...")
    proxy_thread = threading.Thread(target=start_proxy, args=(port,), daemon=True)
    proxy_thread.start()
    time.sleep(3)

    import openevolve.api
    openevolve.api.run_evolution(
        initial_program=os.path.join(BASE, 'initial_program_v5.py'),
        evaluator=os.path.join(BASE, 'evaluator_v5.py'),
        config=os.path.join(BASE, 'config_v5.yaml'),
        output_dir=os.path.join(BASE, 'output_v5'),
    )

if __name__ == '__main__':
    main()
