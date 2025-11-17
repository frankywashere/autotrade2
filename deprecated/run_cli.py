import os
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from alternator.cli import run_cli_menu  # noqa: E402


if __name__ == "__main__":
    run_cli_menu()

