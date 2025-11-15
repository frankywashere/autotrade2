import os
import sys

import uvicorn


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from alternator.config import load_app_config  # noqa: E402
from alternator.server import create_app  # noqa: E402


def main() -> None:
    app_config = load_app_config()
    app = create_app()
    uvicorn.run(
        app,
        host=app_config.server.host,
        port=app_config.server.port,
        reload=False,
    )


if __name__ == "__main__":
    main()

