"""
Entry point for the DREAMS Analysis Pipeline API.

Starts the background worker thread and the Flask server on port 5001.

Usage
-----
    python -m analysis_pipeline.api.server
    # or
    python -m analysis_pipeline.api
"""

import logging
import sys

from .app import create_app
from .worker import worker


def _configure_logging() -> None:
    """Set up one clean log stream (avoids duplicate lines)."""
    root = logging.getLogger()
    # Remove any existing handlers to prevent duplicates
    for h in root.handlers[:]:
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    # Quiet noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def main() -> None:
    _configure_logging()
    logger = logging.getLogger(__name__)

    app = create_app()
    worker.start()

    logger.info(
        "DREAMS Analysis Pipeline API running on http://0.0.0.0:5001  "
        "(Ctrl+C to stop)"
    )

    try:
        # debug=False — avoids the reloader spawning a duplicate worker
        app.run(host="0.0.0.0", port=5001, debug=False)
    except KeyboardInterrupt:
        logger.info("Shutting down …")
    finally:
        worker.stop()
        logger.info("Goodbye.")


if __name__ == "__main__":
    main()
