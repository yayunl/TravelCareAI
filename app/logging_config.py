"""Central logging setup. Call ``configure_logging()`` early (e.g. from ``app.main`` after ``load_dotenv``).

Environment:
- ``LOG_LEVEL``: root log level (default ``INFO``), e.g. ``DEBUG``, ``WARNING``.
- ``LOG_FILE``: log file path (absolute, or relative to the project root next to ``app/``). Default: ``logs/travelcare.log``.
- ``LOG_DISABLE_FILE``: set to ``1`` / ``true`` to log only to the console (no file).
- ``LOG_FILE_MAX_MB``: max size per log file before rotation (default ``10``).
- ``LOG_FILE_BACKUP_COUNT``: number of rotated backups to keep (default ``5``).
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

_APP_ROOT = Path(__file__).resolve().parent.parent


def _file_handler_for_path(root: logging.Logger, path: Path) -> RotatingFileHandler | None:
    target = path.resolve()
    for h in root.handlers:
        if isinstance(h, RotatingFileHandler):
            try:
                if Path(h.baseFilename).resolve() == target:
                    return h
            except OSError:
                continue
    return None


def configure_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)
    root = logging.getLogger()
    fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    if not root.handlers:
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        root.addHandler(stream)
    else:
        for h in root.handlers:
            if h.formatter is None:
                h.setFormatter(formatter)

    root.setLevel(level)

    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    disable_file = os.getenv("LOG_DISABLE_FILE", "").strip().lower() in ("1", "true", "yes")
    if disable_file:
        return

    path_str = os.getenv("LOG_FILE", "").strip()
    log_path = Path(path_str) if path_str else (_APP_ROOT / "logs" / "travelcare.log")
    if not log_path.is_absolute():
        log_path = (_APP_ROOT / log_path).resolve()
    else:
        log_path = log_path.resolve()

    if _file_handler_for_path(root, log_path) is not None:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        max_mb = float(os.getenv("LOG_FILE_MAX_MB", "10") or "10")
    except ValueError:
        max_mb = 10.0
    max_bytes = max(int(max_mb * 1024 * 1024), 1_048_576)
    try:
        backups = int(os.getenv("LOG_FILE_BACKUP_COUNT", "5") or "5")
    except ValueError:
        backups = 5
    backups = max(backups, 1)

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backups,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    logging.getLogger(__name__).info("File logging enabled at %s", log_path)
