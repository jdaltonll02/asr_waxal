from __future__ import annotations

import json
import logging
from pathlib import Path


def setup_logging(output_dir: str) -> logging.Logger:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("waxal_net")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(Path(output_dir) / "run.log")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def write_json(output_dir: str, filename: str, payload: dict) -> None:
    with open(Path(output_dir) / filename, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
