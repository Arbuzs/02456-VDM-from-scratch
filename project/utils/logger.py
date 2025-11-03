import logging
from rich.logging import RichHandler
from config import settings
import sys

logging.basicConfig(
    level=settings.log_level,
    handlers=[
        #logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/train.log', mode='a'),
        RichHandler(rich_tracebacks=True,show_path=True)
        ]
    )

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

logger = get_logger(__name__)

def log_metrics(epoch, **metrics):
    msg = ' | '.join(f"{k}={v:.4f}" for k, v in metrics.items())
    logger.info(f" Epoch {epoch:03d} | {msg}")