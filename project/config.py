import logging
import torch as t
import wandb
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    root_dir: str = '/dtu/datasets1/02516/ucf101_noleakage'#'data/ufc10'##
    api_key: str = Field(..., alias='api_key')
    device: str = 'cuda' if t.cuda.is_available() else 'cpu'

    # seed: int = 42
    log_level: int = logging.INFO

    model_config = SettingsConfigDict(env_file='.env') # optional prefix for environment variables

settings = Settings()

logger.info(f'Using device: {settings.device}')
logger.info(f"Using dataset: {settings.root_dir}")
logger.info(f"Logging on Weights & Biases")

wandb.login(key = settings.api_key)