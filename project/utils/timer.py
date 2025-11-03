import time
from utils.logger import get_logger

log = get_logger(__name__)

def timer(func):
    def wrapper_timer(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        time_elapsed = time.time() - start_time
        log.info(f"Function '{func.__name__}' executed in {time_elapsed:.4f} seconds")
        return result
    return wrapper_timer