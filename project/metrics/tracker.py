import pandas as pd
from utils.logger import get_logger

logger = get_logger(__name__)

class MetricTracker:
    def __init__(self):
        self.history = {}

    def update(self, epoch, **metrics):
        if "epoch" not in self.history:
            self.history["epoch"] = []
        self.history["epoch"].append(epoch)
        for name, value in metrics.items():
            self.history.setdefault(name, []).append(value)

    def to_dataframe(self):
        return pd.DataFrame(self.history)

    def save_csv(self, path):
        self.to_dataframe().to_csv(path, index=False)