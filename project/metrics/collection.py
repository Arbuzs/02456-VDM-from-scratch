class MetricCollection:
    def __init__(self, metrics: dict):
        self.metrics = metrics

    def reset(self):
        for m in self.metrics.values():
            m.reset()

    def update(self, preds, labels):
        for m in self.metrics.values():
            m.update(preds, labels)

    def compute(self):
        return {name: m.compute() for name, m in self.metrics.items()}