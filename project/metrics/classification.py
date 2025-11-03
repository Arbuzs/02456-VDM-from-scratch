import torch as t

class BaseMetric:
    def __init__(self):
        self.reset()

    def reset(self):
        raise NotImplementedError

    def update(self, predictions, labels):
        raise NotImplementedError
    
    def compute(self):
        raise NotImplementedError
    

class Accuracy(BaseMetric):
    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, predictions, labels):
        #predictions = predictions.detach()
        #labels = labels.detach()
        self.correct += (predictions.argmax(1) == labels).sum().item()
        self.total += labels.size(0)

    def compute(self):
        return self.correct / self.total * 100.0 #if self.total > 0 else 0.0