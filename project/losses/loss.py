from torch.nn import CrossEntropyLoss
from metrics.classification import BaseMetric

class BaseLoss:
    def __init__(self):
        self.reset()

    def reset(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError
    
    def compute(self):
        raise NotImplementedError
    

class LossFunction(BaseMetric):
    def __init__(self, loss_function):
        self.loss_function = loss_function

    def reset(self):
        self.loss_sum = 0.0
        self.loss_total = 0.0

    def update(self, predictions, labels):
        loss = self.loss_function(predictions, labels)
        self.loss_sum += loss.item()
        self.loss_total += labels.size(0)
        return loss
    
    def compute(self):
        return self.loss_sum / self.loss_total