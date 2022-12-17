import copy

class BaseDetector():
    def __init__(self, name, net, require_grad=False) -> None:
        self.net = copy.deepcopy(net)
        self.name = name
        self.require_grad = require_grad

    def setup(self, train_loader, **kwargs):
        pass

    def score(self, loader):
        raise NotImplemented
    
    def score_batch(self, data):
        raise NotImplemented
    
    def __str__(self) -> str:
        return f"{self.name}"