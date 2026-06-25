import torch, math
from torch.optim.lr_scheduler import CosineAnnealingLR

def make_opt_sched(model, lr=2e-4, weight_decay=1e-4, epochs=100):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = CosineAnnealingLR(opt, T_max=epochs)
    return opt, sched

class EarlyStop:
    def __init__(self, patience=12, mode="max"):
        self.patience = patience
        self.mode = mode
        self.best = -math.inf if mode == "max" else math.inf
        self.epochs_bad = 0
        self.checkpoint = None
    def step(self, value, model):
        improved = (value > self.best) if self.mode == "max" else (value < self.best)
        if improved:
            self.best = value
            self.epochs_bad = 0
            self.checkpoint = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            self.epochs_bad += 1
        return improved
    def should_stop(self):
        return self.epochs_bad >= self.patience
