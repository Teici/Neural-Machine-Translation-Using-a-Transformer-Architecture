import math
import torch

def set_seed(seed: int = 42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def subsequent_mask(size: int) -> torch.Tensor:
    # (1, size, size) True = allowed
    return torch.tril(torch.ones(size, size, dtype=torch.bool)).unsqueeze(0)

def pad_keep_mask(x: torch.Tensor, pad_id: int) -> torch.Tensor:
    # (B, T) -> (B,1,1,T) True = keep, False = pad
    return (x != pad_id).unsqueeze(1).unsqueeze(2)

def log_sum_exp(a: torch.Tensor, dim: int = -1):
    m, _ = torch.max(a, dim=dim, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(a - m), dim=dim, keepdim=True))

class SimpleLRScheduler:
    """
    Constant LR by default. Keeping it simple helps from-scratch training.
    """
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self):
        return self.optimizer.param_groups[0]["lr"]
