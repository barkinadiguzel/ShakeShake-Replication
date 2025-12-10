import torch

class AlphaScheduler:
    def __init__(self, method='shake'):
        self.method = method

    def get_alpha(self, batch_size, device):
        if self.method == 'even':
            return torch.full((batch_size,1,1,1), 0.5, device=device)
        elif self.method == 'shake':
            return torch.rand(batch_size,1,1,1, device=device)
        else:
            raise ValueError(f"Unknown alpha scheduling method: {self.method}")
