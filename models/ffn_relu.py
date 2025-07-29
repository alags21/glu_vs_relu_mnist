import torch
import torch.nn as nn
import torch.nn.functional as F

class FFNReLU(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        # used randn for random weights

        # Weight Matrix - d_model to d_hidden (w/o bias, scaling)
        self.W_in = nn.Parameter(torch.randn(d_model, d_hidden))

        # Weight Matrix - d_hidden to d_model (w/o bias, scaling)
        self.W_out = nn.Parameter(torch.randn(d_hidden, d_model))

    def forward(self, x):
        hidden = torch.einsum('btd,df->btf', x, self.W_in) # Project x to hidden dimension (x @ W_in)
        activated = torch.relu(hidden) # Apply ReLU (get rid of negatives, keep positives)
        out = torch.einsum('btf,fd->btd', activated, self.W_out) # Project down (activated @ W_out)
        return out
