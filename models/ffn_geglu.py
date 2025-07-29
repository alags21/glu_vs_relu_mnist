import torch
import torch.nn as nn
import torch.nn.functional as F

class FFNGEGLU(nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        # used randn for random weights

        # Weight Matrix - d_model to d_hidden (w/o bias, scaling)
        self.W_in = nn.Parameter(torch.randn(d_model, d_hidden))

        # Weight Matrix - gating projection
        self.W_gate = nn.Parameter(torch.randn(d_model, d_hidden))

        # Weight Matrix - d_hidden to d_model (w/o bias, scaling)
        self.W_out = nn.Parameter(torch.randn(d_hidden, d_model))

    def forward(self, x):
        proj = torch.einsum('btd,df->btf', x, self.W_in) # Project x up to hidden dimension (x @ W_in)
        gate_pre = torch.einsum('btd,df->btf', x, self.W_gate) # Apply Gating Transformation
        gate = F.gelu(gate_pre) # Apply GELU activation to gate signal
        gated = proj * gate # Element-wise multiplication of proj and gate filter
        out = torch.einsum('btd,fd->btd', gated, self.W_out) # Project back to model dimension (gated @ W_out)
        return out