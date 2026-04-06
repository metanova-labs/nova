import torch
from torch import Tensor, nn

import boltz.model.layers.initialize as init


class OuterProductMean(nn.Module):
    """Outer product mean layer."""

    def __init__(self, c_in: int, c_hidden: int, c_out: int) -> None:
        """Initialize the outer product mean layer.

        Parameters
        ----------
        c_in : int
            The input dimension.
        c_hidden : int
            The hidden dimension.
        c_out : int
            The output dimension.

        """
        super().__init__()
        self.c_hidden = c_hidden
        self.norm = nn.LayerNorm(c_in)
        self.proj_a = nn.Linear(c_in, c_hidden, bias=False)
        self.proj_b = nn.Linear(c_in, c_hidden, bias=False)
        self.proj_o = nn.Linear(c_hidden * c_hidden, c_out)
        init.final_init_(self.proj_o.weight)
        init.final_init_(self.proj_o.bias)

    def forward(self, m: Tensor, mask: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        m : torch.Tensor
            The sequence tensor (B, S, N, c_in).
        mask : torch.Tensor
            The mask tensor (B, S, N).

        Returns
        -------
        torch.Tensor
            The output tensor (B, N, N, c_out).

        """
        # Expand mask
        mask = mask.unsqueeze(-1).to(m)

        # Compute projections
        m = self.norm(m)
        a = self.proj_a(m) * mask
        b = self.proj_b(m) * mask
        mask = mask[:, :, None, :] * mask[:, :, :, None]
        num_mask = mask.sum(1).clamp(min=1)
        z = torch.einsum("bsic,bsjd->bijcd", a.float(), b.float())
        z = z.reshape(*z.shape[:3], -1)
        z = z / num_mask

        # Project to output
        z = self.proj_o(z.to(m))
        return z
