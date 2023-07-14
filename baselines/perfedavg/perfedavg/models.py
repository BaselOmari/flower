import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.utils.data import DataLoader


class Net(nn.Module):
    """
    Neural network architecture as described in section 5 (Numerical Experiments)
    of the PerFedAvg paper
    """

    def __init__(self, input_size: int, num_classes: int) -> None:
        super().__init__()
        self.h1 = nn.Linear(input_size, 80)
        self.h2 = nn.Linear(80, 60)
        self.out = nn.Linear(60, num_classes)
        

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CNN.

        Parameters
        ----------
        x : torch.Tensor
            Input Tensor that will pass through the network

        Returns
        -------
        torch.Tensor
            The resulting Tensor after it has passed through the network
        """
        z1 = F.elu(self.h1(input_tensor), alpha=0.2)
        z2 = F.elu(self.h2(z1), alpha=0.2)
        logits = F.elu(self.out(z2), alpha=0.2)
        return logits
