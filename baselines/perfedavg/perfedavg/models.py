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
        self.flat = nn.Flatten()
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
        inp = self.flat(input_tensor)
        z1 = F.elu(self.h1(inp), alpha=0.2)
        z2 = F.elu(self.h2(z1), alpha=0.2)
        logits = F.elu(self.out(z2), alpha=0.2)
        return logits


def train(  # pylint: disable=too-many-arguments
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    learning_rate: float = 0.01,
    steps: int = 10,
    step_size: float = 0.01,
    first_order: bool = True,
) -> None:
    """Train the network on the training set.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The DataLoader containing the data to train the network on.
    device : torch.device
        The device on which the model should be trained, either 'cpu' or 'cuda'.
    learning_rate : float
        The learning rate for the SGD optimizer.
        Denoted by beta in PerFedAvg paper.
    steps : int
        The number of training step the model should be trained for.
        Denoted by tau in PerFedAvg paper.
    step_size : float
        Parameter for the step size in the MAML loss function.
        Denoted by alpha in PerFedAvg paper.
    first_order : bool
        Approximation method for gradients
        If True use the first order approximation - Per-FedAvg (FO)
        Else use the hessian free approximation - Per-FedAvg (HF)
    """
    criterion = torch.nn.CrossEntropyLoss()

    # Create copy of net which represents the model one step in
    # advance (denoted by omega with superscript tilde)
    net_maml = deepcopy(net)
    optimizer_maml = torch.optim.SGD(net_maml.parameters(), lr=step_size)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

    net.train()
    net_maml.train()
    for _ in range(steps):
        # Step 0: Sync clone model with current
        net_maml.load_state_dict(net.state_dict())

        # Step 1: Update clone model by one step
        D1_X, D1_y = next(iter(trainloader))
        optimizer_maml.zero_grad()
        D1_y_hat = F.softmax(net_maml(D1_X))
        loss = criterion(D1_y_hat, D1_y)
        loss.backward()
        optimizer_maml.step()

        # Step 2:
        D2_X, D2_y = next(iter(trainloader))
        if first_order:
            # Calculate first order approximation of the gradients
            optimizer_maml.zero_grad()
            D2_y_hat = F.softmax(net_maml(D2_X))
            loss = criterion(D2_y_hat, D2_y)
            loss.backward()

            # Copy gradients
            for net_p, net_maml_p in zip(
                net.named_parameters(), net_maml.named_parameters()
            ):
                net_p[1].grad = net_maml_p[1].grad

            # Update weights according to new gradients
            optimizer.step()

        else:
            # TODO: Implement training with Hessian Free Approximation
            raise NotImplementedError("Hessian Free Approximation not implemented")


def evaluate(  # pylint: disable=too-many-arguments
    net: nn.Module,
    testloader: DataLoader,
    device: torch.device,
    step_size: float = 0.01,
) -> None:
    # Step 1: Perform one step of SGD
    net_maml = deepcopy(net)
    optimizer_maml = torch.optim.SGD(net_maml.parameters(), lr=step_size)
    criterion = torch.nn.CrossEntropyLoss()
    D1_X, D1_y = next(iter(testloader))
    optimizer_maml.zero_grad()
    D1_y_hat = F.softmax(net_maml(D1_X))
    loss = criterion(D1_y_hat, D1_y)
    loss.backward()
    optimizer_maml.step()

    # Step 2: Evaluate model
    net_maml.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for input, target in testloader:
            output = net_maml(input)
            correct += (output.argmax(1) == target).sum().item()
            total += target.size(0)
    return correct / total
