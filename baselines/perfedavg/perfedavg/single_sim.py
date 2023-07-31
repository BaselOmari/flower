# region PathSetup
import os
import sys

project_base_dir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
sys.path.append(project_base_dir)
# endregion PathSetup

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import Net, train, evaluate
from dataset import load_datasets, _download_data
from torch.utils.data import Dataset, DataLoader
from random import sample
from copy import deepcopy

K = 1000
select_count = 10


dataloaders, testloader = load_datasets("mnist", seed=0)

avg = Net(28 * 28, 10)
for round in range(K):
    print("Starting round", round)
    clients = sample(dataloaders, select_count)
    models = [deepcopy(avg) for _ in range(select_count)]
    for model, dset in zip(models, clients):
        train(
            model,
            dset,
            None,
            mode="hf",
        )

    # create blank model
    for i, param in enumerate(avg.parameters()):
        param.data = torch.zeros_like(param.data)

    # average params
    for key in models[0].state_dict().keys():
        temp = torch.zeros_like(avg.state_dict()[key])
        for client_idx in range(len(models)):
            temp += (1 / 10) * models[client_idx].state_dict()[key]
        avg.state_dict()[key].data.copy_(temp)

    # evaluate
    if round % 10 == 0:
        acc = evaluate(avg, testloader, None)
        print("ACC:", acc)
