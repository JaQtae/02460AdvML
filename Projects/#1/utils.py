import numpy as np
import os

import torch
import torch.nn as nn
from torchvision import (
    datasets,
    transforms,
)

from torch.utils.data import (
    DataLoader,
)

from src.DeepGenerativeModels.Week2.flow_mine import (
    GaussianBase,
    MaskedCouplingLayer,
)


def _get_encoder(M: int):
    encoder =   nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, M*2),
        )
    return encoder

def _get_decoder(M: int):
    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28))
    )
    return decoder_net

def _get_mask_tranformations(dim: int):
    """
    Generates the masking transformation based on some dimension D.
    
    Returns:
        base: Base Gaussian distribution.
        transformations: list of the required masks.
    """
    D = dim
    base = GaussianBase(D)

    # Define transformations
    transformations = []
    num_transformations = 5
    num_hidden = 8

    # Make a mask that is 1 for the first half of the features and 0 for the second half
    mask = torch.zeros((D,))
    mask[D//2:] = 1
    
    for i in range(num_transformations):
        mask = (1-mask) # Flip the mask
        scale_net = nn.Sequential(nn.Linear(D, num_hidden), nn.ReLU(), nn.Linear(num_hidden, D))
        translation_net = nn.Sequential(nn.Linear(D, num_hidden), nn.ReLU(), nn.Linear(num_hidden, D))
        transformations.append(MaskedCouplingLayer(scale_net, translation_net, mask))
        
    return base, transformations

def _get_binarized_mnist(path: str, batch_size):
    """
    Gives you back the binarized MNIST dataset in batches of <batch_size>. 
    Saves the data to <dir-name>/data/*.
    """
    # Load MNIST as binarized at 'threshhold' and create data loaders
    threshold = 0.5
    mnist_train_loader = DataLoader(datasets.MNIST(path+'data/', train=True, download=True,
                                                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (threshold < x).float().squeeze())])),
                                                    batch_size=batch_size, shuffle=True)
    mnist_test_loader = DataLoader(datasets.MNIST(path+'data/', train=False, download=True,
                                                                transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (threshold < x).float().squeeze())])),
                                                    batch_size=batch_size, shuffle=False)

    return mnist_train_loader, mnist_test_loader
