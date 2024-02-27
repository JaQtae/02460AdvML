import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm

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

def _get_flow_decoder(M: int):
    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        #nn.Unflatten(-1, (28, 28))
    )
    return decoder_net

def _get_mask_tranformations(D_: int):
    """
    Generates the masking transformation based on some dimension D.
    
    Returns:
        base: Base Gaussian distribution.
        transformations: list of the required masks.
    """
    D = D_ # 784
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

def _get_mnist(path: str, batch_size: int, binarized: bool, prior: str):
    """
    Gives you back the binarized MNIST dataset in batches of <batch_size>. 
    Saves the data to <dir-name>/data/*.
    """
    if binarized:
    # Load MNIST as binarized at 'threshhold' and create data loaders
        threshold = 0.5
        if prior == 'flow':
            # Flow-based prior requires us to flatten the data...
            mnist_train_loader = DataLoader(datasets.MNIST(path+'data/', train=True, download=True,
                                                                            transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (threshold < x).float().squeeze()),  transforms.Lambda(lambda x: x.flatten())])),
                                                            batch_size=batch_size, shuffle=True)
            mnist_test_loader = DataLoader(datasets.MNIST(path+'data/', train=False, download=True,
                                                                        transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (threshold < x).float().squeeze()),  transforms.Lambda(lambda x: x.flatten())])),
                                                            batch_size=batch_size, shuffle=False)
        else:
            mnist_train_loader = DataLoader(datasets.MNIST(path+'data/', train=True, download=True,
                                                                            transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (threshold < x).float().squeeze())])),
                                                            batch_size=batch_size, shuffle=True)
            mnist_test_loader = DataLoader(datasets.MNIST(path+'data/', train=False, download=True,
                                                                        transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (threshold < x).float().squeeze())])),
                                                            batch_size=batch_size, shuffle=False)
        
    else: 
        mnist_train_loader = DataLoader(datasets.MNIST(path+'data/', train=True, download=True,
                                                                    transform=transforms.Compose([transforms.ToTensor (), transforms.Lambda(lambda x: x.squeeze ())])),
                                                    batch_size=batch_size, shuffle=True)
        mnist_test_loader = DataLoader(datasets.MNIST(path+'data/', train=False, download=True,
                                                                transform=transforms.Compose([transforms.ToTensor (), transforms.Lambda(lambda x: x.squeeze ())])),
                                                    batch_size=batch_size, shuffle=False)

    return mnist_train_loader, mnist_test_loader

    

def plot_prior_and_aggr_posterior_2d(model, data_loader, latent_dim, n_samples, device): 
    # Define a grid for the latent space
    model.eval()
    x = torch.linspace(-18, 18, 100).to(device)
    y = torch.linspace(-20, 20, 100).to(device)
    xx, yy = torch.meshgrid(x, y)

    # Evaluate the prior log probability on the grid

    # Collect posterior samples
    post_samples = torch.empty((0, latent_dim)).to(device)
    targets = torch.empty((0)).to(device)
    with torch.no_grad():
        for x, target in tqdm(data_loader):
            x = x.to(device)
            target = target.to(device)
            q = model.encoder(x)
            z = q.sample()  # [batch_size, latent_dim]
            post_samples = torch.cat((post_samples, z), dim=0)
            targets = torch.cat((targets, target), dim=0)

        prior_log_prob = model.prior().log_prob(torch.stack([xx, yy], dim=-1)).view(100, 100).cpu().numpy()

        # Plot the prior contour
        plt.contourf(xx.cpu().numpy(), yy.cpu().numpy(), prior_log_prob, cmap='viridis', alpha=0.4)
        # Plot the projected posterior samples
        plt.scatter(post_samples[:n_samples, 0].cpu().numpy(), post_samples[:n_samples, 1].cpu().numpy(), c=targets[:n_samples].cpu().numpy(), cmap='tab10')
        plt.colorbar()
        plt.savefig(f"{model.prior.__class__.__name__}.png")
        # Show the plot
        plt.show()
    

