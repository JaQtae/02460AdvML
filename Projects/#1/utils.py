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

def _get_mnist(path: str, batch_size: int, binarized: bool):
    """
    Gives you back the binarized MNIST dataset in batches of <batch_size>. 
    Saves the data to <dir-name>/data/*.
    """
    if binarized:
    # Load MNIST as binarized at 'threshhold' and create data loaders
        threshold = 0.5
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

    

def plot_prior_and_aggr_posterior(model, data_loader, batch_size,device): 
    # TODO: Should probably be more of a plot like: https://jmtomczak.github.io/blog/7/7_priors.html
    from sklearn.mixture import GaussianMixture
    model.eval()
    post_samples = torch.empty((0,batch_size)).to(device)
    prior_samples = torch.empty((0,batch_size)).to(device)
    targets = torch.empty((0)).to(device)

    
    # just plots the labels and the distribution of the samples PCA'd...          
    # PCA to take it from latent_dim --> 2 dimensions
    pca = PCA(n_components=2)
    print(f"Post_samples: {post_samples.shape}")
    print(f"Prior_samples: {prior_samples.shape}")
    
    # Do I have to fit to both aggr and prior, seperately??
    pca.fit(post_samples.cpu().numpy())
    
    post_samples_pca = pca.transform(post_samples.cpu().numpy())
    prior_samples_pca = pca.transform(prior_samples.cpu().numpy())
    
    plt.figure(figsize=(12, 5))
    # Plot Aggregate Posterior
    plt.subplot(1, 2, 1)
    plt.scatter(post_samples_pca[:, 0], post_samples_pca[:, 1], c=targets.cpu().numpy(), cmap='tab10')
    plt.title('Aggregate Posterior')
    plt.colorbar()

    # Plot Prior
    plt.subplot(1, 2, 2)
    plt.scatter(prior_samples_pca[:, 0], prior_samples_pca[:, 1], cmap='tab10', marker='x')
    plt.title('Prior')

    plt.show()


