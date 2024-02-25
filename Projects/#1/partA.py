from DeepGenerativeModels.Week1.vae_bernoulli_mine import (
    VAE,
    BernoulliDecoder,
    GaussianEncoder,
    GaussianPrior,
    MoGPrior,
    train as train_vae,
    evaluate,
) 

from DeepGenerativeModels.Week2.flow_mine import (
    GaussianBase,
    MaskedCouplingLayer,
    Flow,
    train as train_flow,
)

from torchvision import (
    datasets,
    transforms,
)
from torchvision.utils import (
    save_image, 
    make_grid,
)
from torch.utils.data import (
    DataLoader,
)

import torch
import torch.nn as nn
import torch.distributions as td
from sklearn.decomposition import PCA

import glob
import logging
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

logger = logging.getLogger()

######################
#####  CODE DUMP #####
######################
if __name__ == "__main__":
    # TODO: Make sure it's working, might need args for the flow part?
    # TODO: Add the if args.mode == 'train': ...
    # TODO: Figure out where the standard prior is // MoG // Flow-based and make seamless integration
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'eval'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--prior', type=str, default='flowprior.pt', help='file to save prior to or load prior from (default: %(default)s)')  
    parser.add_argument('--prior_type', type=str, default='SG', choices=['sg', 'mog', 'flow'], help='choice of prior (choices: %(choices)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='V', help='learning rate for training (default: %(default)s)')


    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device
    
    # Load MNIST as binarized at 'threshhold' and create data loaders
    threshold = 0.5
    mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True,
                                                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (threshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                                                transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (threshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=False)

    # Define prior distribution
    M = args.latent_dim
    
    # Choose model
    if args.prior_type == 'sg':
        prior = GaussianPrior(M)
    elif args.prior_type == 'mog':
        prior = MoGPrior(M, args.batch_size, device)
        raise NotImplementedError
    elif args.prior_type == 'flow':
        # Define prior distribution
        D = next(iter(mnist_train_loader)).shape[1]
        base = GaussianBase(D)

        # Define transformations
        transformations =[]
        mask = torch.Tensor([1 if (i+j) % 2 == 0 else 0 for i in range(28) for j in range(28)])
        
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
            
        # TODO: Add Flow prior (Is this from a pre-trained Flow model?)
        flowprior = Flow(base, transformations).to(device)
        flowprior.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        
    # Define encoder and decoder networks
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, M*2),
    )
    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28))
    )

    # Define VAE model
    decoder = BernoulliDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)
    model = VAE(prior, decoder, encoder).to(device)


    if args.mode == 'train':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        logger.info("Starting training...")
        train_vae(model, optimizer, mnist_train_loader, args.epochs, args.device)
        
        logger.info(f"Saving model with name: {args.model}")
        torch.save(model.state_dict(), args.model)
        
    elif args.mode == 'eval':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        
        logger.info(f"Test loss: {evaluate(model, mnist_test_loader)}")
        