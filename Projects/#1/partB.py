from src.DeepGenerativeModels.Week2.flow_mine import (
    GaussianBase,
    MaskedCouplingLayer,
    Flow,
    train as train_flow,
)

from src.DeepGenerativeModels.Week3.ddpm import (
    FcNetwork,
    DDPM,
    train as train_ddpm,
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
import pdb

logger = logging.getLogger()



###
# Flowbased model trained on standard mnist

# DDPM trained on -||-


if __name__ == "__main__":
    # TODO: Make sure it's working, might need args for the flow part?
    # TODO: Add the if args.mode == 'train': ...
    # TODO: Figure out where the standard prior is // MoG // Flow-based and make seamless integration
    import os
    dir_name = os.path.dirname(os.path.abspath(__file__)) + '/'

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'eval'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--model_type', type=str, default='flow', choices=['flow', 'ddpm'], help='choice of prior (choices: %(choices)s)')
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
    
    
    # Define a transform to normalize the data
    _transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    _transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.flatten())])

    # Download and load the training data
    mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST(dir_name+'data/', train=True, download=True,
                                                                    transform=_transform),
                                                    batch_size=args.batch_size, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST(dir_name+'data/', train=False, download=True,
                                                                transform=_transform),
                                                    batch_size=args.batch_size, shuffle=False)

    ######################
    ### Flow file dump ###
    ######################
    # TODO: fix this with if statements, maybe...
    D = next(iter(mnist_train_loader))[0].shape[1]

    base = GaussianBase(D)

    # Define transformations
    transformations =[]
    mask = torch.Tensor([1 if (i+j) % 2 == 0 else 0 for i in range(28) for j in range(28)])

    num_transformations = 10
    num_hidden = 128*2

    # Make a mask that is 1 for the first half of the features and 0 for the second half
    mask = torch.zeros((D,))
    mask[D//2:] = 1

    # checkboard mask
    mask = (torch.arange(D) % 2).float()

    for i in range(num_transformations):
        mask = (1-mask) # Flip the mask
        scale_net = nn.Sequential(nn.Linear(D, D//2), nn.ReLU(), nn.Linear(D//2, 2*num_hidden), nn.ReLU(), nn.Linear(2*num_hidden, num_hidden), nn.ReLU(), nn.Linear(num_hidden, D), nn.Tanh())
        translation_net = nn.Sequential(nn.Linear(D, D//2), nn.ReLU(), nn.Linear(D//2, 2*num_hidden), nn.ReLU(), nn.Linear(2*num_hidden, num_hidden), nn.ReLU(), nn.Linear(num_hidden, D))
        transformations.append(MaskedCouplingLayer(scale_net, translation_net, mask))   

    ##########
    ## DDPM ##
    ##########
    # Define the network for DDPM
    num_hidden = 64
    network = FcNetwork(D, num_hidden)

    # Set the number of steps in the diffusion process
    T = 1000
    
    if args.model_type == 'flow':
        model = Flow(base, transformations).to(device)
    elif args.model_type == 'ddpm':
        model = DDPM(network, T=T).to(args.device)

    if args.mode == 'train':   
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        ## Flow     
        if args.model_type == 'flow':
            logger.info(f"Starting training of {args.model_type} model")
            train_flow(model, optimizer, mnist_train_loader, args.epochs, args.device)

        ## DDPM
        elif args.model_type == 'ddpm':
            logger.info(f"Starting training of {args.model_type} model")
            train_ddpm(model, optimizer, mnist_train_loader, args.epochs, args.device)
        
        logger.info(f"Saving model as: {args.model}")
        torch.save(model.state_dict(), dir_name+args.model)
        
    elif args.mode == 'sample':
        if args.model_type == 'flow':
            logger.info(f"Sampling {args.model_type} model")
            model.load_state_dict(torch.load(dir_name+args.model, map_location=torch.device(args.device)))

        # Generate samples
            model.eval()
            with torch.no_grad():
                import os
                if not os.path.exists(dir_name + "mnist_flow_samples"):
                    os.makedirs(dir_name + "mnist_flow_samples")
                n_samples = 64
                for sample in range(n_samples):
                    samples = (model.sample((1,))).cpu()
                    #samples = (samples - samples.min()) / (samples.max() - samples.min()) 
                    save_image(samples.view(1, 1, 28, 28), f"{dir_name}\mnist_flow_samples\sample_{sample}.png")


            
        elif args.model_type == 'ddpm':
            import matplotlib.pyplot as plt
            import numpy as np

            # Load the model
            model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

            # Generate samples
            model.eval()
            with torch.no_grad():
                samples = (model.sample((10000,D))).cpu() 

            # Transform the samples back to the original space
            samples = samples /2 + 0.5
            
            # TODO: Update the below to reflect it happening on MNIST
            # Plot the density of the toy data and the model samples
            coordinates = [[[x,y] for x in np.linspace(*toy.xlim, 1000)] for y in np.linspace(*toy.ylim, 1000)]
            prob = torch.exp(toy().log_prob(torch.tensor(coordinates)))

            fig, ax = plt.subplots(1, 1, figsize=(7, 5))
            im = ax.imshow(prob, extent=[toy.xlim[0], toy.xlim[1], toy.ylim[0], toy.ylim[1]], origin='lower', cmap='YlOrRd')
            ax.scatter(samples[:, 0], samples[:, 1], s=1, c='black', alpha=0.5)
            ax.set_xlim(toy.xlim)
            ax.set_ylim(toy.ylim)
            ax.set_aspect('equal')
            fig.colorbar(im)
            plt.savefig(args.samples)
            plt.close()
            
        
    # elif args.mode == 'eval':
    #     model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
    #     logger.info(f"Test loss: {evaluate(model, mnist_test_loader)}")
        