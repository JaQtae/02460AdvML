# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.1 (2024-01-29)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)
    
class MoGPrior(nn.Module):

    def __init__(self, M, num_components, device = 'cpu', init_radius: float = 4.0):
        """
        Define a Mixture of Gaussian (MoG) prior distribution.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        """
        super(MoGPrior, self).__init__()
        self.M = M
        self.num_components = num_components
        self.device = device
        self.init_radius = init_radius
        
        # init_radius ensures it never diverges outside of bounds given
        self.mean = nn.Parameter(torch.randn(self.num_components, self.M).uniform_(-self.init_radius, self.init_radius), requires_grad=False).to(self.device)
        #self.logvars = nn.Parameter(torch.ones(self.num_components, self.M), requires_grad=False).to(self.device)
        self.stds = nn.Parameter(1 + abs(torch.randn(self.num_components, self.M)), requires_grad = True)
        
        self.weights = nn.Parameter(torch.ones(self.num_components), requires_grad=True)

    def forward(self):
        # https://github.com/pytorch/pytorch/blob/main/torch/distributions/mixture_same_family.py
        
        # Mixing probabilities
        mixture_dist = td.Categorical(probs=F.softmax(self.weights, dim=0))
        comp_dist = td.Independent(td.Normal(loc=self.mean, scale=self.stds), 1)
        return td.MixtureSameFamily(mixture_dist, comp_dist)
    
    def log_prob(self, z):
        # To return a log_prob of the forward, thus matching Flow-class callable in ELBO.
        return self.forward().log_prob(z)

   

class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]             
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)
    
    def log_prob(self, x):
        return self.forward(x).log_prob()


class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.std = nn.Parameter(torch.ones(28, 28)*0.5, requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits, validate_args=False), 2)



class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module] 
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """
            
        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder
        

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
        """
        q = self.encoder(x)
        z = q.rsample() # [batch, latent_dim]
    
        if self.prior.__class__.__name__ == "GaussianPrior":
            elbo = torch.mean(self.decoder(z).log_prob(x) - td.kl_divergence(q, self.prior()), dim=0)
        else:
            # non-Gaussian prior (e.g. MoGPrior and Flow-based prior)
            regularization_term = q.log_prob(z) - self.prior.log_prob(z) # Uses the inverse
            elbo = torch.mean(self.decoder(z).log_prob(x) - regularization_term, dim=0)
               
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior.sample(torch.Size([n_samples]))
        return self.decoder(z).sample()
    
    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()
    num_steps = len(data_loader)*epochs
    epoch = 0

    with tqdm(range(num_steps)) as pbar:
        for step in pbar:
            x = next(iter(data_loader))[0]
            x = x.to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            # Report
            if step % 5 ==0 :
                loss = loss.detach().cpu()
                pbar.set_description(f"epoch={epoch}, step={step}, loss={loss:.1f}")

            if (step+1) % len(data_loader) == 0:
                epoch += 1



#### Added methods for exercises or ease-of-use :)                
def evaluate(model, dataset):
    test_loss = 0.0
    for x, y in dataset:
        test_loss += model(x).item() #log_prob
        
    test_loss /= len(dataset)
        
    return test_loss

def sample_projection(model, data_loader, device):
    # Courtesey of Daniel the Man
    """
    Sample from the latent space of a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to sample from.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for sampling.
    device: [torch.device]
        The device to use for sampling.
    """
    model.eval()
    samples = torch.empty((0,10)).to(device)
    targets = torch.empty((0)).to(device)
    with torch.no_grad():
        for x, target in data_loader:
            x = x.to(device)
            target = target.to(device)
            q = model.encoder(x)
            z = q.sample()
            samples = torch.cat((samples, z), dim=0)
            targets = torch.cat((targets, target), dim=0)


    pca = PCA(n_components = 2)
    pca.fit(samples.cpu().numpy())
    samples = pca.transform(samples.cpu().numpy())
    plt.scatter(samples[:,0], samples[:,1], c=targets.cpu().numpy(), cmap='tab10')
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image, make_grid
    import glob

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'eval'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    # Load MNIST as binarized at 'thresshold' and create data loaders
    thresshold = 0.5
    mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True,
                                                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                                                transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=False)

    # Define prior distribution
    M = args.latent_dim
    prior = GaussianPrior(M)

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

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train model
        train(model, optimizer, mnist_train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), args.model)

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample(64)).cpu() 
            save_image(samples.view(64, 1, 28, 28), args.samples)
    
    elif args.mode == 'eval':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        
        sample_projection(model=model,
                          data_loader=mnist_test_loader,
                          device=device)

## Train
# python vae_bernoulli_mine.py train --device cuda --latent-dim 10 --epochs 5 --batch-size 128 --model model.pt

## Sample
# python vae_bernoulli_mine.py sample --device cuda --latent-dim 10 --epochs 5 --batch-size 128 --model model.pt --samples samples.png

## Plot PCA
# python vae_bernoulli_mine.py eval --device cuda --latent-dim 10 --epochs 5 --batch-size 128 --model model.pt