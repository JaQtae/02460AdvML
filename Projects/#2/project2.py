# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen and Søren Hauberg, 2024
# Version 1.0 (2024-01-27)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

import torch
import torch.nn as nn
import torch.distributions as td
from torch.distributions.kl import kl_divergence as KL
import torch.utils.data
from tqdm import tqdm
from torch.optim import LBFGS
import pdb


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
        return td.Independent(td.ContinuousBernoulli(logits=logits), 3)


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
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample()
        elbo = torch.mean(self.decoder(z).log_prob(x) - td.kl_divergence(q, self.prior()), dim=0)
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
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

    def noise(x, std=0.05):
        eps = std * torch.randn_like(x)
        return torch.clamp(x + eps, min=0.0, max=1.0)

    with tqdm(range(num_steps)) as pbar:
        for step in pbar:
            x = next(iter(data_loader))[0]
            x = noise(x.to(device))
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


def proximity(curve_points, latent):
    """
    Compute the average distance between points on a curve and a collection
    of latent variables.

    Parameters:
    curve_points: [torch.tensor]
        M points along a curve in latent space. tensor shape: M x latent_dim
    latent: [torch.tensor]
        N points in latent space (latent means). tensor shape: N x latent_dim

    The function returns a scalar.
    """
    pd = torch.cdist(curve_points, latent)  # M x N
    pd_min, _ = torch.min(pd, dim=0)
    pd_min_max = pd_min.max()
    return pd_min_max

### TODO: Geodesic function ( this is basicly straight lines or third-order polynomials? )
import torch
import torch.distributions as td

def create_curve(c0, c1, N = 25, order = 1):
    # create line between c0 and c1
    # Rembember that c0 and c1 both are fixed points
    t = torch.linspace(0, 1, N-1)
    curve = (1 - t[:, None]) * c0 + t[:, None] * c1
    if order >= 2:
        #weights = torch.rand((N, 2, order - 1))
        weights = torch.cat((torch.linspace(start = 0, end = 1, steps = N//2)[:-1], torch.linspace(1, 0, N//2)))
        weights = weights.view(N-1, 1, 1).repeat(1, 2, order - 1) 
        weights = weights + (torch.randn_like(weights) * 0.05) 
        weights[0, :, :] = 0
        weights[-1, :, :] = 0 #-torch.sum(weights[: -1, :], dim=0)
        for i in range(2, order + 1):
            curve += weights[:, :, i - 2] * t[:, None]**i

    return curve, t, weights

def update_curve(c0, c1, t, weights, order):
    weights_update = weights.clone()
    # update curve
    curve = (1 - t[:, None]) * c0 + t[:, None] * c1
    for i in range(2, order + 1):
        weights_update[0, :, :] = 0
        weights_update[-1, :, :] = 0 #-torch.sum(weights_update[: -1, :], dim=0)
        curve += weights_update[:, :, i - 2] * t[:, None]**i
    return curve

def fr_energy(
    model, c0, c1, t, weights, order):
    """ Fisher-Rao metric """
    curve = update_curve(c0, c1, t, weights, order)
    energy = torch.tensor([0.])
    num_points = curve.size(0)
    for i in range(num_points - 1):
        z0 = curve[i]
        z1 = curve[i + 1]
        energy += td.kl.kl_divergence(model.decoder(z0), model.decoder(z1))
    return energy

if __name__ == "__main__":
    from torchvision import datasets, transforms
    import glob

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'plot'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--plot', type=str, default='plot.png', help='file to save latent plot in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=15, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=2, metavar='N', help='dimension of latent variable (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    # Load a subset of MNIST and create data loaders
    def subsample(data, targets, num_data, num_classes):
        idx = targets < num_classes
        new_data = data[idx][:num_data].unsqueeze(1).to(torch.float32) / 255
        new_targets = targets[idx][:num_data]

        return torch.utils.data.TensorDataset(new_data, new_targets)
    
    num_train_data = 2048
    num_test_data = 16  # we keep this number low to only compute a few geodesics
    num_classes = 3
    train_tensors = datasets.MNIST('data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    train_data = subsample(train_tensors.data, train_tensors.targets, num_train_data, num_classes)
    mnist_train_loader = torch.utils.data.DataLoader(train_data, batch_size=32)
    
    # Define prior distribution
    M = args.latent_dim
    prior = GaussianPrior(M)

    encoder_net = nn.Sequential(
        nn.Conv2d(1, 16, 3, stride=2, padding=1),
        nn.Softplus(),
        nn.Conv2d(16, 32, 3, stride=2, padding=1),
        nn.Softplus(),
        nn.Conv2d(32, 32, 3, stride=2, padding=1),
        nn.Flatten(),
        nn.Linear(512, 2*M),
    )

    def new_decoder():
        decoder_net = nn.Sequential(
            nn.Linear(M, 512),
            nn.Unflatten(-1, (32, 4, 4)),
            nn.Softplus(),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),
            nn.Softplus(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.Softplus(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
        )
        return decoder_net

    # Define VAE model
    encoder = GaussianEncoder(encoder_net)
    decoder = BernoulliDecoder(new_decoder())
    model = VAE(prior, decoder, encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer

        # Train model
        train(model, optimizer, mnist_train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), args.model)

    elif args.mode == 'plot':
        import matplotlib.pyplot as plt
        ## Load trained model
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        model.eval()

        ## Encode test and train data
        latents, labels = [], []
        with torch.no_grad():
            for x, y in mnist_train_loader:
                z = model.encoder(x)
                latents.append(z.mean)
                labels.append(y)
            latents = torch.concatenate(latents, dim=0)
            labels = torch.concatenate(labels, dim=0)

        ## Plot training data
        fig, ax = plt.subplots()
        for k in range(num_classes):
            idx = labels == k
            ax.scatter(latents[idx, 0], latents[idx, 1])
            
        ## Plot random geodesics
        num_curves = 1
        curve_indices = torch.randint(num_train_data, (num_curves, 2))  # (num_curves) x 2
        for k in range(num_curves):
            i = curve_indices[k, 0]
            j = curve_indices[k, 1]
            z0 = latents[i] 
            z1 = latents[j]

            order = 4
            N = 30
            curve, t, weights = create_curve(z0, z1, N = N, order = order) # 2 x num_points [[c0_x, c0_y], [c1_x, c1_y], ..., [cN_x, cN_y]]
            weights.requires_grad = True
            print(f"points along z0->z1 given curve c:{curve}") 
            print(f"init params:{weights}") 
            
            # TODO: Minimize energy of each curve.
            def closure():
                optimizer.zero_grad()
                energy = fr_energy(model,
                                   z0, z1, t, weights, order)
                #energy.requires_grad = True
                print(f"Energy: {energy}")
                energy.backward()
                return energy
            
            optimizer = LBFGS([weights], lr=1, line_search_fn='strong_wolfe') # line_search_fn='strong_wolfe'
            
            for _ in range(10): 
                optimizer.step(closure)

            ax.scatter(z0[0], z0[1], c='r')
            ax.scatter(z1[0], z1[1], c='r')
            ax.plot(curve[:, 0], curve[:, 1], 'r')
            print(f"init params:{weights}") 
        
        plt.savefig(args.plot)

