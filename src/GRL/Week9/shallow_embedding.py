# %% Programming excercise: Shallow embedding

# %% Import libraries
import torch
from tqdm import tqdm

# Changing directory and stuff just cuz
import os
dir_name = os.path.dirname(os.path.abspath(__file__).replace('\\','/'))
os.chdir(dir_name)


# %% Device
device = 'cpu'

# %% Load graph data
# Load graph from file
A = torch.load('data.pt')

# Get number of nodes
n_nodes = A.shape[0]

# Number of un-ordered node pairs (possible links)
n_pairs = n_nodes*(n_nodes-1)//2

# Get indices of all un-ordered node pairs excluding self-links (shape: 2 x n_pairs)
# NOTE: triu is upper triangular matrix --> 2xN Tensor:
#   https://pytorch.org/docs/stable/generated/torch.triu_indices.html
idx_all_pairs = torch.triu_indices(n_nodes,n_nodes,1)

# Collect all links/non-links in a list (shape: n_pairs)
target = A[idx_all_pairs[0],idx_all_pairs[1]]

# %% Shallow node embedding
class Shallow(torch.nn.Module):
    '''Shallow node embedding

    Args: 
        n_nodes (int): Number of nodes in the graph
        embedding_dim (int): Dimension of the embedding
    '''
    def __init__(self, n_nodes, embedding_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(n_nodes, embedding_dim=embedding_dim)
        self.bias = torch.nn.Parameter(torch.tensor([0.]))

    def forward(self, rx, tx):
        '''The link probability between pairs of nodes, where rx and tx represent the indices of nodes.'''
        return torch.sigmoid((self.embedding.weight[rx]*self.embedding.weight[tx]).sum(1) + self.bias)

# Embedding dimension
embedding_dim = 1
loss_at_embed_dim = []

for e in range(8):
    embedding_dim = embedding_dim*2 

    # Instantiate the model                
    model = Shallow(n_nodes, embedding_dim)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # Loss function
    cross_entropy = torch.nn.BCELoss()

    # %% Fit the model
    # Number of gradient steps
    max_step = 600
    
    loss_e = []
    # Optimization loop
    for i in (progress_bar := tqdm(range(max_step))):    
        # Compute probability of each possible link
        link_probability = model(idx_all_pairs[0], idx_all_pairs[1])

        # Cross entropy loss
        loss = cross_entropy(link_probability, target)

        # Gradient step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Display loss on progress bar
        progress_bar.set_description(f'Embed dim: {embedding_dim} | Loss = {loss.item():.3f}')
        loss_e.append(loss.item())
    loss_at_embed_dim.append((embedding_dim, loss_e))

import matplotlib.pyplot as plt
# Plotting
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Loss Curves for Different Embedding Dimensions', fontsize=16)

for i, (embed_dim, loss_values) in enumerate(loss_at_embed_dim):
    ax = axes[i // 4, i % 4]
    ax.plot(loss_values, color='b', linewidth=2, label=f'Embedding Dim: {embed_dim}')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('EmbedDimLosses.png', dpi=300)
plt.show()

# %% Save final estimated link probabilities
link_probability = model(idx_all_pairs[0], idx_all_pairs[1])
torch.save(link_probability, 'link_probability.pt')