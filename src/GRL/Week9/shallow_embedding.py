# %% Programming excercise: Shallow embedding

# %% Import libraries
import torch
from tqdm import tqdm

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
        '''Returns the probability of a links between nodes in lists rx and tx'''
        return torch.sigmoid((self.embedding.weight[rx]*self.embedding.weight[tx]).sum(1) + self.bias)

# Embedding dimension
embedding_dim = 2

# Instantiate the model                
model = Shallow(n_nodes, embedding_dim)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# Loss function
cross_entropy = torch.nn.BCELoss()

# %% Fit the model
# Number of gradient steps
max_step = 1000

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
    progress_bar.set_description(f'Loss = {loss.item():.3f}')

# %% Save final estimated link probabilities
link_probability = model(idx_all_pairs[0], idx_all_pairs[1])
torch.save(link_probability, 'link_probability.pt')