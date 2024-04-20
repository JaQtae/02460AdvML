# %% Programming excercise: Shallow embedding

# %% Import libraries
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# Changing directory and stuff just cuz
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


# %% Question D.4

# Split indices into training and validation sets (80/20)
train_idx, val_idx = train_test_split(range(n_pairs), test_size=0.2, random_state=42)

# Embedding dimension
embed_dims = [2**i for i in range(1,9)]

validation_losses = {}
trained_models = {}
loss_at_embed_dim = []


# Early stopping parameters
patience = 10 
counter = 0  # No. epochs with no improvement
best_val_loss = float('inf')


for embed_dim in embed_dims:
    # Instantiate the model                
    model = Shallow(n_nodes, embed_dim)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                           patience=5, factor=0.5, 
                                                           verbose=True)

    # Loss function
    cross_entropy = torch.nn.BCELoss()
    loss_e = []
    
    # Number of gradient steps
    max_step = 1000

    # Optimization loop
    for i in (progress_bar := tqdm(range(max_step))):    
        # Compute probability of each possible link
        train_link_probability = model(idx_all_pairs[0][train_idx], 
                                    idx_all_pairs[1][train_idx])

        # Cross entropy loss
        train_loss = cross_entropy(train_link_probability, target[train_idx])

        # Gradient step
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # Display loss on progress bar
        progress_bar.set_description(f'Embed dim: {embed_dim} | Loss = {train_loss.item():.3f}')
        loss_e.append(train_loss.item())
        
    loss_at_embed_dim.append((embed_dim, loss_e))

    val_link_probability = model(idx_all_pairs[0][val_idx], idx_all_pairs[1][val_idx])
    val_loss = cross_entropy(val_link_probability, target[val_idx])
    
    scheduler.step(val_loss)
    
    # Check for improvement in validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0  # Reset counter
        # Only save the models with the newest best validation loss
        trained_models[embed_dim] = {'model': model, 'validation_loss': val_loss.item()}
    else:
        counter += 1
        if counter >= patience:
            print(f'Validation loss did not improve for {patience} epochs. Early stopping...')
            break
    

# Find the optimal embedding dimension with the lowest validation loss
optimal_embedding_dim = min(trained_models, key = lambda x: trained_models[x]['validation_loss'])

best_model = trained_models[optimal_embedding_dim]['model']
best_validation_loss = trained_models[optimal_embedding_dim]['validation_loss']

print(f'Best model:\nEmbedding Dimension: {optimal_embedding_dim} | Validation loss: {best_validation_loss:.3f}')

# Save best model
torch.save(best_model.state_dict(), f'bestModel_{optimal_embedding_dim}_{max_step}.pt')



# Plotting training losses for all the models (Question D.3)
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Training Loss Curves for Different Embedding Dimensions', fontsize=16)

for i, (embed_dim, loss_values) in enumerate(loss_at_embed_dim):
    ax = axes[i // 4, i % 4]
    ax.plot(loss_values, color='b', linewidth=2, label=f'Embedding Dim: {embed_dim}')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'Embedding Dim: {embed_dim}', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('EmbedDimTrainLosses.png', dpi=300)
plt.show()

# %% Save final estimated link probabilities
link_probability = best_model(idx_all_pairs[0], idx_all_pairs[1])
torch.save(link_probability, 'link_probability.pt')
