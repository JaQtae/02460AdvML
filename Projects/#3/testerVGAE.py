import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.datasets import ZINC, TUDataset
from torch_geometric.nn import VGAE, GCNConv
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from torch.utils.data import random_split

device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
class GRUUpdate(torch.nn.Module):
    """GRU-like state update for GNN."""

    def __init__(self, state_dim):
        super().__init__()
        self.state_dim = state_dim
        self.reset_gate = torch.nn.Sequential(
            torch.nn.Linear(2 * state_dim, state_dim),
            torch.nn.Sigmoid()
        )
        self.update_gate = torch.nn.Sequential(
            torch.nn.Linear(2 * state_dim, state_dim),
            torch.nn.Sigmoid()
        )
        self.transform = torch.nn.Sequential(
            torch.nn.Linear(2 * state_dim, state_dim),
            torch.nn.Tanh()
        )

    def forward(self, prev_state, new_state):
        combined_state = torch.cat([prev_state, new_state], dim=1)
        reset = self.reset_gate(combined_state)
        update = self.update_gate(combined_state)
        combined_reset = torch.cat([prev_state * reset, new_state], dim=1)
        candidate = self.transform(combined_reset)
        new_state = prev_state * (1 - update) + candidate * update
        return new_state
    

class VariationalGraphEncoder(torch.nn.Module):
    """
    This is the baseline encoder, which uses linear layers
    and a GRU update on a message passing using node-level latents.
    """
    def __init__(self, in_channels, state_dim, num_message_passing_rounds):
        super().__init__()
        self.in_channels = in_channels
        self.state_dim = state_dim
        self.num_message_passing_rounds = num_message_passing_rounds
        
        # Input network
        self.input_net = torch.nn.Sequential(
            torch.nn.Linear(self.in_channels, self.state_dim),
            torch.nn.ReLU()
        )

        # Message networks
        self.message_net = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.state_dim, self.state_dim),
                torch.nn.ReLU()
            ) for _ in range(self.num_message_passing_rounds)
        ])

        # GRU-like update mechanism
        self.update_net = torch.nn.ModuleList([
            GRUUpdate(self.state_dim) for _ in range(self.num_message_passing_rounds)
        ])
        
    def forward(self, x, edge_index, batch):
        # x is the adjacency matrix A.
        num_nodes = batch.shape[0] # |V|
        #x = torch.nn.functional.dropout(x, p = 0.25, training = True)
        state = self.input_net(x)

        for r in range(self.num_message_passing_rounds):
            message = self.message_net[r](state)
            aggregated = x.new_zeros((num_nodes, self.state_dim))
            aggregated = aggregated.index_add(0, edge_index[1], message[edge_index[0]])
            state = self.update_net[r](state, aggregated)
            
        return state
    
    
class VGAE_Linear(torch.nn.Module):
    """Variational Graph Auto Encoder (with Linear layers).
    A version of the VGAE paper's model, see: https://arxiv.org/abs/1611.07308
    
    The state_network is our own encoder, which uses linear layers and GRU for message passing.
    The decoder is the InnerProductDecoder(), see PyG. ( sigmoid(z, z^T) )
    """
    def __init__(self, in_channels, state_dim, num_message_passing_rounds):
        super().__init__()
        self.in_channels = in_channels
        self.state_dim = state_dim
        self.num_message_passing_rounds = num_message_passing_rounds
        
        self.state_network = VariationalGraphEncoder(self.in_channels,
                                               self.state_dim,
                                               self.num_message_passing_rounds)

        self.mu = torch.nn.Linear(self.state_dim, self.state_dim)
        self.logvar = torch.nn.Linear(self.state_dim, self.state_dim)


    def forward(self, x, edge_index, batch):
        state = self.state_network.forward(x, edge_index, batch) # week 10

        #z = self.reparametrize(mean, logvar) # This is taken care of if we use PyG's VGAE/GAE. Less code.
        return self.mu(state), self.logvar(state) # Tuple([|V| x d], [|V| x d])


# transform = T.Compose([
#     T.RandomLinkSplit(num_val=0, num_test=0.1, is_undirected=True,
#                       split_labels=True, add_negative_train_samples=True)
# ])

dataset = TUDataset(root='./data/', name='MUTAG').to(device)
num_message_passing_rounds = 3

# We do not do validation, hence we append these to training.

# Split into training and test
rng = torch.Generator().manual_seed(0)
train_dataset, _, test_dataset = random_split(dataset, (144, 0, 44), generator=rng)
 
# Create dataloader for training and test
train_loader = DataLoader(train_dataset, batch_size=1) # 1 while testing
test_loader = DataLoader(test_dataset, batch_size=1)

in_channels, state_dim, lr, n_epochs = dataset.num_features, 28, 1e-2, 50
model = VGAE(VGAE_Linear(in_channels, state_dim, num_message_passing_rounds))

# Loss function
cross_entropy = torch.nn.BCEWithLogitsLoss() # Not used directly

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

loss_list = []

from tqdm import tqdm
for epoch in tqdm(range(n_epochs), desc='Epoch'):
        # Loop over training batches
        model.train()
        train_accuracy = 0.
        train_loss = 0.
        for data in train_loader:
            z = model.encode(data.x, data.edge_index, batch=data.batch)
            #... Wouldn't the loss more be the difference between our reconstruction and the original graph?
            # Unsure if this is actually anywhere near what this does...
            loss = model.recon_loss(z, data.edge_index) # BCE loss for positive/negative edge samples. (??)
            loss = loss + (1 / data.num_nodes) * model.kl_loss()
            loss_list.append(loss.detach().cpu().item())  
            if epoch % 10 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        scheduler.step()

fig, ax = plt.subplots()
ax.plot(loss_list)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Loss over training')
plt.show()


def generate_z(min_nodes=10, max_nodes=28, n_samples=1000):
    """Generates the z from a standard normal (prior) distribution with a given number of nodes,
    which follows a random distribution between the min and max of the MUTAG dataset.
    TODO: Swap this to the empirical distribution of MUTAG"""
    res = []
    for _ in range(n_samples):
        n_nodes = np.random.randint(min_nodes, max_nodes)
        res.append(torch.normal(0, 1, size=(n_nodes, state_dim)))
    return res

@torch.no_grad
def generate_graph_from_z(z_list, threshold = 0.5):
    """Takes latents and decodes them with InnerProductDecoder. 
    More certainty than given threshold puts an edge between two nodes.
    
    Returns:
        A_hats: A list of all the adjacency matrices predicted by the model given z
        graphs: NetworkX objects of the A_hats
    """
    A_hats = []
    graphs = []
    for z in z_list:
        A_prob = model.decoder.forward_all(z) # InnerProductDecoder [sigmoid(z@z.T)] for all z's. 
        # Gives back probabilities of edges between each node in the graph of size given by generate_z()
        A_hat = torch.where(A_prob > threshold, 1, 0)
        A_hats.append(A_hat)
        graphs.append(nx.Graph(A_hat.numpy()))
    return A_hats, graphs


zs = generate_z()
adjs, graphs = generate_graph_from_z(zs)


n_graphs_plot = 5

fig, axes = plt.subplots(1, n_graphs_plot, figsize=(15, 3))

for i in range(n_graphs_plot):
    ax = axes[i]
    nx.draw(graphs[i], ax=ax)
    ax.set_title(f'Graph {i+1}')
plt.tight_layout()
plt.show()

