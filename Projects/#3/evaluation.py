import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import networkx as nx
from tqdm import tqdm
import numpy as np
import seaborn as sns
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
from baseline import Erdos_renyi
from testerVGAE import VGAE_Linear, VariationalGraphEncoder, VGAE

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load data
dataset = TUDataset(root='./data/', name='MUTAG').to(device)
node_feature_dim = 7

# Split into training and validation
rng = torch.Generator().manual_seed(0)
train_dataset, validation_dataset, test_dataset = random_split(dataset, (144, 0, 44), generator = rng)

# Create dataloader for training and validation
train_loader = DataLoader(train_dataset, batch_size = 1)
#validation_loader = DataLoader(validation_dataset, batch_size = 44)
test_loader = DataLoader(test_dataset, batch_size = 1)

# Create baseline
baseline = Erdos_renyi(train_dataset)
node_degree_gen, node_degree_train, clustering_gen, clustering_train, eigenvector_gen, eigenvector_train, hashes_baseline, hashes_train = baseline.novel_and_unique(N = 1000, plotting = False)

# Create VGAE model
num_message_passing_rounds = 3
in_channels, state_dim, lr, n_epochs = dataset.num_features, 28, 1e-2, 250
model = VGAE(VGAE_Linear(in_channels, state_dim, num_message_passing_rounds)).to(device)

# Loss function
cross_entropy = torch.nn.BCEWithLogitsLoss() # Not used directly

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

loss_list = []

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

def generate_z(node_counts, n_samples=1000):
    """Generates the z from a standard normal (prior) distribution with a given number of nodes,
    which follows a random distribution between the min and max of the MUTAG dataset."""
    res = []
    for _ in range(n_samples):
        idx_N = torch.randint(low = 0, high = len(node_counts), size = (1,)).to(device)
        n_nodes = node_counts[idx_N].item()
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
        A_hat = torch.triu(A_hat, diagonal = 1) # ensure no self-loops
        A_hat = A_hat + A_hat.t() # create undirected graph
        A_hats.append(A_hat)
        graphs.append(nx.Graph(A_hat.numpy()))
    return A_hats, graphs

zs = generate_z(baseline.node_counts, n_samples = 1000)
adjs, graphs = generate_graph_from_z(zs)

node_degree_vgae = []
clustering_vgae = []
eigenvector_vgae = []
hashes_vgae = []
for graph in graphs:
    hashes_vgae.append(nx.weisfeiler_lehman_graph_hash(graph))
    # Node degree
    node_degree = dict(graph.degree())
    node_degree_vgae += list(node_degree.values())
    # Cluster coefficient
    clustering = nx.clustering(graph)
    clustering_vgae += list(clustering.values())

    # Eigenvector centrality
    eigenvector = nx.eigenvector_centrality(graph, max_iter = 1000)
    eigenvector_vgae += list(eigenvector.values())

min_bin = min(min(node_degree_gen), min(node_degree_train))
max_bin = max(max(node_degree_gen), max(node_degree_train))
n_bins = 20
bin_width = (max_bin - min_bin) / n_bins
bins = np.arange(min_bin, max_bin + bin_width, bin_width)

plt.figure()
sns.histplot(node_degree_gen, color="blue", label='Generated', kde=False, alpha=0.5, bins = bins, stat = "probability", discrete = True)
sns.histplot(node_degree_train, color="red", label='Training', kde=False, alpha=0.5, bins = bins, stat = "probability", discrete = True)
sns.histplot(node_degree_vgae, color="green", label='VGAE', kde=False, alpha=0.5, bins = bins, stat = "probability", discrete = True)
plt.xlabel('Node Degree', fontsize=16)
plt.ylabel('Probability', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("Node degree distribution", fontsize=18)
plt.legend(fontsize=18)
plt.tight_layout()
plt.savefig("node_degree_distribution.png")
plt.show()
plt.clf()

# Clustering histogram
min_bin = min(min(clustering_gen), min(clustering_train))
max_bin = max(max(clustering_gen), max(clustering_train))
n_bins = 20
bin_width = (max_bin - min_bin) / n_bins
bins = np.arange(min_bin, max_bin + bin_width, bin_width)
plt.figure()
# Plot the histogram for 'node_degree_gen'
sns.histplot(clustering_gen, color="blue", label='Generated', kde=False, alpha=0.5, bins = bins, stat = "probability")
sns.histplot(clustering_train, color="red", label='Training', kde=False, alpha=0.5, bins = bins, stat = "probability")
sns.histplot(clustering_vgae, color="green", label='VGAE', kde=False, alpha=0.5, bins = bins, stat = "probability")
plt.xlabel('Clustering coefficient', fontsize=16)
plt.ylabel('Probability', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("Clustering coefficient distribution", fontsize=18)
plt.legend(fontsize=18)
plt.tight_layout()
plt.savefig("clustering_coefficients.png")
plt.show()
plt.clf()

# Eigenvector centrality histogram
min_bin = min(min(eigenvector_gen), min(eigenvector_train))
max_bin = max(max(eigenvector_gen), max(eigenvector_train))
n_bins = 20
bin_width = (max_bin - min_bin) / n_bins
bins = np.arange(min_bin, max_bin + bin_width, bin_width)
plt.figure()
# Plot the histogram for 'node_degree_gen'
sns.histplot(eigenvector_gen, color="blue", label='Generated', kde=False, alpha=0.5, bins = bins, stat = "probability")
sns.histplot(eigenvector_train, color="red", label='Training', kde=False, alpha=0.5, bins = bins, stat = "probability")
sns.histplot(eigenvector_vgae, color="green", label='VGAE', kde=False, alpha=0.5, bins = bins, stat = "probability")
plt.xlabel('Eigenvector centrality', fontsize=16)
plt.ylabel('Probability', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title("Eigenvector centrality distribution", fontsize=18)
plt.legend(fontsize=18)
plt.tight_layout()
plt.savefig("eigenvector_centrality.png")
plt.show()
plt.clf()