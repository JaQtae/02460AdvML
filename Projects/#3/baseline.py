import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import networkx as nx
from tqdm import tqdm
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load data
dataset = TUDataset(root='./data/', name='MUTAG').to(device)
node_feature_dim = 7

# Split into training and validation
rng = torch.Generator().manual_seed(0)
train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator = rng)

# Create dataloader for training and validation
train_loader = DataLoader(train_dataset, batch_size = 100)
validation_loader = DataLoader(validation_dataset, batch_size = 44)
test_loader = DataLoader(test_dataset, batch_size = 44)

class Erdos_renyi():
    def __init__(self, dataset: TUDataset):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset = dataset
        index = self.dataset.indices
        self.data = self.dataset.dataset[index]

        self.node_counts = torch.tensor([data.num_nodes for data in self.data]).to(self.device)
        self.edge_counts = torch.tensor([data.num_edges for data in self.data]).to(self.device)
        # Compute graph density
        # Graph density is the ratio of the number of edges to the number of possible edges
        # https://www.baeldung.com/cs/graph-density
        # Density = 2*|E|/(|V|*(|V|-1)) (for the undirected case) if directed just remove 2*
        self.densities = torch.tensor([2*self.edge_counts[i] / (self.node_counts[i] * 
                        (self.node_counts[i]-1)) for i in range(len(self.node_counts))]).to(self.device)

    def generate_graph(self):
        N = torch.randint(low = self.node_counts.min(), high = self.node_counts.max(), size = (1,)).to(self.device)
        # random sample a element from densities
        idx = torch.randint(low = 0, high = len(self.densities), size = (1,)).to(self.device)
        r = self.densities[idx].to(self.device)

        return torch.rand(N, N, device = self.device) < r
    
    def novel_and_unique(self, N: int = 1000):
        hashes_baseline = []
        hashes_train = []

        # Generate and hash graphs
        for i in tqdm(range(N), desc = "Generating and hashing graphs"):
            A = self.generate_graph()
            G = nx.from_numpy_array(A.cpu().numpy())
            hashes_baseline.append(nx.weisfeiler_lehman_graph_hash(G))

        # hash training graphs
        for data in tqdm(self.data, desc = "Hashing training graphs"):
            edge_list = data.edge_index.cpu().numpy()
            edges = list(map(tuple, edge_list.T))
            G = nx.Graph()
            G.add_edges_from(edges)
            hashes_train.append(nx.weisfeiler_lehman_graph_hash(G))
        

        hashes_baseline = np.array(hashes_baseline)
        hashes_train = np.array(hashes_train)
        # TODO: strange we see such a high number of unique graphs
        # Follow up on this!

if __name__ == '__main__':
    # Generate a graph
    er = Erdos_renyi(train_dataset)
    A = er.generate_graph().cpu().numpy()
    G = nx.from_numpy_array(A)

    er.novel_and_unique()

    nx.draw_spring(G)
    plt.show()

