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
        idx_N = torch.randint(low = self.node_counts.min(), high = self.node_counts.max(), size = (1,)).to(self.device)
        N = self.node_counts[idx_N].item()
        # random sample a element from densities
        idx_r = torch.randint(low = 0, high = len(self.densities), size = (1,)).to(self.device)
        r = self.densities[idx_r].item()
        A = torch.rand(N, N, device = self.device) < r
        G = nx.from_numpy_array(A.cpu().numpy())
        return G
    
    def novel_and_unique(self, N: int = 1000):
        hashes_baseline = []
        hashes_train = []

        # Generate and hash graphs
        node_degree_gen = []
        clustering_gen = []
        eigenvector_gen = []
        for i in tqdm(range(N), desc = "Generating and hashing graphs"):
            G = self.generate_graph()
            hashes_baseline.append(nx.weisfeiler_lehman_graph_hash(G))

            # node degree
            node_degree = dict(G.degree())
            node_degree_gen += list(node_degree.values())

            # Cluster coefficient
            clustering = nx.clustering(G)
            clustering_gen += list(clustering.values())

            # Eigenvector centrality
            eigenvector = nx.eigenvector_centrality(G, max_iter = 1000)
            eigenvector_gen += list(eigenvector.values())

        # hash training graphs
        node_degree_train = []
        clustering_train = []
        eigenvector_train = []
        for data in tqdm(self.data, desc = "Hashing training graphs"):
            edge_list = data.edge_index.cpu().numpy()
            edges = list(map(tuple, edge_list.T))
            G = nx.Graph()
            G.add_edges_from(edges)
            hashes_train.append(nx.weisfeiler_lehman_graph_hash(G))

            # node degree
            node_degree = dict(G.degree())
            node_degree_train += list(node_degree.values())
            # Cluster coefficient
            clustering = nx.clustering(G)
            clustering_train += list(clustering.values())
            
            # Eigenvector centrality
            eigenvector = nx.eigenvector_centrality(G, max_iter = 1000)
            eigenvector_train += list(eigenvector.values())	

        # Node degree histogram
        min_bin = min(min(node_degree_gen), min(node_degree_train))
        max_bin = max(max(node_degree_gen), max(node_degree_train))
        n_bins = 20
        bin_width = (max_bin - min_bin) / n_bins
        bins = np.arange(min_bin, max_bin + bin_width, bin_width)
        plt.figure(figsize=(10, 6))
        sns.histplot(node_degree_gen, color="blue", label='Generated', kde=False, alpha=0.5, bins = bins, stat = "probability")
        sns.histplot(node_degree_train, color="red", label='Training', kde=False, alpha=0.5, bins = bins, stat = "probability")
        plt.xlabel('Node Degree')
        plt.ylabel('Probability')
        plt.title("Node degree distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig("node_degree_distribution.png")

        # Clustering histogram
        min_bin = min(min(clustering_gen), min(clustering_train))
        max_bin = max(max(clustering_gen), max(clustering_train))
        n_bins = 20
        bin_width = (max_bin - min_bin) / n_bins
        bins = np.arange(min_bin, max_bin + bin_width, bin_width)
        plt.figure(figsize=(10, 6))
        # Plot the histogram for 'node_degree_gen'
        sns.histplot(clustering_gen, color="blue", label='Generated', kde=False, alpha=0.5, bins = bins, stat = "probability")
        sns.histplot(clustering_train, color="red", label='Training', kde=False, alpha=0.5, bins = bins, stat = "probability")
        plt.xlabel('Clustering coefficient')
        plt.ylabel('Probability')
        plt.title("Clustering coefficient distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig("clustering_coefficients.png")

        # Eigenvector centrality histogram
        min_bin = min(min(eigenvector_gen), min(eigenvector_train))
        max_bin = max(max(eigenvector_gen), max(eigenvector_train))
        n_bins = 20
        bin_width = (max_bin - min_bin) / n_bins
        bins = np.arange(min_bin, max_bin + bin_width, bin_width)
        plt.figure(figsize=(10, 6))
        # Plot the histogram for 'node_degree_gen'
        sns.histplot(eigenvector_gen, color="blue", label='Generated', kde=False, alpha=0.5, bins = bins, stat = "probability")
        sns.histplot(eigenvector_train, color="red", label='Training', kde=False, alpha=0.5, bins = bins, stat = "probability")
        plt.xlabel('Eigenvector centrality')
        plt.ylabel('Probability')
        plt.title("Eigenvector centrality distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig("eigenvector_centrality.png")

        # Maximum disgreement between generated and training graphs
        # Half the L1 norm
        # TODO


        hashes_baseline = np.array(hashes_baseline)
        hashes_train = np.array(hashes_train)
        # TODO: strange we see such a high number of unique graphs
        # Follow up on this!

if __name__ == '__main__':
    # Generate a graph
    er = Erdos_renyi(train_dataset)
    G = er.generate_graph()

    er.novel_and_unique()

    nx.draw_spring(G)
    plt.show()

