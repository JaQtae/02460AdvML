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
        node_degree_gen = {i: 0 for i in range(self.node_counts.max() + 1)}
        clustering_gen = {i: 0 for i in range(self.node_counts.max() + 1)}
        eigenvector_gen = {i: 0 for i in range(self.node_counts.max() + 1)}
        for i in tqdm(range(N), desc = "Generating and hashing graphs"):
            G = self.generate_graph()
            hashes_baseline.append(nx.weisfeiler_lehman_graph_hash(G))

            # node degree
            node_degree = dict(G.degree())
            for key in node_degree.keys():
                node_degree_gen[key] += node_degree[key]

            # Cluster coefficient
            clustering = nx.clustering(G)
            for key in clustering.keys():
                clustering_gen[key] += clustering[key]

            # Eigenvector centrality
            eigenvector = nx.eigenvector_centrality(G)
            for key in eigenvector.keys():
                eigenvector_gen[key] += eigenvector[key]

        # hash training graphs
        node_degree_train = {i: 0 for i in range(self.node_counts.max() + 1)}
        clustering_train = {i: 0 for i in range(self.node_counts.max() + 1)}
        eigenvector_train = {i: 0 for i in range(self.node_counts.max() + 1)}
        for data in tqdm(self.data, desc = "Hashing training graphs"):
            edge_list = data.edge_index.cpu().numpy()
            edges = list(map(tuple, edge_list.T))
            G = nx.Graph()
            G.add_edges_from(edges)
            hashes_train.append(nx.weisfeiler_lehman_graph_hash(G))

            # node degree
            node_degree = dict(G.degree())
            for key in node_degree.keys():
                node_degree_train[key] += node_degree[key]

            # Cluster coefficient
            clustering = nx.clustering(G)
            for key in clustering.keys():
                clustering_train[key] += clustering[key]
            
            # Eigenvector centrality
            eigenvector = nx.eigenvector_centrality(G, max_iter = 1000)
            for key in eigenvector.keys():
                eigenvector_train[key] += eigenvector[key]

        # Normalize with the respective number of graphs
        node_degree_gen = {key: value/N for key, value in node_degree_gen.items()}
        clustering_gen = {key: value/N for key, value in clustering_gen.items()}
        eigenvector_gen = {key: value/N for key, value in eigenvector_gen.items()}
        node_degree_train = {key: value/len(self.data) for key, value in node_degree_train.items()}
        clustering_train = {key: value/len(self.data) for key, value in clustering_train.items()}
        eigenvector_train = {key: value/len(self.data) for key, value in eigenvector_train.items()}

        # plot histograms
        fig, axs = plt.subplots(3, 2, figsize = (15, 15))
        axs[0, 0].bar(node_degree_gen.keys(), node_degree_gen.values())
        axs[0, 0].set_title("Node degree distribution of generated graphs")
        axs[0, 1].bar(node_degree_train.keys(), node_degree_train.values())
        axs[0, 1].set_title("Node degree distribution of training graphs")
        axs[1, 0].bar(clustering_gen.keys(), clustering_gen.values())
        axs[1, 0].set_title("Clustering coefficient distribution of generated graphs")
        axs[1, 1].bar(clustering_train.keys(), clustering_train.values())
        axs[1, 1].set_title("Clustering coefficient distribution of training graphs")
        axs[2, 0].bar(eigenvector_gen.keys(), eigenvector_gen.values())
        axs[2, 0].set_title("Eigenvector centrality distribution of generated graphs")
        axs[2, 1].bar(eigenvector_train.keys(), eigenvector_train.values())
        axs[2, 1].set_title("Eigenvector centrality distribution of training graphs")
        plt.show()

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

