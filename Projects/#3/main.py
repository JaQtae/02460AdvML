import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn

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

