import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.distributions as td
from tqdm import tqdm
import networkx as nx
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


plt.ion() # Enable interactive plotting
def drawnow():
    """Force draw the current plot."""
    plt.gcf().canvas.draw()
    plt.gcf().canvas.flush_events()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load data
dataset = TUDataset(root='./data/', name='MUTAG').to(device)
node_feature_dim = 7

# Split into training and validation
rng = torch.Generator().manual_seed(0)
train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator=rng)

# Create dataloader for training and validation
train_loader = DataLoader(train_dataset, batch_size=100)
validation_loader = DataLoader(validation_dataset, batch_size=44)
test_loader = DataLoader(test_dataset, batch_size=44)





class GCN(torch.nn.Module):
    def __init__(self, node_feature_dim, state_dim):
        super(GCN, self).__init__()
        self.node_feature_dim = node_feature_dim
        self.state_dim = state_dim
        
        self.linear1 = torch.nn.Linear(self.node_feature_dim, self.state_dim)
        self.conv1 = GCNConv(self.state_dim, self.state_dim)
        self.conv2 = GCNConv(self.state_dim, self.state_dim)
        # self.linear2 = torch.nn.Linear(self.state_dim,1)

        self.mu = torch.nn.Linear(self.state_dim, self.state_dim)
        self.logvar = torch.nn.Linear(self.state_dim, self.state_dim)
    
    def forward(self, x, edge_index, batch):
        num_nodes = x.shape[0]
        x = self.linear1(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        # x = self.linear2(x)
        
        mean = self.mu(x)
        logvar = self.logvar(x)

        z = self.reparametrize(mean, logvar)
        
        # decoder
        z_dc = torch.sigmoid(torch.mm(z, z.t())) # bias? 
        
        return z, z_dc, mean, logvar

    def kl_loss(self, mu, logvar):
        # KL divergence between q(z|x) and p(z)
        # This assumes a standard normal prior p(z)
        return -0.5 * torch.mean(torch.sum(1 + 2 * logvar - mu**2 - logvar.exp()**2, dim=1))
    
    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def sample(self, n_samples):
        z = torch.randn(n_samples, self.state_dim)
        return torch.sigmoid(torch.mm(z, z.t()))
        




state_dim = 5

model = GCN(node_feature_dim=node_feature_dim, state_dim=8).to(device)
print(model)



# Loss function
# mse_loss = F.mse_loss()
mse_loss = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)#, weight_decay=5e-4)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

epochs = 10000
loss_list = []

for epoch in tqdm(range(epochs), desc='Epoch'):
    # Loop over training batches
    model.train()
    train_accuracy = 0.
    train_loss = 0.
    for data in train_loader:
        z, z_dc, mu, logstd = model(data.x, data.edge_index, batch=data.batch)
        loss = model.kl_loss(mu, logstd)
        loss_list.append(loss.detach().cpu().item())    
        # loss = cross_entropy(out, data.y.float())

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

        # Gradient step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    # Learning rate scheduler step
    scheduler.step()

# Fit a torch distribution to the latent space


fig, ax = plt.subplots()
ax.plot(loss_list)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Loss over training')
plt.show()





# def train():
#     model.train()
#     optimizer.zero_grad()  # Clear gradients.
#     out = model(data.x, data.edge_index)  # Perform a single forward pass.
#     loss = mse_loss(out, data.y)  # Compute the loss solely based on the training nodes.
#     loss.backward()  # Derive gradients.
#     optimizer.step()  # Update parameters based on gradients.
#     scheduler.step() # Update the learning rate.
#     return loss



# for epoch in range(epochs):
#     loss = train()
#     if epoch % 1000 == 0:
#         print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')













class SimpleGNN(torch.nn.Module):
    """Simple graph neural network for graph classification."""

    def __init__(self, node_feature_dim, state_dim, num_message_passing_rounds):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.state_dim = state_dim
        self.num_message_passing_rounds = num_message_passing_rounds

        # Input network
        self.input_net = torch.nn.Sequential(
            torch.nn.Linear(self.node_feature_dim, self.state_dim),
            torch.nn.ReLU()
        )

        # Message networks
        self.message_net = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(self.state_dim, self.state_dim),
                torch.nn.ReLU()
            ) for _ in range(num_message_passing_rounds)
        ])

        # GRU-like update mechanism
        self.update_net = torch.nn.ModuleList([
            GRUUpdate(self.state_dim) for _ in range(num_message_passing_rounds)
        ])

        self.mu = torch.nn.Linear(self.state_dim, self.state_dim)
        self.logvar = torch.nn.Linear(self.state_dim, self.state_dim)

        # State output network
        self.output_net = torch.nn.Linear(self.state_dim, 1)

    def forward(self, x, edge_index, batch):
        num_graphs = batch.max() + 1
        num_nodes = batch.shape[0]
        x = torch.nn.functional.dropout(x, p = 0.25, training = True)
        state = self.input_net(x)

        for r in range(self.num_message_passing_rounds):
            message = self.message_net[r](state)
            aggregated = x.new_zeros((num_nodes, self.state_dim))
            aggregated = aggregated.index_add(0, edge_index[1], message[edge_index[0]])
            state = self.update_net[r](state, aggregated)

        
        mean = self.mu(state)
        logvar = self.logvar(state)

        z = self.reparametrize(mean, logvar)
        # decoded
        z_dc = torch.sigmoid(torch.mm(z, z.t())) # bias? 

        return z, z_dc, mean, logvar
        # Aggregate graph states
        graph_state = x.new_zeros((num_graphs, self.state_dim))
        graph_state = torch.index_add(graph_state, 0, batch, state)
        out = self.output_net(graph_state).flatten()
        return out
    
    def kl_loss(self, mu, logvar):
        # KL divergence between q(z|x) and p(z)
        # This assumes a standard normal prior p(z)
        return -0.5 * torch.mean(torch.sum(1 + 2 * logvar - mu**2 - logvar.exp()**2, dim=1))
    
    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def sample(self, n_samples):
        z = torch.randn(n_samples, self.state_dim)
        return torch.sigmoid(torch.mm(z, z.t()))

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
    
if __name__ == '__main__':
    state_dim = 3
    num_message_passing_rounds = 14
    model = SimpleGNN(node_feature_dim, state_dim, num_message_passing_rounds).to(device)

    # Loss function
    cross_entropy = torch.nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

        # Number of epochs
    epochs = 500

    loss_list = []

    for epoch in tqdm(range(epochs), desc='Epoch'):
        # Loop over training batches
        model.train()
        train_accuracy = 0.
        train_loss = 0.
        for data in train_loader:
            z, z_dc, mu, logstd = model(data.x, data.edge_index, batch=data.batch)
            loss = model.kl_loss(mu, logstd)
            loss_list.append(loss.detach().cpu().item())    
            # loss = cross_entropy(out, data.y.float())

            if epoch % 10 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

            # Gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # Learning rate scheduler step
        scheduler.step()

    # Fit a torch distribution to the latent space

    
    fig, ax = plt.subplots()
    ax.plot(loss_list)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss over training')
    plt.show()






