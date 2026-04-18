from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool


class GraphOCGINLossConstrains(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, device):
        super().__init__()

        self.device = device
        self.num_layers = num_layers

        self.convs = nn.ModuleList()

        for i in range(num_layers):
            input_dim = in_dim if i == 0 else hidden_dim

            mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

            self.convs.append(GINConv(mlp))

        # center of Deep SVDD
        self.register_buffer("center", torch.zeros(hidden_dim))

    def set_constrains_params(self, constrains_obj, loss_type, constrains_cache=None):
        self.constrains_obj = constrains_obj
        self.loss_type = loss_type
        self.constrains_cache = constrains_cache

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        # graph-level embedding
        z = global_mean_pool(x, batch)

        return z
    
    def loss_add_constrain(self, z, constrain_L):
        return torch.mean(torch.sum((z - self.center) ** 2, dim=1)) + constrain_L.mean()

    def loss_graph_weighting(self, z, constrain_L):
        return torch.mean(torch.sum((z - self.center)**2, dim=1) * (1 + constrain_L))

    def loss_graph_irnoring_sus(self, z, constrain_L):
        return torch.mean(torch.sum((z - self.center) ** 2, dim=1) * (1 - constrain_L))

    @torch.no_grad()
    def init_center(self, loader):
        self.eval()
        n_samples = 0
        center = torch.zeros_like(self.center)

        for data in loader:
            data = data.to(self.device)
            z = self.forward(data)
            center += z.sum(dim=0)
            n_samples += z.size(0)

        self.center.copy_(center / n_samples)

    def anomaly_score(self, z, batch_idx):
        base_score = torch.sum((z - self.center) ** 2, dim=1)

        batch_constraints = self.constrains_cache[batch_idx].to(self.device)

        if self.loss_type == "add":
            return base_score + batch_constraints
        elif self.loss_type == "node_weighting":
            return base_score * (1 + batch_constraints)
        elif self.loss_type == "node_irnoring_sus":
            return base_score * (1 - batch_constraints)

# wrapper
def train_graph_ocgin_add_loss_constrains(model, loader, epochs, lr, constrains_obj, dataset):
    train_graph_ocgin_loss_constrains(model, loader, epochs, lr, constrains_obj, dataset, loss_type="add")

# wrapper    
def train_graph_ocgin_weighting(model, loader, epochs, lr, constrains_obj, dataset):
    train_graph_ocgin_loss_constrains(model, loader, epochs, lr, constrains_obj, dataset, loss_type="node_weighting")

# wrapper    
def train_graph_ocgin_irnoring_sus(model, loader, epochs, lr, constrains_obj, dataset):
    train_graph_ocgin_loss_constrains(model, loader, epochs, lr, constrains_obj, dataset, loss_type="node_irnoring_sus")


def train_graph_ocgin_loss_constrains(model, loader, epochs, lr, constrains_obj, dataset, loss_type="add"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.init_center(loader)
    L_constrains = constrains_obj.get_constraint_value(dataset)
    model.set_constrains_params(constrains_obj, loss_type, L_constrains)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for data in loader:
            data = data.to(model.device)
            z = model(data)
            
            batch_idx = data.idx
            batch_constraints = L_constrains[batch_idx].to(model.device)
            
            if loss_type == "add":
                loss = model.loss_add_constrain(z, batch_constraints)
            elif loss_type == "node_weighting":
                loss = model.loss_graph_weighting(z, batch_constraints)
            elif loss_type == "node_irnoring_sus":
                loss = model.loss_graph_irnoring_sus(z, batch_constraints)            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0:
            logger.debug(f"Epoch {epoch:03d} | Loss {total_loss / len(data):.6f}")
