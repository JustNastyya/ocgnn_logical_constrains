from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool

class GraphOCGINAdditionalAttribute(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, device):
        super().__init__()

        self.device = device
        self.num_layers = num_layers

        self.convs = nn.ModuleList()

        for i in range(num_layers):
            input_dim = in_dim + 1 if i == 0 else hidden_dim

            mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

            self.convs.append(GINConv(mlp))

        # center of Deep SVDD
        self.register_buffer("center", torch.zeros(hidden_dim))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        # graph-level embedding
        z = global_mean_pool(x, batch)

        return z

    def loss(self, z):
        return torch.mean(torch.sum((z - self.center) ** 2, dim=1))

    @torch.no_grad()
    def init_center(self, loader, L_constrains):
        self.eval()
        n_samples = 0
        center = torch.zeros_like(self.center)

        for data in loader:
            # the hole idea of the model
            batch_idx = data.idx
            batch_constraints = L_constrains[batch_idx]
            data.x = torch.cat([data.x, batch_constraints], dim=1)

            data = data.to(self.device)
            z = self.forward(data)
            center += z.sum(dim=0)
            n_samples += z.size(0)

        self.center.copy_(center / n_samples)

    def anomaly_score(self, z):
        return torch.sum((z - self.center) ** 2, dim=1)


def train_graph_ocgin_additional_attribute(model, loader, epochs, lr, constrains_obj, dataset):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    L_constrains = constrains_obj.get_constraint_value(loader)
    L_constrains = L_constrains.unsqueeze(1)
    
    model.init_center(loader, L_constrains)
    

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for data in loader:
            batch_idx = data.idx
            batch_constraints = L_constrains[batch_idx].to(model.device)
            
            data.x = torch.cat([data.x, batch_constraints], dim=1)

            data = data.to(model.device)
            z = model(data)
            loss = model.loss(z)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        logger.debug(f"Epoch {epoch:03d} | Loss {total_loss / len(loader):.6f}")
