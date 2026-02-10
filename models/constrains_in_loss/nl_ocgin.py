from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool


class NodeOCGINLossConstrains(nn.Module):
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
                nn.Linear(hidden_dim, hidden_dim, bias=False),
            )

            self.convs.append(GINConv(mlp))

        # center of Deep SVDD
        self.register_buffer("center", torch.zeros(hidden_dim))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
    
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        return x

    def loss_add_constrain(self, z, constrain_L):
        return torch.mean(torch.sum((z - self.center) ** 2, dim=1)) + constrain_L.mean()

    def loss_node_defined_constraint(self, z, constrain_L):
        return torch.mean(torch.sum((z - self.center) ** 2, dim=1) + constrain_L)

    @torch.no_grad()
    def init_center(self, data, train_mask):
        self.eval()

        data = data.to(self.device)
        z = self.forward(data)
        center = z[train_mask].sum(dim=0)
        n_samples = z.size(0)

        self.center.copy_(center / n_samples)

    def anomaly_score(self, z):
        return torch.sum((z - self.center) ** 2, dim=1)


# wrapper
def train_node_ocgin_add_loss_constrains(model, data, train_mask, test_mask, epochs, lr, constrains_obj):
    train_node_ocgin_loss_constrains(model, data, train_mask, test_mask, epochs, lr, constrains_obj, loss_type="add")

# wrapper    
def train_node_ocgin_node_defined_loss_constrains(model, data, train_mask, test_mask, epochs, lr, constrains_obj):
    train_node_ocgin_loss_constrains(model, data, train_mask, test_mask, epochs, lr, constrains_obj, loss_type="node_certain")


def train_node_ocgin_loss_constrains(model, data, train_mask, test_mask, epochs, lr, constrains_obj, loss_type="add"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    model.init_center(data, train_mask)

    L_constrains = constrains_obj.get_constraint_value(data)
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        data = data.to(model.device)
        z = model(data)
        if loss_type == "add":
            loss = model.loss_add_constrain(z[train_mask], L_constrains[train_mask])
        elif loss_type == "node_certain":
            loss = model.loss_node_defined_constraint(z[train_mask], L_constrains[train_mask])
            
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if epoch % 10 == 0:
            logger.debug(f"Epoch {epoch:03d} | Loss {total_loss / len(data):.6f}")
            
