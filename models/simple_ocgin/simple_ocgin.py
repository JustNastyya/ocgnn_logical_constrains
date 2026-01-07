import torch
import torch.nn as nn
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torch.utils.data import random_split
from loguru import logger


class OCGIN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, device):
        super().__init__()

        self.device = device
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.mlps = nn.ModuleList()

        for i in range(num_layers):
            input_dim = in_dim if i == 0 else hidden_dim

            mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

            self.convs.append(GINConv(mlp))
            self.mlps.append(mlp)

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

    def anomaly_score(self, z):
        return torch.sum((z - self.center) ** 2, dim=1)


def train_ocgin(model, loader, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.init_center(loader)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for data in loader:
            data = data.to(model.device)
            z = model(data)
            loss = model.loss(z)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch:03d} | Loss {total_loss / len(loader):.6f}")


def compute_anomaly_scores(model, loader):
    model.eval()
    scores = []

    with torch.no_grad():
        for data in loader:
            data = data.to(model.device)
            z = model(data)
            scores.append(model.anomaly_score(z).cpu())

    return torch.cat(scores, dim=0)


if __name__ == "__main__":

    # ------------------
    # CONFIGURATION
    # ------------------

    config = {
        "hidden_dim": 64,
        "num_layers": 3,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "lr": 1e-3,
        "epochs": 50,
        "batch_size": 32,
        "weight_decay": 1e-5,
        "norm_layer": "gn",
        "learning_rate": 0.001,
        "result_folder": "results",
        "save_scores": False,
        "num_repeat": 1,
        "shuffle": True,
        "bias": False,
        "loss": "OCGIN",
        "l2": 0,
        "aggregation": "add",
        "dataset": "MUTAG"
    }

    # ------------------
    # LOAD DATA
    # ------------------
    # IMPORTANT:
    # train_dataset must contain ONLY NORMAL GRAPHS
    logger.info("##################### Loading data")
    
    dataset = TUDataset(root="data/TUDataset", name=config["dataset"])    
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    NORMAL_LABEL = 0  # or 1

    normal_graphs = [d for d in dataset if d.y.item() == NORMAL_LABEL]
    anomalous_graphs = [d for d in dataset if d.y.item() != NORMAL_LABEL]

    num_normal = len(normal_graphs)
    train_size = int(0.8 * num_normal)
    test_size = num_normal - train_size

    train_dataset, test_normal = random_split(
        normal_graphs, [train_size, test_size]
    )

    test_dataset = test_normal + anomalous_graphs

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False
    )
    logger.info("##################### creating model")

    # ------------------
    # CREATE MODEL
    # ------------------
    dim_features = 7 # train_dataset.num_node_features
    model = OCGIN(dim_features, config["hidden_dim"], config["num_layers"], config["device"]).to(config["device"])

    # ------------------
    # TRAIN
    # ------------------
    logger.info("##################### staring training")
    train_ocgin(model, loader, config["epochs"], config["lr"])

    # ------------------
    # TEST / SCORE
    # ------------------
    logger.info("##################### computing scores")
    scores = compute_anomaly_scores(model, test_loader)

    logger.info("Anomaly scores:")
    logger.info(scores[:10])