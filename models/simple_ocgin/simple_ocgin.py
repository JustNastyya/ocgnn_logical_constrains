from GraphNets import GIN
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from loguru import logger


class OCGIN(nn.Module):
    def __init__(self, dim_features, config):
        super(OCGIN, self).__init__()

        self.dim_targets = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.device = config['device']
        self.net = GIN(dim_features, self.dim_targets, config)
        self.center = torch.zeros(1, self.dim_targets * self.num_layers, requires_grad=False).to('cpu')
        self.reset_parameters()
    def forward(self, data):
        data = data.to(self.device)
        z = self.net(data)
        return [z, self.center]

    def init_center(self, train_loader):
        with torch.no_grad():
            for data in train_loader:
                data = data.to('cpu')
                z = self.forward(data)
                self.center += torch.sum(z[0], 0, keepdim=True)
            self.center = self.center / len(train_loader.dataset)

    def reset_parameters(self):
        self.net.reset_parameters()


def ocgin_loss(z, center):
    return torch.mean(torch.sum((z - center) ** 2, dim=1))


def train_ocgin(model, train_loader, config):
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config.get('weight_decay', 0.0)
    )

    # ---- Phase 1: Initialize center ----
    logger.info("Initializing center...")
    model.eval()
    model.init_center(train_loader)

    # ---- Phase 2: Train GIN ----
    logger.info("Training OCGIN...")
    model.train()

    for epoch in range(config['epochs']):
        total_loss = 0.0

        for data in train_loader:
            z, center = model(data)
            loss = ocgin_loss(z, center)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch:03d} | Loss: {avg_loss:.6f}")


def compute_anomaly_scores(model, loader):
    model.eval()
    scores = []

    with torch.no_grad():
        for data in loader:
            z, center = model(data)
            score = torch.sum((z - center) ** 2, dim=1)
            scores.append(score.cpu())

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
    }

    # ------------------
    # LOAD DATA
    # ------------------
    # IMPORTANT:
    # train_dataset must contain ONLY NORMAL GRAPHS
    logger.info("##################### Loading data")
    
    dataset = TUDataset(root="data/TUDataset", name='REDDIT-BINARY')    
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
    dim_features = 0# train_dataset.num_node_features
    model = OCGIN(dim_features, config).to(config["device"])

    # ------------------
    # TRAIN
    # ------------------
    logger.info("##################### staring training")
    train_ocgin(model, train_loader, config)

    # ------------------
    # TEST / SCORE
    # ------------------
    logger.info("##################### computing scores")
    scores = compute_anomaly_scores(model, test_loader)

    logger.info("Anomaly scores:")
    logger.info(scores[:10])