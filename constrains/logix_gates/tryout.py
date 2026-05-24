import torch
from torch_geometric.datasets import Planetoid
from torchlogix.layers import LogicDense, FixedBinarization, GroupSum

DATASET_REFERENCE = {
    "Cora": Planetoid,
    "CiteSeer": Planetoid,
}

NORMAL_LABEL = 0


def load_cora():
    dataset = DATASET_REFERENCE["Cora"](root='data/Cora', name='Cora')
    data = dataset[0]
    X = data.x.float()
    y = data.y
    print(f"Cora: {data.num_nodes} nodes, {data.num_features} features, {y.max()+1} classes")
    return X, y


def load_binary_features():
    """Load Cora and binarize (for logic gates)"""
    X, y = load_cora()
    X_bin = (X > 0.0).float()
    return X_bin, y


def select_features(X, y, n_features=64):
    """Select top features by variance - simple feature selection
    
    For torchlogix, we need: out_dim * lut_rank >= in_dim
    With lut_rank=4, we need out_dim >= in_dim/4
    
    For Cora: 1433/4 = 358 hidden - still large
    Let's reduce to 64 features -> need 16 hidden (16*4=64)
    """
    variances = X.var(dim=0)
    _, indices = torch.topk(variances, n_features)
    X_selected = X[:, indices]
    print(f"Selected {n_features} features by variance")
    return X_selected, indices


def simple_logic_model(input_dim, hidden_size=16, lut_rank=4, num_classes=7):
    """Small logic network for interpretable constraints
    
    torchlogix requirement: hidden_size * lut_rank >= input_dim
    For input_dim=64, lut_rank=4: hidden_size >= 16
    """
    model = torch.nn.Sequential(
        FixedBinarization(thresholds=[0.0]),
        LogicDense(input_dim, hidden_size, lut_rank=lut_rank, parametrization='warp'),
        LogicDense(hidden_size, num_classes, lut_rank=lut_rank, parametrization='warp'),
        GroupSum(k=num_classes, tau=4)
    )
    return model


def train_model(model, X, y, epochs=50, lr=0.01):
    """Train the logic model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            pred = output.argmax(dim=1)
            acc = (pred == y).float().mean()
            print(f"Epoch {epoch+1}: loss={loss.item():.4f}, acc={acc:.4f}")
    
    return model


def extract_gate_formula(layer):
    """Extract boolean formula from a LogicDense layer"""
    try:
        # Try to get gate types (works with raw parametrization)
        gate_types = layer.gate_types.cpu().numpy()
        
        gate_names = {
            0: "ZERO", 1: "ONE", 2: "INPUT", 3: "NOT_INPUT",
            4: "AND", 5: "NAND", 6: "OR", 7: "NOR",
            8: "XOR", 9: "XNOR", 10: "LESS", 11: "LEQ",
            12: "GREATER", 13: "GEQ", 14: "CONST", 15: "PARAM"
        }
        
        formulas = []
        for out_idx in range(min(gate_types.shape[0], 3)):
            row = []
            for in_idx in range(gate_types.shape[1]):
                gate_type = int(gate_types[out_idx, in_idx])
                row.append(gate_names.get(gate_type, f"UNK{gate_type}"))
            formulas.append(f"Output {out_idx}: {' AND '.join(row)}")
        
        return formulas
    except AttributeError:
        # For warp/light parametrization, show connections
        connections = layer.connections.indices.cpu().numpy()
        formulas = []
        for out_idx in range(min(layer.out_dim, 3)):
            connected = connections[out_idx]
            parts = [f"in_{idx}" for idx in connected if idx >= 0]
            formulas.append(f"Output {out_idx}: connected to [{', '.join(parts)}]")
        return formulas


def main():
    print("Loading Cora dataset...")
    X, y = load_binary_features()
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Class distribution: {torch.bincount(y).tolist()}")
    
    # Feature selection to make model interpretable
    n_features = 64  # Small enough for readable formulas
    X_selected, indices = select_features(X, y, n_features)
    
    input_dim = X_selected.shape[1]
    hidden_size = 16  # With lut_rank=4: 16*4=64 >= 64
    num_classes = int(y.max().item()) + 1
    
    print(f"\nBuilding model: input={input_dim}, hidden={hidden_size}, lut_rank=4, classes={num_classes}")
    model = simple_logic_model(input_dim, hidden_size, num_classes=num_classes)
    
    print("\nTraining...")
    model = train_model(model, X_selected, y, epochs=50)
    
    print("\nExtracting formulas...")
    for i, layer in enumerate(model.modules()):
        if isinstance(layer, LogicDense):
            print(f"\nLayer {i}: LogicDense({layer.in_dim}, {layer.out_dim})")
            formulas = extract_gate_formula(layer)
            for f in formulas:
                print(f"  {f}")


if __name__ == "__main__":
    main()