import torch
from torch_geometric.datasets import Planetoid
from torchlogix.layers import LogicDense, FixedBinarization, GroupSum

DATASET_REFERENCE = {
    "Cora": Planetoid,
    "CiteSeer": Planetoid,
    "PubMed": Planetoid,
}

NORMAL_LABEL = 0


def load_cora():
    """Load Cora dataset - node classification (multi-class)"""
    dataset = DATASET_REFERENCE["Cora"](root='data/Cora', name='Cora')
    data = dataset[0]
    X = data.x.float()
    y = data.y
    print(f"Cora: {data.num_nodes} nodes, {data.num_features} features, {y.max()+1} classes")
    return X, y


def binarize_features(X, threshold=0.0):
    """Convert continuous features to binary"""
    return (X > threshold).float()


def load_binary_features():
    """Load Cora and binarize (for logic gates)"""
    X, y = load_cora()
    X_bin = binarize_features(X)
    return X_bin, y


def simple_logic_model(input_dim, hidden_gates=4, num_classes=7):
    """Small logic network for interpretable constraints
    
    Architecture:
    - Input: binary features
    - Hidden: few logic gates (readable)
    - Output: class logits
    
    With hidden_gates=4, the formula will have at most 4 gates per layer,
    making it reasonably interpretable.
    """
    model = torch.nn.Sequential(
        FixedBinarization(thresholds=[0.0]),
        LogicDense(input_dim, hidden_gates, num_gates=hidden_gates),
        LogicDense(hidden_gates, num_classes),
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


def extract_gate_formula(layer, feature_names=None):
    """Extract boolean formula from a LogicDense layer
    
    Args:
        layer: LogicDense layer
        feature_names: optional list of feature names for readable output
    
    Returns:
        List of formulas, one per output class
    """
    gate_types = layer.gate_types.cpu().numpy()  # (output_dims, num_gates, 2)
    gate_params = layer.gate_params.cpu().numpy()  # (output_dims, num_gates, num_params)
    
    formulas = []
    for out_idx in range(gate_types.shape[0]):
        formulas.append(f"Output {out_idx}: ")
        for gate_idx in range(gate_types.shape[1]):
            gate_type = gate_types[out_idx, gate_idx]
            # Map gate type index to name
            gate_name = ["ZERO", "ONE", "INPUT", "NOT_INPUT", "AND", "NAND", "OR", "NOR",
                         "XOR", "XNOR", "LESS", "LEQ", "GREATER", "GEQ", "CONST", "PARAM"][gate_type]
            formulas.append(f"  Gate {gate_idx}: {gate_name}")
    
    return formulas


def main():
    print("Loading Cora dataset...")
    X, y = load_binary_features()
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Class distribution: {torch.bincount(y)}")
    
    input_dim = X.shape[1]
    hidden_gates = 4  # Keep small for interpretability
    num_classes = int(y.max().item()) + 1
    
    print(f"\nBuilding model: input={input_dim}, hidden_gates={hidden_gates}, classes={num_classes}")
    model = simple_logic_model(input_dim, hidden_gates, num_classes)
    
    print("\nTraining...")
    model = train_model(model, X, y, epochs=50)
    
    print("\nExtracting formulas...")
    for i, layer in enumerate(model.modules()):
        if isinstance(layer, LogicDense):
            print(f"\nLayer {i}: {layer}")
            formulas = extract_gate_formula(layer)
            for f in formulas[:10]:  # First 10 lines
                print(f"  {f}")


if __name__ == "__main__":
    main()