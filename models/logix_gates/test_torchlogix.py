import torch
from torchlogix.layers import LogicDense, FixedBinarization, GroupSum


def simple_test():
    X = torch.randint(0, 2, (100, 10)).float()
    y = torch.randint(0, 2, (100,))
    
    print(f"Data: X={X.shape}, y={y.shape}")
    
    # Model needs: out_dim * lut_rank >= in_dim
    # For in_dim=10, lut_rank=2 -> need out_dim >= 5
    # With lut_rank=2: out_dim=8 means 8*2=16 >= 10
    
    model = torch.nn.Sequential(
        FixedBinarization(thresholds=[0.0]),
        LogicDense(10, 8, lut_rank=2),   # 10 inputs -> 8 hidden, lut_rank=2
        LogicDense(8, 2, lut_rank=2),    # 8 hidden -> 2 classes
        GroupSum(k=2, tau=4)
    )
    
    print(f"Model: {model}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(20):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            pred = output.argmax(dim=1)
            acc = (pred == y).float().mean()
            print(f"Epoch {epoch+1}: loss={loss.item():.4f}, acc={acc:.4f}")
    
    # Extract gate info
    print("\nGate types learned:")
    for layer in model.modules():
        if isinstance(layer, LogicDense):
            gate_types = layer.gate_types.cpu().numpy()
            print(f"  Layer: gate_types shape = {gate_types.shape}")
            print(f"  First 3 outputs: {gate_types[:3, :4]}")  # First 3 outputs, first 4 gates

if __name__ == "__main__":
    simple_test()