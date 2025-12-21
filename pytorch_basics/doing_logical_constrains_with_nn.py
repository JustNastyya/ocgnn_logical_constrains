import torch
import torch.nn as nn
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
    
    def forward(self, x):
        return self.net(x)


def implecation_lodd(logits):
    probs = torch.sigmoid(logits)
    A = probs[:, 0]
    B = probs[:, 1]
    return torch.mean(A * (1 - B))

# toy dataset
x = torch.tensor([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
])

# labels: [A, B]
y = torch.tensor([
    [0.0, 0.0],
    [1.0, 0.0],  # violates A ⇒ B
    [0.0, 1.0],
    [1.0, 1.0],
])

# binary cross entropy
bce = nn.BCEWithLogitsLoss()

# logic is: A->B or not(A) + B

#### train

model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.05)

lambda_logic = 1.0
for epoch in range(500):
    optimizer.zero_grad()

    logits = model(x)
    data_loss = bce(logits, y)
    logic_loss = implecation_lodd(logits)
    loss = data_loss + lambda_logic * logic_loss

    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch:03d} | "
            f"Data: {data_loss.item():.4f} | "
            f"Logic: {logic_loss.item():.4f}"
        )

