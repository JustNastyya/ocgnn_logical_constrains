import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output):
        super().__init__()
        self.first_layer = nn.Linear(dim_input, dim_hidden)
        self.second_layer = nn.Linear(dim_hidden, dim_output)
    
    def forward(self, x):
        x = F.relu(self.first_layer(x))
        x = self.second_layer(x)
        return x



def train(model, X, y):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()
        print(f"epoch {epoch} and loss: {loss.item()}")


model = MyModel(10, 32, 1)
X = torch.randn(100, 10)
y = torch.randn(100, 1)
train(model, X, y)
