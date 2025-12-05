import torch

x = torch.tensor([1.0, 2.0, 3.0])
y = torch.rand(4, 3) # matrix 3 x 4

# i can move stuff to gpu but i have no idea how yet.

z = x + y

x * 2

torch.matmul(x, x.T) # matrix multiplication
