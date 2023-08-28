import torch
import numpy as np
x = torch.tensor([[1, 2]])
y = torch.tensor([[1], [2]])

print(x.shape)
print(y.shape)
print(x.dtype)

print(torch.zeros((3, 4)))
print(torch.ones((3, 4)))
print(torch.randint(low=0, high=10, size=(3, 4)))
print(torch.rand((3, 4)))
print(torch.randn((3, 4)))

x = np.array([[10, 20, 30], [2, 3, 4]])
y = torch.tensor(x)
print(type(x), type(y))
