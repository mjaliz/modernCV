import torch

x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
print(x*10)

y = x.add(10)
print(y)

y = torch.tensor([2, 3, 1, 0])
print(y)
print(y.shape)

y = y.view(4, 1)
print(y)
print(y.shape)

x = torch.randn(10, 1, 10)
z1 = torch.squeeze(x, 1)
z2 = x.squeeze(1)
assert torch.all(z1 == z2)
print("Squeeze:\n", x.shape, z1.shape)
print(x)
print(z1)

x = torch.randn(10, 10)
print(x.shape)
z1 = x.unsqueeze(0)
print(z1.shape)
z2, z3, z4 = x[None], x[:, None], x[:, :, None]
print(z2.shape, z3.shape, z4.shape)

# Matrix multiplication
x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
print(torch.matmul(x, y))
print(x@y)

# Tensor concatenation
x = torch.randn(10, 10, 10)
z = torch.cat([x, x], axis=0)  # np.concatenate()
print("Cat axis 0:", x.shape, z.shape)

z = torch.cat([x, x], axis=1)  # np.concatenate()
print("Cat axis 1:", x.shape, z.shape)

# Extract max value
x = torch.arange(25).reshape(5, 5)
print("Max: ", x.shape, x.max())

print(x.max(dim=0))

# max across the columns
m, argm = x.max(dim=1)
print("Max in axis 1:\n", m, argm)

# Permute the dimension of tensor
x = torch.randn(10, 20, 30)
z = x.permute(2, 0, 1)  # np.permute()
print("Permute dimensions:", x.shape, z.shape)
