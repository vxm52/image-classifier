import torch

data = [[1, 2, 3, 4],[3, 4, 5, 6]]

x_data = torch.tensor(data)  # Convert matrix to tensor

x_rand = torch.rand_like(x_data, dtype=torch.float)

# Linear algebra
tensor1 = torch.randn(3, 4)
tensor2 = torch.randn(4)

result = torch.matmul(tensor1, tensor2)  # Multiplies the tensors together

