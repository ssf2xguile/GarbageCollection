import torch
import torch.nn.functional as F

torch.manual_seed(1)

x = torch.tensor([1, 2, 3], dtype=torch.float32)

fc1 = torch.nn.Linear(3, 2)

fc2 = torch.nn.Linear(2, 1)

y = fc2(F.relu(fc1(x)))

t = torch.tensor([1], dtype=torch.float32)
loss = F.mse_loss(y, t)
print(loss)