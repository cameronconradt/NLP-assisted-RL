import torch

test = torch.empty([1, 54, 40, 768])
test2 = torch.empty([1, 54, 40, 768])
output = torch.empty([])
torch.stack([test, test2], 1, out=output)
print(output)