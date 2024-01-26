import torch
print(torch.cuda.device_count())
print(torch.cuda.is_available())
print(torch.cuda.current_device())
torch.zeros(1).cuda()
