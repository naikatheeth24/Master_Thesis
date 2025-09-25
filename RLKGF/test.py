import torch

num_gpus = torch.cuda.device_count()
print("Available GPUs:", num_gpus)

device_id = 0  # default to GPU 0

if device_id >= num_gpus:
    print(f"Warning: device_id {device_id} is invalid, falling back to cpu")
    device = torch.device('cpu')
else:
    device = torch.device(f'cuda:{device_id}')

print("Using device:", device)
