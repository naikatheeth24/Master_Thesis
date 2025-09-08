import torch
print(torch.__version__)       # Should show torch 2.7.1+cu126 (or similar)
print(torch.cuda.is_available())  # Should output True
print(torch.cuda.get_device_name(0))  # Should output your GPU name (e.g., NVIDIA GeForce RTX 4090)
