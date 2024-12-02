import torch

if torch.cuda.is_available():
    print(f"CUDA is available. GPU name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available.")
