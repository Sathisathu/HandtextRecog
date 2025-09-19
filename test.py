import torch
print("CUDA available:", torch.cuda.is_available())      # Should print True
print("GPU name:", torch.cuda.get_device_name(0))        # Should print NVIDIA GeForce RTX 3050
