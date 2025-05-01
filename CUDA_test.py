import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda if torch.cuda.is_available() else "Not available")

if torch.cuda.is_available():
    print("GPU device name:", torch.cuda.get_device_name(0))
    print("Number of GPUs:", torch.cuda.device_count())
else:
    print("No GPU available. Using CPU only.")

# Test creating a tensor on GPU
try:
    x = torch.tensor([1.0, 2.0, 3.0]).cuda()
    print("\nSuccessfully created a tensor on GPU:", x)
    print("Tensor device:", x.device)
except Exception as e:
    print("\nError creating tensor on GPU:", str(e))