import torch
import sys

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")

# בדיקה האם יש כרטיס מסך זמין (במקרה שלך נצפה ל-False, וזה בסדר)
cuda_available = torch.cuda.is_available()
print(f"CUDA (GPU) Available: {cuda_available}")

if not cuda_available:
    print("Running on CPU (as expected for your setup).")
else:
    print("Running on GPU!")

# בדיקת חישוב מהירה
x = torch.rand(5, 3)
print("\nTest Tensor:\n", x)
print("\nInstallation Successful! ✅")