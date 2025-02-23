import torch
import numpy as np
import os

# **Parameter Configuration**
N_INPUTS = 128  # Number of input values for tanh

# **Ensure 'data' directory exists**
os.makedirs("data", exist_ok=True)

# **1. Generate input data within [-32, 31]**
torch.manual_seed(42)
inBuffer = (torch.rand(N_INPUTS, dtype=torch.float32) * 31) - 16  # Scale to [-16, 15]

# **2. Compute tanh using PyTorch**
out_torch = torch.tanh(inBuffer).numpy()  # Apply tanh function

# **3. Save data (.npy format)**
np.save("data/inBuffer_Tanh.npy", inBuffer.numpy())  # Input values
np.save("data/out_torch_Tanh.npy", out_torch)       # Tanh results

# **4. Save data (.txt format)**
np.savetxt("data/inBuffer_Tanh.txt", inBuffer.numpy(), fmt="%.6f")
np.savetxt("data/out_torch_Tanh.txt", out_torch, fmt="%.6f")

print("Tanh input & output data saved to `data/` directory.")
