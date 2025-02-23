import torch
import numpy as np
import os

# **Parameter Configuration**
OUT_C = 3      # Number of output channels
H_OUT = 10     # Output height
W_OUT = 12     # Output width

# **Ensure 'data' directory exists**
os.makedirs("data", exist_ok=True)

# **1. Generate random input data (including negative values)**
torch.manual_seed(42)
inBuffer = torch.randn(OUT_C, H_OUT, W_OUT, dtype=torch.float32)  # Allow negative values

# **2. Apply ReLU activation**
relu_layer = torch.nn.ReLU()
out_torch = relu_layer(inBuffer).numpy()

# **3. Save data in both .txt and .npy formats**
np.save("data/inBuffer_relu.npy", inBuffer.numpy())   # Save input values
np.save("data/out_torch_relu.npy", out_torch)         # Save ReLU output
np.savetxt("data/inBuffer_relu.txt", inBuffer.numpy().flatten(), fmt="%.6f")
np.savetxt("data/out_torch_relu.txt", out_torch.flatten(), fmt="%.6f")

print("ReLU input & output data saved in the `data/` directory (both `.npy` and `.txt` formats).")
