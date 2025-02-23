import torch
import numpy as np
import os

# **Parameter Configuration**
IN_DEPTH = 3
H_IN, W_IN = 10, 12  # Input size including padding

# **Ensure 'data' directory exists**
os.makedirs("data", exist_ok=True)

# **1. Generate random input data**
torch.manual_seed(42)
inBuffer = torch.randn(IN_DEPTH, H_IN, W_IN, dtype=torch.float32)  # Input feature map
weights = torch.randn(IN_DEPTH, 3, 3, dtype=torch.float32)  # 3x3 Depthwise kernel
bias = torch.randn(IN_DEPTH, dtype=torch.float32)  # Bias

# **2. Compute Depthwise Convolution (stride=1, padding=0)**
conv = torch.nn.Conv2d(IN_DEPTH, IN_DEPTH, kernel_size=3, stride=2, padding=0, groups=IN_DEPTH, bias=True)
conv.weight.data = weights.unsqueeze(1)  # Depthwise conv weight format
conv.bias.data = bias
out_torch = conv(inBuffer.unsqueeze(0)).squeeze(0).detach().numpy()  # Remove batch dimension

# **3. Save data (.npy format)**
np.save("data/inBuffer_dwconv.npy", inBuffer.numpy())  # Input values
np.save("data/weights_dwconv.npy", weights.numpy())    # 3x3 Kernel
np.save("data/bias_dwconv.npy", bias.numpy())          # Bias
np.save("data/out_torch_dwconv.npy", out_torch)        # PyTorch results

# **4. Save data (.txt format)**
np.savetxt("data/inBuffer_dwconv.txt", inBuffer.numpy().flatten(), fmt="%.6f")
np.savetxt("data/weights_dwconv.txt", weights.numpy().flatten(), fmt="%.6f")
np.savetxt("data/bias_dwconv.txt", bias.numpy(), fmt="%.6f")
np.savetxt("data/out_torch_dwconv.txt", out_torch.flatten(), fmt="%.6f")

print("Depthwise Conv 3x3 input & output data saved to `data/` directory.")
