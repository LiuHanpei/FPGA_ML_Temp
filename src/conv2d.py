import torch
import numpy as np
import os

# **Parameter Configuration**
IN_C = 3      
OUT_C = 16     
H_IN = 10   
W_IN = 12     
K = 3         
STRIDE = 2

H_OUT = (H_IN - K) // STRIDE + 1
W_OUT = (W_IN - K) // STRIDE + 1

# **Ensure the `data/` directory exists**
os.makedirs("data", exist_ok=True)

# **1. Generate random data**
torch.manual_seed(42)
inBuffer = torch.rand(1, IN_C, H_IN, W_IN, dtype=torch.float32)  
weights = torch.rand(OUT_C, IN_C, K, K, dtype=torch.float32)
bias = torch.rand(OUT_C, dtype=torch.float32)

# **2. Compute Conv2D**
conv_layer = torch.nn.Conv2d(IN_C, OUT_C, kernel_size=K, stride=STRIDE, padding=0, bias=True)
conv_layer.weight.data = weights
conv_layer.bias.data = bias
out_torch = conv_layer(inBuffer).squeeze(0).detach().numpy()  # Remove batch dimension

# **3. Save data in `.npy` format**
np.save("data/inBuffer_conv2d.npy", inBuffer.numpy())   # Input
np.save("data/weights_conv2d.npy", weights.numpy())     # Weights
np.save("data/bias_conv2d.npy", bias.numpy())           # Bias
np.save("data/out_torch_conv2d.npy", out_torch)         # PyTorch computed result

# **4. Save data in `.txt` format**
np.savetxt("data/inBuffer_conv2d.txt", inBuffer.numpy().flatten(), fmt="%.6f")
np.savetxt("data/weights_conv2d.txt", weights.numpy().flatten(), fmt="%.6f")
np.savetxt("data/bias_conv2d.txt", bias.numpy(), fmt="%.6f")
np.savetxt("data/out_torch_conv2d.txt", out_torch.flatten(), fmt="%.6f")

print("Data has been saved to the `data/` directory, including `.npy` and `.txt` formats.")
