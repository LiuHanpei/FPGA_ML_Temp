import torch
import numpy as np
import os

# **Parameter Configuration**
BATCH_SIZE = 1      # Batch size (1 for simplicity)
IN_FEATURES = 64    # Number of input features
OUT_FEATURES = 32   # Number of output features

# **Ensure 'data' directory exists**
os.makedirs("data", exist_ok=True)

# **1. Generate input data, weights, and bias**
torch.manual_seed(42)
X = torch.rand(BATCH_SIZE, IN_FEATURES, dtype=torch.float32) * 2 - 1  # Input in range [-1, 1]
W = torch.rand(OUT_FEATURES, IN_FEATURES, dtype=torch.float32) * 2 - 1  # Weights in range [-1, 1]
B = torch.rand(OUT_FEATURES, dtype=torch.float32) * 2 - 1  # Bias in range [-1, 1]

# **2. Compute Linear transformation**
linear_layer = torch.nn.Linear(IN_FEATURES, OUT_FEATURES, bias=True)
linear_layer.weight.data = W
linear_layer.bias.data = B

Y = linear_layer(X).detach().numpy()  # Compute output

# **3. Save input, weights, bias, and output**
np.save("data/X_linear.npy", X.numpy())   # Input data
np.save("data/W_linear.npy", W.numpy())   # Weights
np.save("data/B_linear.npy", B.numpy())   # Bias
np.save("data/Y_torch_linear.npy", Y)     # PyTorch computed output

np.savetxt("data/X_linear.txt", X.numpy().flatten(), fmt="%.6f")
np.savetxt("data/W_linear.txt", W.numpy().flatten(), fmt="%.6f")
np.savetxt("data/B_linear.txt", B.numpy(), fmt="%.6f")
np.savetxt("data/Y_torch_linear.txt", Y.flatten(), fmt="%.6f")

print("✅ Linear layer data saved in `data/` directory.")
