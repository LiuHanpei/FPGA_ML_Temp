import torch
import numpy as np
import os

# **Configuration Parameters**
OUT_C = 3     
H_OUT = 10    
W_OUT = 12    

# **Ensure the `data` directory exists**
os.makedirs("data", exist_ok=True)

# **1. Generate Random Data**
torch.manual_seed(42)
inBuffer = torch.rand(1, OUT_C, H_OUT, W_OUT, dtype=torch.float32)  # Random input
gamma = torch.rand(OUT_C, dtype=torch.float32)   # Scale parameter (γ)
beta = torch.rand(OUT_C, dtype=torch.float32)    # Bias parameter (β)
running_mean = torch.rand(OUT_C, dtype=torch.float32)  # Running mean (used in inference)
running_var = torch.rand(OUT_C, dtype=torch.float32)   # Running variance (used in inference)

# **2. Create PyTorch BatchNorm2d Layer**
batchnorm = torch.nn.BatchNorm2d(OUT_C, affine=True, track_running_stats=True)
batchnorm.weight.data = gamma
batchnorm.bias.data = beta
batchnorm.running_mean = running_mean
batchnorm.running_var = running_var

# **3. Compute BatchNorm**
batchnorm.eval()  # Set to inference mode
out_torch = batchnorm(inBuffer).squeeze(0).detach().numpy()  # Remove batch dimension

# **4. Save Data (in `.npy` format)**
np.save("data/inBuffer_bn2d.npy", inBuffer.numpy())   # Input
np.save("data/gamma_bn2d.npy", gamma.numpy())         # γ (Scale)
np.save("data/beta_bn2d.npy", beta.numpy())           # β (Bias)
np.save("data/running_mean_bn2d.npy", running_mean.numpy())  # Running mean
np.save("data/running_var_bn2d.npy", running_var.numpy())   # Running variance
np.save("data/out_torch_bn2d.npy", out_torch)         # PyTorch computation result

# **5. Save Data (in `.txt` format)**
np.savetxt("data/inBuffer_bn2d.txt", inBuffer.numpy().flatten(), fmt="%.6f")
np.savetxt("data/gamma_bn2d.txt", gamma.numpy(), fmt="%.6f")
np.savetxt("data/beta_bn2d.txt", beta.numpy(), fmt="%.6f")
np.savetxt("data/running_mean_bn2d.txt", running_mean.numpy(), fmt="%.6f")
np.savetxt("data/running_var_bn2d.txt", running_var.numpy(), fmt="%.6f")
np.savetxt("data/out_torch_bn2d.txt", out_torch.flatten(), fmt="%.6f")

print("BatchNorm2d data has been saved to the `data/` directory, including `.npy` and `.txt` formats.")
