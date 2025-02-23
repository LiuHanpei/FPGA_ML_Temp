import numpy as np

# **File paths**
torch_output_file = "data/out_torch_dwconv.txt"  # PyTorch output
cpp_output_file = "data/out_cpp_dwconv.txt"      # C++ output

# **1. Load PyTorch computation results**
out_torch = np.loadtxt(torch_output_file, dtype=np.float32)

# **2. Load C++ computation results**
out_cpp = np.loadtxt(cpp_output_file, dtype=np.float32)

# **3. Check if shapes match**
if out_torch.shape != out_cpp.shape:
    print(f"hape mismatch: PyTorch {out_torch.shape} vs C++ {out_cpp.shape}")
    exit(1)

# **4. Compute Mean Squared Error (MSE)**
mse = np.mean((out_torch - out_cpp) ** 2)

# **5. Print results**
print(f"MSE Error: {mse:.10f}")

# **6. Check error tolerance**
tolerance = 1e-4
if mse < tolerance:
    print("C++ and PyTorch computation results match!")
else:
    print("Error too large, please check C++ computation!")
