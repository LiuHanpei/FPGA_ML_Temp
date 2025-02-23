# FPGA_ML_Temp

## Description
This repository contains Python and C++ implementations of fundamental deep learning operations, including:  
- Batch Normalization  
- Activation Functions (ReLU, Tanh)  
- Fully Connected (Linear) Layers  
- Standard & Depthwise Convolutions  
Each operation is implemented in Python (for generating test data and reference results) and in C++ (for efficient computation, designed for FPGA deployment with HLS optimizations).  

## Environment Setup
- Python: 3.12.8  
- C++ Compiler: Apple clang version 16.0.0 (clang-1600.0.26.6)  
- Xilinx Toolchain: Vivado 2022.1, Vitis HLS 2022.1  

## FPGA Implementations
- Implement the same functions as Python, but in a hardware-friendly manner.  
- The first function in each C++ file is the core operation (e.g., batch_norm(), relu(), conv2d()).  
- The remaining code is for loading input data, running the function, and saving the output for comparison.  
- No HLS-specific pragmas are included, they can be optimize based on available FPGA resources.  

## AI-Generated Code Disclosure
Some parts of this code were generated using ChatGPT.
