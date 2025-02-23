/* 
This C++ implementation of Batch Normalization explicitly calculate the mean, variance, and standard deviation for normalization using the formula from
https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
This involves square root and division operations, which are computationally expensive and inefficient for hardware implementation.
In contrast, many hardware-friendly implementations assume that the input is already normalized, meaning the mean and variance are preprocessed, 
and the network directly scales the input with learned gamma (scale) and beta (bias) parameters. 
This avoids costly operations like square root and division, making the design more efficient for hardware deployment.
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

// **Data Parameters**
#define OUT_C 3
#define H_OUT 10
#define W_OUT 12

typedef float fm_t;

// **BatchNorm Computation**
void batch_norm(
    fm_t outBuffer[OUT_C][H_OUT][W_OUT],
    fm_t inBuffer[OUT_C][H_OUT][W_OUT], 
    fm_t gamma[OUT_C], 
    fm_t beta[OUT_C],
    fm_t running_mean[OUT_C], 
    fm_t running_var[OUT_C], 
    fm_t epsilon = 1e-5
    ) {
    for (int oc = 0; oc < OUT_C; oc++) {
        fm_t mean = running_mean[oc];  
        fm_t var = running_var[oc];    
        fm_t inv_std = 1.0 / sqrt(var + epsilon);  // Compute inverse standard deviation (1 / sqrt(var + eps))

        for (int i = 0; i < H_OUT; i++) {
            for (int j = 0; j < W_OUT; j++) {
                fm_t norm_x = (inBuffer[oc][i][j] - mean) * inv_std;  // Normalize
                outBuffer[oc][i][j] = gamma[oc] * norm_x + beta[oc];  // Scale and shift
            }
        }
    }
}

// **Load Data from File**
void load_data(const std::string &filename, std::vector<fm_t> &data) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        exit(1);
    }
    fm_t value;
    while (file >> value) {
        data.push_back(value);
    }
}

int main() {
    // **Create Buffers**
    fm_t inBuffer[OUT_C][H_OUT][W_OUT];
    fm_t gamma[OUT_C], beta[OUT_C], running_mean[OUT_C], running_var[OUT_C];
    fm_t outBuffer[OUT_C][H_OUT][W_OUT];

    // **Load Data**
    std::vector<fm_t> in_data, gamma_data, beta_data, mean_data, var_data;
    load_data("data/inBuffer_bn2d.txt", in_data);
    load_data("data/gamma_bn2d.txt", gamma_data);
    load_data("data/beta_bn2d.txt", beta_data);
    load_data("data/running_mean_bn2d.txt", mean_data);
    load_data("data/running_var_bn2d.txt", var_data);

    // **Fill inBuffer**
    int index = 0;
    for (int oc = 0; oc < OUT_C; oc++)
        for (int i = 0; i < H_OUT; i++)
            for (int j = 0; j < W_OUT; j++)
                inBuffer[oc][i][j] = in_data[index++];

    // **Fill γ (scale), β (shift), running mean, and variance**
    for (int oc = 0; oc < OUT_C; oc++) {
        gamma[oc] = gamma_data[oc];
        beta[oc] = beta_data[oc];
        running_mean[oc] = mean_data[oc];
        running_var[oc] = var_data[oc];
    }

    // **Execute BatchNorm**
    batch_norm(inBuffer, outBuffer, gamma, beta, running_mean, running_var);

    // **Write Computation Results to File**
    std::ofstream file("data/out_cpp_bn2d.txt");
    if (!file) {
        std::cerr << "Unable to write file: data/out_cpp_bn2d.txt" << std::endl;
        return 1;
    }
    for (int oc = 0; oc < OUT_C; oc++)
        for (int i = 0; i < H_OUT; i++)
            for (int j = 0; j < W_OUT; j++)
                file << outBuffer[oc][i][j] << "\n";

    std::cout << "C++ computation complete, results saved to `data/out_cpp_bn2d.txt`" << std::endl;
    return 0;
}
