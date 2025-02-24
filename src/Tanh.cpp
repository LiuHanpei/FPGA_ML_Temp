// Experimental results indicate that different LUT (Lookup Table) lengths 
// have varying levels of precision compared to Python calculations (torch.tanh).
// As the LUT length increases, the MSE error gradually decreases, 
// demonstrating improved accuracy. The specific error values are as follows:
// - LUT length 256:  MSE Error = 0.0006584101
// - LUT length 512:  MSE Error = 0.0002050189
// - LUT length 1024: MSE Error = 0.0000431783
// - LUT length 2048: MSE Error = 0.0000054822
// @note In actual FPGA deployment, users should precompute the lookup table (LUT) and load it onto the FPGA。

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

#define N_INPUTS 128  // Number of inputs
#define TABLE_SIZE 256  // LUT table size
#define TANH_RANGE 32   // Input range [-32, 31]
#define STEP (2.0 * TANH_RANGE / TABLE_SIZE)  // LUT step size

typedef float fm_t;

// Lookup table for tanh function
fm_t tanh_table[TABLE_SIZE];

// **Initialize the tanh lookup table**
void init_tanh_table() {
    for (int i = 0; i < TABLE_SIZE; i++) {
        float x = -TANH_RANGE + i * STEP;  // Map index to input range
        tanh_table[i] = std::tanh(x);  // Compute tanh value
    }
}

// **Tanh computation using LUT**
void tanh_lut(fm_t inBuffer[], fm_t outBuffer[], int size) {
    for (int i = 0; i < size; i++) {
        // Clamp input to valid range
        fm_t x = (inBuffer[i] < -TANH_RANGE) ? -TANH_RANGE : 
                 (inBuffer[i] > TANH_RANGE) ? TANH_RANGE : inBuffer[i];
        // Compute index in the LUT
        int index = (x + TANH_RANGE) * (TABLE_SIZE / (2 * TANH_RANGE));
        // Ensure index is within bounds
        if (index < 0) index = 0;
        if (index >= TABLE_SIZE) index = TABLE_SIZE - 1;
        // Fetch the precomputed tanh value
        outBuffer[i] = tanh_table[index];
    }
}

// **Load data from file**
void load_data(const std::string &filename, std::vector<fm_t> &data) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        exit(1);
    }
    fm_t value;
    while (file >> value) {
        data.push_back(value);
    }
}

int main() {
    // **Initialize tanh lookup table**
    init_tanh_table();
    // **Load input data from Python-generated file**
    std::vector<fm_t> in_data;
    load_data("data/inBuffer_Tanh.txt", in_data);
    // **Prepare input and output buffers**
    fm_t inBuffer[N_INPUTS];
    fm_t outBuffer[N_INPUTS];
    // **Copy data into input buffer**
    for (int i = 0; i < N_INPUTS; i++) {
        inBuffer[i] = in_data[i];
    }
    // **Compute tanh using LUT**
    tanh_lut(inBuffer, outBuffer, N_INPUTS);
    // **Save output results**
    std::ofstream file("data/out_cpp_Tanh.txt");
    if (!file) {
        std::cerr << "Cannot write to file: data/out_cpp_Tanh.txt" << std::endl;
        return 1;
    }
    for (int i = 0; i < N_INPUTS; i++) {
        file << outBuffer[i] << "\n";
    }
    std::cout << "C++ tanh computation completed, results saved in `data/out_cpp_Tanh.txt`" << std::endl;
    return 0;
}
