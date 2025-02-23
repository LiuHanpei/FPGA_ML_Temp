#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

// **Depthwise Convolution Configuration**
#define IN_C 3       // Input channels (same as output channels)
#define H_IN 10      // Input height (including padding)
#define W_IN 12      // Input width (including padding)
#define K 3          // Kernel size
#define STRIDE 2     // Stride
#define H_OUT ((H_IN - K) / STRIDE + 1)  // Output height
#define W_OUT ((W_IN - K) / STRIDE + 1)  // Output width

typedef float fm_t;

// **Depthwise Convolution Function**
void dw_conv3x3(
    fm_t outBuffer[IN_C][H_OUT][W_OUT],
    fm_t inBuffer[IN_C][H_IN][W_IN], 
    fm_t weights[IN_C][K][K], 
    fm_t bias[IN_C]
) {
    for (int c = 0; c < IN_C; c++) {  
        for (int i = 0; i < H_OUT; i++) {  
            for (int j = 0; j < W_OUT; j++) {  
                fm_t sum = bias[c];  // Start with bias
                for (int ki = 0; ki < K; ki++) {  
                    for (int kj = 0; kj < K; kj++) {  
                        int in_i = i * STRIDE + ki;  
                        int in_j = j * STRIDE + kj;  
                        sum += inBuffer[c][in_i][in_j] * weights[c][ki][kj];  
                    }
                }
                outBuffer[c][i][j] = sum;  // Store result in output buffer
            }
        }
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
    // **Create Buffers**
    fm_t inBuffer[IN_C][H_IN][W_IN];
    fm_t weights[IN_C][K][K];
    fm_t bias[IN_C];
    fm_t outBuffer[IN_C][H_OUT][W_OUT];

    // **Load Data**
    std::vector<fm_t> in_data, weight_data, bias_data;
    load_data("data/inBuffer_dwconv.txt", in_data);
    load_data("data/weights_dwconv.txt", weight_data);
    load_data("data/bias_dwconv.txt", bias_data);

    // **Fill inBuffer**
    int index = 0;
    for (int c = 0; c < IN_C; c++)
        for (int i = 0; i < H_IN; i++)
            for (int j = 0; j < W_IN; j++)
                inBuffer[c][i][j] = in_data[index++];

    // **Fill weights**
    index = 0;
    for (int c = 0; c < IN_C; c++)
        for (int ki = 0; ki < K; ki++)
            for (int kj = 0; kj < K; kj++)
                weights[c][ki][kj] = weight_data[index++];

    // **Fill bias**
    for (int c = 0; c < IN_C; c++)
        bias[c] = bias_data[c];

    // **Perform Depthwise Convolution**
    dw_conv3x3(outBuffer, inBuffer, weights, bias);

    // **Write output to file**
    std::ofstream file("data/out_cpp_dwconv.txt");
    if (!file) {
        std::cerr << "Cannot write to file: data/out_cpp_dwconv.txt" << std::endl;
        return 1;
    }
    for (int c = 0; c < IN_C; c++)
        for (int i = 0; i < H_OUT; i++)
            for (int j = 0; j < W_OUT; j++)
                file << outBuffer[c][i][j] << "\n";

    std::cout << "Depthwise Conv 3x3 computation completed, results saved to `data/out_cpp_dwconv.txt`" << std::endl;
    return 0;
}
