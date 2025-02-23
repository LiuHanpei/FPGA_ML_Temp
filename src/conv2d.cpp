#include <iostream>
#include <fstream>
#include <vector>

#define IN_C 3
#define OUT_C 16
#define H_IN 10
#define W_IN 12
#define K 3
#define STRIDE 2
#define H_OUT ((H_IN - K) / STRIDE + 1)
#define W_OUT ((W_IN - K) / STRIDE + 1)

typedef float fm_t; 

// **2D Convolution Function**
void conv2d (
    fm_t outBuffer[OUT_C][H_OUT][W_OUT],
    fm_t inBuffer[IN_C][H_IN][W_IN],
    fm_t weights[OUT_C][IN_C][K][K],
    fm_t bias[OUT_C]
){
    for (int oc = 0; oc < OUT_C; oc++) {  
        for (int i = 0; i < H_OUT; i++) {  
            for (int j = 0; j < W_OUT; j++) {  
                fm_t sum = bias[oc];  
                for (int ic = 0; ic < IN_C; ic++) {  
                    for (int ki = 0; ki < K; ki++) {  
                        for (int kj = 0; kj < K; kj++) {  
                            int in_i = i * STRIDE + ki;  
                            int in_j = j * STRIDE + kj;  
                            sum += inBuffer[ic][in_i][in_j] * weights[oc][ic][ki][kj];  
                        }
                    }
                }
                outBuffer[oc][i][j] = sum;  // Store result in output buffer
            }
        }
    }
}

// **Function to Load Data from a File**
void load_data(const std::string &filename, std::vector<fm_t> &data) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error: Cannot open file: " << filename << std::endl;
        exit(1);
    }
    fm_t value;
    while (file >> value) {
        data.push_back(value);  // Read data line by line into vector
    }
}

int main() {
    // **Declare Buffers**
    fm_t inBuffer[IN_C][H_IN][W_IN];  // Input feature map
    fm_t weights[OUT_C][IN_C][K][K];  // Convolution kernel weights
    fm_t bias[OUT_C];  // Bias terms
    fm_t outBuffer[OUT_C][H_OUT][W_OUT];  // Output feature map

    // **Load Data from Files**
    std::vector<fm_t> in_data, weight_data, bias_data;
    load_data("data/inBuffer_conv2d.txt", in_data);
    load_data("data/weights_conv2d.txt", weight_data);
    load_data("data/bias_conv2d.txt", bias_data);

    // **Fill inBuffer with Loaded Data**
    int index = 0;
    for (int ic = 0; ic < IN_C; ic++)
        for (int i = 0; i < H_IN; i++)
            for (int j = 0; j < W_IN; j++)
                inBuffer[ic][i][j] = in_data[index++];

    // **Fill weights with Loaded Data**
    index = 0;
    for (int oc = 0; oc < OUT_C; oc++)
        for (int ic = 0; ic < IN_C; ic++)
            for (int ki = 0; ki < K; ki++)
                for (int kj = 0; kj < K; kj++)
                    weights[oc][ic][ki][kj] = weight_data[index++];

    // **Fill bias with Loaded Data**
    for (int oc = 0; oc < OUT_C; oc++)
        bias[oc] = bias_data[oc];

    // **Perform 2D Convolution**
    conv2d(outBuffer, inBuffer, weights, bias);

    // **Write Computation Results to File**
    std::ofstream file("data/out_cpp_conv2d.txt");
    if (!file) {
        std::cerr << "Error: Cannot write to file: data/out_cpp_conv2d.txt" << std::endl;
        return 1;
    }
    for (int oc = 0; oc < OUT_C; oc++)
        for (int i = 0; i < H_OUT; i++)
            for (int j = 0; j < W_OUT; j++)
                file << outBuffer[oc][i][j] << "\n";

    std::cout << "C++ computation completed, results saved to `data/out_cpp_conv2d.txt`" << std::endl;
    return 0;
}
