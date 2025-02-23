#include <iostream>
#include <fstream>
#include <vector>

#define OUT_C 3
#define H_OUT 10
#define W_OUT 12
typedef float fm_t; 

// ReLU activation function
void relu(fm_t inBuffer[OUT_C][H_OUT][W_OUT]) {
    for (int oc = 0; oc < OUT_C; oc++) {
        for (int i = 0; i < H_OUT; i++) {
            for (int j = 0; j < W_OUT; j++) {
                if (inBuffer[oc][i][j] < 0) {
                    inBuffer[oc][i][j] = 0;  
                }
            }
        }
    }
}

// **Load data from file**
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

    // **Load input data**
    std::vector<fm_t> in_data;
    load_data("data/inBuffer_relu.txt", in_data);

    // **Fill inBuffer**
    int index = 0;
    for (int oc = 0; oc < OUT_C; oc++)
        for (int i = 0; i < H_OUT; i++)
            for (int j = 0; j < W_OUT; j++)
                inBuffer[oc][i][j] = in_data[index++];

    // **Apply ReLU activation**
    relu(inBuffer);

    // **Save computed results**
    std::ofstream file("data/out_cpp_relu.txt");
    if (!file) {
        std::cerr << "Unable to write to file: data/out_cpp_relu.txt" << std::endl;
        return 1;
    }
    for (int oc = 0; oc < OUT_C; oc++)
        for (int i = 0; i < H_OUT; i++)
            for (int j = 0; j < W_OUT; j++)
                file << inBuffer[oc][i][j] << "\n";

    std::cout << "C++ ReLU computation complete. Results saved in `data/out_cpp_relu.txt`" << std::endl;
    return 0;
}
