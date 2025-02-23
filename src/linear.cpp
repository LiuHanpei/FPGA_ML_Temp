#include <iostream>
#include <fstream>
#include <vector>

#define BATCH_SIZE 1
#define IN_FEATURES 64
#define OUT_FEATURES 32
typedef float fm_t;

// **Linear Layer Computation**
void linear(fm_t X[BATCH_SIZE][IN_FEATURES], fm_t W[OUT_FEATURES][IN_FEATURES], fm_t B[OUT_FEATURES], fm_t Y[BATCH_SIZE][OUT_FEATURES]) {
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int out = 0; out < OUT_FEATURES; out++) {
            fm_t sum = B[out];  // Initialize with bias
            for (int in = 0; in < IN_FEATURES; in++) {
                sum += X[batch][in] * W[out][in];  // Matrix multiplication
            }
            Y[batch][out] = sum;
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
    fm_t X[BATCH_SIZE][IN_FEATURES];
    fm_t W[OUT_FEATURES][IN_FEATURES];
    fm_t B[OUT_FEATURES];
    fm_t Y[BATCH_SIZE][OUT_FEATURES];

    // **Load Data**
    std::vector<fm_t> x_data, w_data, b_data;
    load_data("data/X_linear.txt", x_data);
    load_data("data/W_linear.txt", w_data);
    load_data("data/B_linear.txt", b_data);

    // **Fill Buffers**
    int index = 0;
    for (int batch = 0; batch < BATCH_SIZE; batch++)
        for (int in = 0; in < IN_FEATURES; in++)
            X[batch][in] = x_data[index++];

    index = 0;
    for (int out = 0; out < OUT_FEATURES; out++)
        for (int in = 0; in < IN_FEATURES; in++)
            W[out][in] = w_data[index++];

    for (int out = 0; out < OUT_FEATURES; out++)
        B[out] = b_data[out];

    // **Execute Linear Computation**
    linear(X, W, B, Y);

    // **Save Computed Output**
    std::ofstream file("data/Y_cpp_linear.txt");
    if (!file) {
        std::cerr << "Cannot write to file: data/Y_cpp_linear.txt" << std::endl;
        return 1;
    }
    for (int batch = 0; batch < BATCH_SIZE; batch++)
        for (int out = 0; out < OUT_FEATURES; out++)
            file << Y[batch][out] << "\n";

    std::cout << "✅ C++ Linear computation completed, results saved in `data/Y_cpp_linear.txt`" << std::endl;
    return 0;
}
