#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <stdfloat>
#include <vector>

typedef std::bfloat16_t bf16;
typedef std::uint64_t u64;
typedef std::uint32_t u32;

// tensors are structured as [batch_sizes..., rows, columns]
// in memory, the column values are contiguous,
// and rows are in order
//
// `Tensor` is a misnomer--here it's just being treated as batched matrices
struct Tensor {
    bf16 *data;
    u32 batches;
    std::vector<u32> shape;

    Tensor(std::vector<u32> shape) {
        this->shape = shape;

        u64 size = 1;
        this->batches = 1;
        for (unsigned long i = 0; i < shape.size(); i++) {
            size *= shape[i];
            if (i < shape.size() - 2) {
                this->batches *= shape[i];
            }
        }

        this->data = (bf16 *)malloc(size * sizeof(bf16));
    }

    void print() {
        u64 size = 1;
        for (unsigned long i = 0; i < shape.size(); i++) {
            size *= shape[i];
        }

        for (u64 i = 0; i < size; i++) {
            std::cout << this->data[i] << " ";
        }

        std::cout << std::endl;
    }

    ~Tensor() { free(this->data); }

    inline bf16 get(u64 index) const { return this->data[index]; }
};

Tensor random_tensor(std::vector<u32> shape) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    Tensor t(shape);

    u64 size = 1;
    for (unsigned long i = 0; i < shape.size(); i++) {
        size *= shape[i];
    }

    for (u64 i = 0; i < size; i++) {
        t.data[i] = static_cast<bf16>(dis(gen));
    }

    return t;
}

void matmul(Tensor &a, Tensor &b, Tensor &out) {
    if (a.shape.size() < 2 || b.shape.size() < 2 || out.shape.size() < 2) {
        throw std::invalid_argument(
            "Input tensors must have at least 2 dimensions, got " +
            std::to_string(a.shape.size()) + ", " +
            std::to_string(b.shape.size()) + ", and " +
            std::to_string(out.shape.size()));
    }

    if (a.shape[a.shape.size() - 1] != b.shape[b.shape.size() - 2]) {
        throw std::invalid_argument(
            "Inner dimensions of input tensors must match, got " +
            std::to_string(a.shape[a.shape.size() - 1]) + " and " +
            std::to_string(b.shape[b.shape.size() - 2]));
    }

    if (out.shape[out.shape.size() - 2] != a.shape[a.shape.size() - 2] ||
        out.shape[out.shape.size() - 1] != b.shape[b.shape.size() - 1]) {
        throw std::invalid_argument(
            "Output tensor shape must match the product of the input tensor "
            "shapes, got " +
            std::to_string(out.shape[out.shape.size() - 2]) + " and " +
            std::to_string(out.shape[out.shape.size() - 1]) + " and " +
            std::to_string(a.shape[a.shape.size() - 2]) + " and " +
            std::to_string(b.shape[b.shape.size() - 1]));
    }

    if (a.batches != out.batches || b.batches != 1) {
        throw std::invalid_argument("Invalid batch sizes (a != out || b != 1), "
                                    "got " +
                                    std::to_string(a.batches) + " and " +
                                    std::to_string(b.batches) + " and " +
                                    std::to_string(out.batches));
    }

    // this will only ever be inference for input tensors against a single
    // weight tensor as such, b will always have 1 batch

    u32 &ar = a.shape[a.shape.size() - 2];
    u32 &ac = a.shape[a.shape.size() - 1];
    u32 &outr = out.shape[out.shape.size() - 2];
    u32 &outc = out.shape[out.shape.size() - 1];

    u32 a_stride = ar * ac;
    u32 out_stride = outr * outc;

    for (u32 batch = 0; batch < a.batches; batch++) {
        for (u32 i = 0; i < outr; i++) {
            for (u32 j = 0; j < outc; j++) {
                for (u32 k = 0; k < ac; k++) {
                    out.data[batch * out_stride + i * outc + j] +=
                        a.data[batch * a_stride + i * ac + k] *
                        b.data[k * ac + j];
                }
            }
        }
    }
}

int main() {
    Tensor a = random_tensor({2, 4, 8});
    Tensor b = random_tensor({1, 8, 4});
    Tensor out = random_tensor({2, 4, 4});

    matmul(a, b, out);

    out.print();

    return 0;
}
