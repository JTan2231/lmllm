#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <stdfloat>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

typedef std::bfloat16_t bf16;
typedef std::uint64_t u64;
typedef std::uint32_t u32;
typedef std::uint16_t u16;

using std::string;
using std::vector;

// TODO: any optimization whatsoever
//       ..
//       this especially needs some of the
//       fundamental math operations to support bf16
//       might as well implement manually from
//       some optimized implementation

struct Environment {
    static Environment &getInstance() {
        static Environment instance;
        return instance;
    }

    string root;

  private:
    Environment() {
        const char *home_arr = std::getenv("HOME");
        if (home_arr == nullptr) {
            home_arr = std::getenv("USERPROFILE");
        }

        string home = (home_arr ? std::string(home_arr) : std::string()) + "/";

        std::ifstream config(home + ".config/lmllm/config");
        if (!config) {
            throw std::invalid_argument("Error opening config file");
        }

        this->root = "";

        std::string line;
        while (std::getline(config, line)) {
            if (line.find("root") != std::string::npos) {
                this->root = line.substr(line.find("=") + 1);
            }
        }

        if (this->root == "") {
            throw std::invalid_argument("Root not found in config");
        }
    }

    Environment(Environment const &) = delete;
    void operator=(Environment const &) = delete;
};

struct LayerMeta {
    string name;
    u32 input_size;
    u32 output_size;
};

vector<LayerMeta> get_layers(string filename) {
    vector<LayerMeta> layers;

    std::ifstream file(filename);
    if (!file) {
        throw std::invalid_argument("Error opening layers file: " + filename);
    }

    string line;
    string separator = " ";
    while (std::getline(file, line)) {
        LayerMeta layer;
        layer.name = line.substr(0, line.find(separator));
        line = line.substr(line.find(separator) + 1);
        layer.output_size = std::stoi(line.substr(0, line.find(separator)));
        layer.input_size = std::stoi(line.substr(line.find(separator) + 1));

        layers.push_back(layer);
    }

    return layers;
}

// tensors are structured as [batch_sizes..., rows, columns]
// in memory, the column values are contiguous,
// and rows are in order
//
// `Tensor` is a misnomer--here it's just being treated as batched matrices
struct Tensor {
    bf16 *data;
    u32 batches;
    vector<u32> shape;
    u64 size;
    size_t mmapped;

    Tensor(vector<u32> shape) {
        this->shape = shape;

        this->size = 1;
        this->batches = 1;
        for (unsigned long i = 0; i < shape.size(); i++) {
            this->size *= shape[i];
            if (i < shape.size() - 2) {
                this->batches *= shape[i];
            }
        }

        this->data = (bf16 *)malloc(size * sizeof(bf16));
        this->mmapped = 0;
    }

    Tensor(vector<u32> shape, string filename) {
        this->shape = shape;

        this->size = 1;
        this->batches = 1;
        for (unsigned long i = 0; i < shape.size(); i++) {

            if (i < shape.size() - 2) {
                this->batches *= shape[i];
            }
        }

        int fd = open(filename.data(), O_RDONLY);
        size_t file_size = lseek(fd, 0, SEEK_END);
        this->data =
            (bf16 *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);

        if (this->data == MAP_FAILED) {
            throw std::invalid_argument("Error mapping file: " + filename);
        }

        close(fd);

        this->mmapped = file_size;
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

    ~Tensor() {
        if (this->mmapped) {
            munmap(this->data, this->mmapped);
        } else {
            free(this->data);
        }
    }

    inline bf16 get(u64 index) const { return this->data[index]; }
};

Tensor random_tensor(vector<u32> shape) {
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
            "Output tensor shape must match the product of the input "
            "tensor "
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

int matmul_test() {
    try {
        Tensor a = random_tensor({2, 4, 8});
        Tensor b = random_tensor({1, 8, 4});
        Tensor out = random_tensor({2, 4, 4});

        matmul(a, b, out);

        out.print();
    } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}

static inline bf16 float_to_bf16(float v) {
    union {
        float f;
        u32 i;
    } conv;
    conv.f = v;

    union {
        bf16 b;
        u16 i;
    } out;
    out.i = (conv.i + (0x7FFF + ((conv.i >> 16) & 1))) >> 16;

    return out.b;
}

static inline float bf16_to_float(bf16 v) {
    union {
        bf16 b;
        u16 i;
    } conv;
    conv.b = v;

    union {
        float f;
        u32 i;
    } out;
    out.i = (u32(conv.i) << 16);

    return out.f;
}

// https://arxiv.org/pdf/1910.07467
// see also https://github.com/meta-llama/llama/blob/main/llama/model.py#L34
void rms_norm(Tensor &t, Tensor &weight) {
    static const float epsilon = 1e-6;

    u32 &rows = t.shape[t.shape.size() - 2];
    u32 &cols = t.shape[t.shape.size() - 1];
    u32 mat_size = rows * cols;

    if (weight.shape[0] != rows || weight.shape[1] != cols) {
        throw std::invalid_argument(
            "Weight shape must match tensor shape, got " +
            (std::to_string(weight.shape[0]) + ", " +
             std::to_string(weight.shape[1]) + ") and (" +
             std::to_string(rows) + ", " + std::to_string(cols)) +
            ")");
    }

    u32 &batches = t.batches;

    for (u32 b = 0; b < batches; b++) {
        for (u32 i = 0; i < rows; i++) {
            bf16 sum = 0;
            for (u32 j = 0; j < cols; j++) {
                bf16 &val = t.data[b * mat_size + i * cols + j];
                sum += val * val;
            }

            bf16 norm =
                float_to_bf16(1 / (sqrt(bf16_to_float(sum) / cols) + epsilon));

            for (u32 j = 0; j < cols; j++) {
                t.data[b * rows * cols + i * cols + j] *= norm;
            }
        }

        for (u32 i = 0; i < rows; i++) {
            for (u32 j = 0; j < cols; j++) {
                t.data[b * mat_size + i * cols + j] *=
                    weight.data[i * cols + j];
            }
        }
    }
}

// TODO
// void apply_rotary_embeddings(Tensor &q, Tensor &k, Tensor &frequencies) {}

// https://github.com/meta-llama/llama/blob/main/llama/model.py#L80
Tensor get_frequency_tensor(u32 dim, u32 end) {
    const float theta = 10000;

    Tensor angles({dim / 2});
    for (u32 i = 0; i < dim; i += 2) {
        // conversion between float and bf16
        //
        // what even is this lol
        // https://github.com/ggerganov/llama.cpp/blob/3855416027cb25d9a708ffa5581cf503a87856a6/ggml-impl.h#L90
        angles.data[i / 2] = 1 / float_to_bf16(pow(theta, float(i) / dim));
    }

    Tensor abs_values({end});
    for (u32 i = 0; i < end; i++) {
        abs_values.data[i] = i;
    }

    u32 rows = end;
    u32 cols = dim / 2;
    Tensor frequencies({rows, cols});

    // outer product
    for (u32 i = 0; i < rows; i++) {
        for (u32 j = 0; j < cols; j++) {
            frequencies.data[i * cols + j] =
                abs_values.data[i] * angles.data[j];
        }
    }

    // polar
    Tensor freq_polar({rows, cols, 2});
    for (u32 i = 0; i < rows; i++) {
        for (u32 j = 0; j < cols; j++) {
            // casting malarkey
            // trig functions for bf16, perhaps?
            freq_polar.data[i * cols * 2 + j * 2] = float_to_bf16(
                sin(bf16_to_float(frequencies.data[i * cols + j])));

            freq_polar.data[i * cols * 2 + j * 2 + 1] = float_to_bf16(
                cos(bf16_to_float(frequencies.data[i * cols + j])));
        }
    }

    return freq_polar;
}

int layer_load_test() {
    try {
        string root = Environment::getInstance().root;
        vector<LayerMeta> layers = get_layers(root + "layers");

        for (size_t i = 0; i < layers.size(); i++) {
            LayerMeta layer = layers[i];
            Tensor weights({layer.input_size, layer.output_size},
                           root + "data/" + std::to_string(i));

            std::cout << "loaded " << layer.name << " " << layer.input_size
                      << " " << layer.output_size << std::endl;
        }
    } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}

int get_frequency_tensor_test() {
    try {
        Tensor freq_polar = get_frequency_tensor(512, 512);
        freq_polar.print();
    } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}

int bf16_float_conversion_test() {
    try {
        float f = 0.503;
        bf16 b = float_to_bf16(f);
        std::cout << b << std::endl;

        f = bf16_to_float(b);
        std::cout << f << std::endl;
    } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}

int main() {
    get_frequency_tensor_test();
    return 0;
}
