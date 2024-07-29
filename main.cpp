#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <stdfloat>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

typedef std::bfloat16_t bf16;
typedef std::uint64_t u64;
typedef std::uint32_t u32;
typedef std::uint16_t u16;

using std::cout;
using std::endl;
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
    string filename;
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
        layer.filename = std::to_string(layers.size());
        line = line.substr(line.find(separator) + 1);
        layer.output_size = std::stoi(line.substr(0, line.find(separator)));
        layer.input_size = std::stoi(line.substr(line.find(separator) + 1));

        layers.push_back(layer);
    }

    return layers;
}

string get_layer_block(string layer_name) {
    std::istringstream iss(layer_name);
    string segment = "";
    string block_name = "";
    int count = 0;
    while (std::getline(iss, segment, '.') && count < 2) {
        if (count++) {
            block_name += '.';
        }

        block_name += segment;
    }

    return block_name;
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
            cout << this->data[i] << " ";
        }

        cout << endl;
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

struct Attention {
    Tensor wq;
    Tensor wk;
    Tensor wv;
    Tensor wo;

    Attention(Tensor wq, Tensor wk, Tensor wv, Tensor wo)
        : wq(wq), wk(wk), wv(wv), wo(wo) {}

    // TODO
    void forward(Tensor &x) {}
};

Attention load_attention(int block_number) {
    string root = Environment::getInstance().root;
    vector<LayerMeta> layers = get_layers(root + "layers");
    vector<LayerMeta> block_layers;
    for (LayerMeta layer : layers) {
        if (get_layer_block(layer.name) ==
                "layers." + std::to_string(block_number) &&
            layer.name.find("attention") != std::string::npos) {
            block_layers.push_back(layer);
        }
    }

    LayerMeta wq, wk, wv, wo;

    for (u32 i = 0; i < block_layers.size(); i++) {
        LayerMeta layer = block_layers[i];
        if (layer.name.find(".wq.") != std::string::npos) {
            wq = layer;
        } else if (layer.name.find(".wk.") != std::string::npos) {
            wk = layer;
        } else if (layer.name.find(".wv.") != std::string::npos) {
            wv = layer;
        } else if (layer.name.find(".wo.") != std::string::npos) {
            wo = layer;
        }
    }

    return Attention(
        Tensor({wq.input_size, wq.output_size}, root + "data/" + wq.filename),
        Tensor({wk.input_size, wk.output_size}, root + "data/" + wk.filename),
        Tensor({wv.input_size, wv.output_size}, root + "data/" + wv.filename),
        Tensor({wo.input_size, wo.output_size}, root + "data/" + wo.filename));
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
        std::cerr << e.what() << endl;
        return 1;
    }

    return 0;
}

// what even are these lol
// https://github.com/ggerganov/llama.cpp/blob/3855416027cb25d9a708ffa5581cf503a87856a6/ggml-impl.h#L90
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

// TODO: this function relies on the assumption that
//       num_heads and hidden_size are multiples of 2
//       that validation shouldn't be done here but somewhere else
//
// https://github.com/meta-llama/llama/blob/main/llama/model.py#L132
void apply_rotary_embeddings(Tensor &q, Tensor &k, Tensor &frequencies) {
    // q.shape = [seq_len, num_heads, hidden_size / num_heads]
    // k.shape = [seq_len, num_heads, hidden_size / num_heads]
    //
    // but the above are going to be treating as if they're polar coordinates
    // making their shapes essentially
    // [seq_len, num_heads, (hidden_size / num_heads) / 2, 2]
    // (but they won't be represented like that here)
    //
    // and so the frequencies shape
    // frequencies.shape = [seq_len, (hidden_size / num_heads) / 2, 2]
    // will need to be broadcasted like so
    // [seq_len, 1, (hidden_size / num_heads) / 2, 2]

    // this assumes that q.batches == k.batches and frequencies.batches == 1

    u32 &rows = q.shape[q.shape.size() - 2];
    u32 &cols = q.shape[q.shape.size() - 1];
    for (u32 b = 0; b < q.batches; b++) {
        for (u32 r = 0; r < rows; r++) {
            for (u32 c = 0; c < cols; c += 2) {
                // c == column index for q, k; row index for frequencies
                bf16 fr = frequencies.data[b * cols * 2 + c * 2];
                bf16 fi = frequencies.data[b * cols * 2 + c * 2 + 1];

                q.data[b * rows * cols + r * cols + c] *= fr;
                q.data[b * rows * cols + r * cols + c + 1] *= fi;

                k.data[b * rows * cols + r * cols + c] *= fr;
                k.data[b * rows * cols + r * cols + c + 1] *= fi;
            }
        }
    }
}

// https://github.com/meta-llama/llama/blob/main/llama/model.py#L80
// returns tensor of shape [end, dim / 2, 2]
Tensor get_frequency_tensor(u32 dim, u32 end) {
    const float theta = 10000;

    Tensor angles({dim / 2});
    for (u32 i = 0; i < dim; i += 2) {
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

            cout << "loaded " << layer.name << " " << layer.input_size << " "
                 << layer.output_size << endl;
        }
    } catch (std::exception &e) {
        std::cerr << e.what() << endl;
        return 1;
    }

    return 0;
}

int get_frequency_tensor_test() {
    try {
        Tensor freq_polar = get_frequency_tensor(512, 512);
        freq_polar.print();
    } catch (std::exception &e) {
        std::cerr << e.what() << endl;
        return 1;
    }

    return 0;
}

int bf16_float_conversion_test() {
    try {
        float f = 0.503;
        bf16 b = float_to_bf16(f);
        cout << b << endl;

        f = bf16_to_float(b);
        cout << f << endl;
    } catch (std::exception &e) {
        std::cerr << e.what() << endl;
        return 1;
    }

    return 0;
}

void memory_requirements_summary(int scale) {
    u64 scalar = 1;
    string label = "B";
    switch (scale) {
    case 0:
        scalar = 1;
        break;
    case 1:
        scalar = 1024;
        label = "KB";
        break;
    case 2:
        scalar = 1024 * 1024;
        label = "MB";
        break;
    case 3:
        scalar = 1024 * 1024 * 1024;
        label = "GB";
        break;
    }

    vector<LayerMeta> layers =
        get_layers(Environment::getInstance().root + "layers");

    std::map<string, u64> per_layer;
    std::map<string, u64> per_block;
    u64 total = 0;

    for (LayerMeta layer : layers) {
        u64 size = layer.input_size * layer.output_size * sizeof(bf16);
        total += size;
        per_layer[layer.name] = size;

        std::istringstream iss(layer.name);
        string segment = "";
        string block_name = "";
        int count = 0;
        while (std::getline(iss, segment, '.') && count < 2) {
            if (count++) {
                block_name += '.';
            }

            block_name += segment;
        }

        per_block[block_name] += size;
    }

    cout << "Per layer:" << endl;
    for (auto &pair : per_layer) {
        cout << "  " << pair.first << ": " << pair.second / scalar << " "
             << label << endl;
    }

    cout << endl;

    cout << "Per block:" << endl;
    for (auto &pair : per_block) {
        cout << "  " << pair.first << ": " << pair.second / scalar << " "
             << label << endl;
    }

    cout << endl << "Total: " << total / scalar << " " << label << endl;
}

int main() {
    memory_requirements_summary(2);
    return 0;
}
