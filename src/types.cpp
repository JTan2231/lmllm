#include "include/types.h"

#include <cmath>
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

#include "include/kernels.h"
#include "include/logger.h"

string byte_unit_to_string(ByteUnit unit) {
    switch (unit) {
    case ByteUnit::B:
        return "B";
    case ByteUnit::KB:
        return "KB";
    case ByteUnit::MB:
        return "MB";
    case ByteUnit::GB:
        return "GB";
    default:
        return "Unknown";
    }
}

Environment::Environment() {
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

    LOG_INFO("Loaded %zu layers from %s", layers.size(), filename.c_str());

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

ModelParams get_llama3_1_params() { return {4096, 1.3, 1024, 128, 8096}; }

// tensors are structured as [batch_sizes..., rows, columns]
// in memory, the column values are contiguous,
// and rows are in order
//
// `Tensor` is a misnomer--here it's just being treated as batched matrices
Tensor::Tensor(vector<u32> shape) {
    this->shape = shape;

    this->batches = 1;
    u64 size = 1;
    for (unsigned long i = 0; i < shape.size(); i++) {
        size *= shape[i];
        if (i < shape.size() - 2) {
            this->batches *= shape[i];
        }
    }

    this->data = (bf16 *)malloc(size * sizeof(bf16));
    this->mmapped = 0;
}

Tensor::Tensor(vector<u32> shape, string filename) {
    this->shape = shape;

    this->batches = 1;
    for (unsigned long i = 0; i < shape.size(); i++) {

        if (i < shape.size() - 2) {
            this->batches *= shape[i];
        }
    }

    int fd = open(filename.data(), O_RDONLY);
    size_t file_size = lseek(fd, 0, SEEK_END);
    this->data = (bf16 *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);

    if (this->data == MAP_FAILED) {
        throw std::invalid_argument("Error mapping file: " + filename);
    }

    close(fd);

    this->mmapped = file_size;
}

void Tensor::print() {
    u64 size = 1;
    for (unsigned long i = 0; i < shape.size(); i++) {
        size *= shape[i];
    }

    for (u64 i = 0; i < size; i++) {
        cout << this->data[i] << " ";
    }

    cout << endl;
}

void Tensor::transpose(vector<u32> permutation) {
    if (permutation.size() != this->shape.size()) {
        throw std::invalid_argument("Permutation must have the same number of "
                                    "dimensions as the tensor, "
                                    "got " +
                                    std::to_string(permutation.size()) +
                                    " and " +
                                    std::to_string(this->shape.size()));
    }

    for (u32 i = 0; i < permutation.size(); i++) {
        if (permutation[i] >= this->shape.size()) {
            throw std::invalid_argument(
                "Permutation must be a permutation of the tensor's dimensions, "
                "got " +
                std::to_string(permutation[i]) + " and " +
                std::to_string(this->shape.size()));
        }
    }

    LOG_INFO("transpose old shape %d %d %d", this->shape[0], this->shape[1],
             this->shape[2]);

    vector<u32> new_shape;
    for (u32 i = 0; i < permutation.size(); i++) {
        new_shape.push_back(this->shape[permutation[i]]);
    }

    LOG_INFO("transpose new shape %d %d %d", new_shape[0], new_shape[1],
             new_shape[2]);

    this->shape = new_shape;
}

void Tensor::view(vector<u32> shape) {
    u64 size = 1;
    for (unsigned long i = 0; i < shape.size(); i++) {
        size *= shape[i];
    }

    u64 this_size = 1;
    for (unsigned long i = 0; i < this->shape.size(); i++) {
        this_size *= this->shape[i];
    }

    if (size != this_size) {
        throw std::invalid_argument("View shape must have the same number of "
                                    "elements as the tensor, got " +
                                    std::to_string(size) + " and " +
                                    std::to_string(this_size));
    }

    this->shape = shape;
}

Tensor::~Tensor() {
    if (this->mmapped) {
        munmap(this->data, this->mmapped);
    } else {
        free(this->data);
    }
}

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

// TODO: query + key caching
Attention::Attention(ModelParams params, std::map<string, string> weight_map)
    : hidden_dim(params.hidden_dim), num_heads(params.num_heads) {
    if (hidden_dim % num_heads != 0) {
        throw std::invalid_argument(
            "Hidden dimension must be divisible by number of heads, got " +
            std::to_string(hidden_dim) + " and " + std::to_string(num_heads));
    }

    this->wq = new Tensor({hidden_dim, hidden_dim}, weight_map["wq"]);
    this->wk = new Tensor({hidden_dim, hidden_dim}, weight_map["wk"]);
    this->wv = new Tensor({hidden_dim, hidden_dim}, weight_map["wv"]);
    this->wo = new Tensor({hidden_dim, hidden_dim}, weight_map["wo"]);
}

Attention::~Attention() {
    delete this->wq;
    delete this->wk;
    delete this->wv;
    delete this->wo;
}

void Attention::forward(Tensor &x, Tensor &frequencies, Tensor &output) {
    u32 seq_len = x.shape[0];

    // x.shape = [seq_len, hidden_dim]
    Tensor *xq = new Tensor(x.shape);
    Tensor *xk = new Tensor(x.shape);
    Tensor *xv = new Tensor(x.shape);

    LOG_INFO("new tensors xq, xk, xv created with shapes %d %d", xq->shape[0],
             xq->shape[1]);

    matmul(x, *this->wq, *xq);
    LOG_INFO("matmul x wq done");
    matmul(x, *this->wk, *xk);
    LOG_INFO("matmul x wk done");
    matmul(x, *this->wv, *xv);
    LOG_INFO("matmul x wv done");

    u32 head_dim = hidden_dim / num_heads;

    // [seq_len, num_heads, head_dim]
    xq->view({xq->shape[0], num_heads, head_dim});
    xk->view({xk->shape[0], num_heads, head_dim});
    xv->view({xv->shape[0], num_heads, head_dim});
    LOG_INFO("views set");

    apply_rotary_embeddings(*xq, *xk, frequencies);
    LOG_INFO("apply_rotary_embeddings xq done");

    xq->transpose({1, 0, 2});

    xk->transpose({1, 0, 2}); // [num_heads, seq_len, head_dim]
    xk->transpose({0, 2, 1});

    xv->transpose({1, 0, 2});

    LOG_INFO("transposes done");

    Tensor *scores = new Tensor({xq->shape[0], xq->shape[1], xk->shape[2]});
    matmul(*xq, *xk, *scores);
    divide(*scores, float_to_bf16(sqrt(head_dim)));
    LOG_INFO("matmul xq xk done");

    // TODO: masking

    columnwise_softmax(*scores);
    LOG_INFO("columnwise_softmax done");

    Tensor *matched = new Tensor({num_heads, seq_len, head_dim});
    matmul(*scores, *xv, *matched);
    LOG_INFO("matmul scores xv done");
    matched->transpose({1, 0, 2});
    matched->view({seq_len, hidden_dim});

    matmul(*matched, *this->wo, output);
    LOG_INFO("matmul matched wo done");

    delete xq;
    delete xk;
    delete xv;
    delete scores;
    delete matched;
}

// assumes a batch size of 1
u64 Attention::memory_requirements(u32 seq_len) {
    // wq, wk, wv, wo should all be the same shape
    u64 total = 4;
    for (u32 i = 0; i < wq->shape.size(); i++) {
        total *= wq->shape[i];
    }

    // xq, xk, xv, scores, matched
    total += seq_len * hidden_dim * 5;

    return total * sizeof(bf16);
}

Attention load_attention(int block_number, ModelParams params) {
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

    std::map<string, string> weight_map;

    for (u32 i = 0; i < block_layers.size(); i++) {
        LayerMeta layer = block_layers[i];
        if (layer.name.find(".wq.") != std::string::npos) {
            LOG_INFO("Found wq: %s", layer.filename.c_str());
            weight_map["wq"] = root + "data/" + layer.filename;
        } else if (layer.name.find(".wk.") != std::string::npos) {
            LOG_INFO("Found wk: %s", layer.filename.c_str());
            weight_map["wk"] = root + "data/" + layer.filename;
        } else if (layer.name.find(".wv.") != std::string::npos) {
            LOG_INFO("Found wv: %s", layer.filename.c_str());
            weight_map["wv"] = root + "data/" + layer.filename;
        } else if (layer.name.find(".wo.") != std::string::npos) {
            LOG_INFO("Found wo: %s", layer.filename.c_str());
            weight_map["wo"] = root + "data/" + layer.filename;
        }
    }

    Attention a(params, weight_map);
    LOG_INFO("Loaded attention block %d", block_number);

    return a;
}

FeedForward::FeedForward(ModelParams params,
                         std::map<string, string> weight_map) {
    u32 ffn_hidden_dim = 4 * params.hidden_dim;
    ffn_hidden_dim = 2 * ffn_hidden_dim / 3;
    ffn_hidden_dim = float(ffn_hidden_dim) * params.ffn_dim_multiplier;
    ffn_hidden_dim =
        params.multiple_of *
        ((ffn_hidden_dim + params.multiple_of - 1) / params.multiple_of);

    this->w1 =
        new Tensor({params.hidden_dim, ffn_hidden_dim}, weight_map["w1"]);
    this->w2 =
        new Tensor({ffn_hidden_dim, params.hidden_dim}, weight_map["w2"]);
    this->w3 =
        new Tensor({params.hidden_dim, ffn_hidden_dim}, weight_map["w3"]);
}

FeedForward::~FeedForward() {
    delete this->w1;
    delete this->w2;
    delete this->w3;
}

void FeedForward::forward(Tensor &x, Tensor &output) {
    LOG_INFO("begin feedforward");

    Tensor *ffn1 = new Tensor({x.shape[0], this->w1->shape[1]});
    matmul(x, *this->w1, *ffn1);
    silu(*ffn1);

    Tensor *ffn2 = new Tensor({x.shape[0], this->w3->shape[1]});
    matmul(x, *this->w3, *ffn2);

    multiply(*ffn1, *ffn2);
    matmul(*ffn1, *this->w2, output);

    delete ffn1;
    delete ffn2;

    LOG_INFO("end feedforward");
}

u64 FeedForward::memory_requirements(u32 seq_len) {
    // w1, w2, w3 should all be the same size
    u64 total = 3;
    for (u32 i = 0; i < w1->shape.size(); i++) {
        total *= w1->shape[i];
    }

    // ffn1, ffn2
    total += seq_len * w1->shape[1] * 2;

    return total * sizeof(bf16);
}

FeedForward load_feed_forward(int block_number, ModelParams params) {
    string root = Environment::getInstance().root;
    vector<LayerMeta> layers = get_layers(root + "layers");
    vector<LayerMeta> block_layers;
    for (LayerMeta layer : layers) {
        if (get_layer_block(layer.name) ==
                "layers." + std::to_string(block_number) &&
            layer.name.find("feed_forward") != std::string::npos) {
            block_layers.push_back(layer);
        }
    }

    std::map<string, string> weight_map;

    for (u32 i = 0; i < block_layers.size(); i++) {
        LayerMeta layer = block_layers[i];
        if (layer.name.find(".w1.") != std::string::npos) {
            LOG_INFO("Found w1: %s", layer.filename.c_str());
            weight_map["w1"] = root + "data/" + layer.filename;
        } else if (layer.name.find(".w2.") != std::string::npos) {
            LOG_INFO("Found w2: %s", layer.filename.c_str());
            weight_map["w2"] = root + "data/" + layer.filename;
        } else if (layer.name.find(".w3.") != std::string::npos) {
            LOG_INFO("Found w3: %s", layer.filename.c_str());
            weight_map["w3"] = root + "data/" + layer.filename;
        }
    }

    if (weight_map.size() != 3) {
        throw std::invalid_argument("Missing weights for feedforward block " +
                                    std::to_string(block_number) +
                                    "--is everything there?");
    }

    FeedForward f(params, weight_map);
    LOG_INFO("Loaded feedforward block %d", block_number);

    return f;
}
