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

ModelParams get_llama3_1_params() {
    return {4096, 1.3, f32(1e-5), 1024, 32, 8, 8096};
}

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

    this->data = (f32 *)malloc(size * sizeof(f32));
    for (u64 i = 0; i < size; i++) {
        this->data[i] = 0;
    }

    this->size = size;

    LOG_INFO("created tensor with size %zu", size);
}

Tensor::Tensor(vector<u32> shape, string filename) {
    LOG_INFO("loading tensor from file %s", filename.c_str());
    this->shape = shape;

    this->batches = 1;
    this->size = 1;
    for (unsigned long i = 0; i < shape.size(); i++) {
        this->size *= shape[i];
        if (i < shape.size() - 2) {
            this->batches *= shape[i];
        }
    }

    int fd = open(filename.data(), O_RDONLY);
    size_t file_size = lseek(fd, 0, SEEK_END);
    bf16 *file_data =
        (bf16 *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);

    u32 item_count = file_size / sizeof(bf16);
    if (item_count != this->size) {
        LOG_ERROR("File size does not match tensor size: %zu and %zu",
                  (size_t)item_count, this->size);
        LOG_ERROR("Filename: %s", filename.c_str());
        LOG_ERROR("Shape:");
        for (u32 i = 0; i < shape.size(); i++) {
            LOG_ERROR("%d ", shape[i]);
        }

        throw std::invalid_argument("File size does not match tensor size: " +
                                    std::to_string(item_count) + " and " +
                                    std::to_string(this->size));
    }

    this->data = (f32 *)malloc(item_count * sizeof(f32));
    for (u32 i = 0; i < item_count; i++) {
        this->data[i] = bf16_to_float(file_data[i]);
    }

    if (this->data == MAP_FAILED) {
        throw std::invalid_argument("Error mapping file: " + filename);
    }

    close(fd);
    munmap(this->data, file_size);

    LOG_INFO("created tensor with size %zu", size);
    LOG_INFO("shape: ");
    for (u32 i = 0; i < shape.size(); i++) {
        LOG_INFO("%d ", shape[i]);
    }
}

f32 &Tensor::operator[](u32 index) {
    if (index >= this->size) {
        throw std::invalid_argument(
            "Index out of bounds: " + std::to_string(index) + " and " +
            std::to_string(this->size));
    }

    return this->data[index];
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

Tensor::~Tensor() { free(this->data); }

Tensor random_tensor(vector<u32> shape) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> dis(-0.2f, 0.2f);

    Tensor t(shape);

    for (u64 i = 0; i < t.size; i++) {
        t.data[i] = dis(gen);
    }

    return t;
}

// TODO: query + key caching
Attention::Attention(ModelParams params, std::map<string, string> weight_map)
    : hidden_dim(params.hidden_dim), num_heads(params.num_heads),
      kv_heads(params.kv_heads) {
    if (hidden_dim % num_heads != 0) {
        throw std::invalid_argument(
            "Hidden dimension must be divisible by number of heads, got " +
            std::to_string(hidden_dim) + " and " + std::to_string(num_heads));
    }

    u32 head_dim = hidden_dim / num_heads;

    this->wq = new Tensor({hidden_dim, num_heads * head_dim}, weight_map["wq"]);
    this->wk = new Tensor({hidden_dim, kv_heads * head_dim}, weight_map["wk"]);
    this->wv = new Tensor({hidden_dim, kv_heads * head_dim}, weight_map["wv"]);
    this->wo = new Tensor({hidden_dim, num_heads * head_dim}, weight_map["wo"]);
}

Attention::~Attention() {
    delete this->wq;
    delete this->wk;
    delete this->wv;
    delete this->wo;
}

void Attention::forward(Tensor &x, Tensor &frequencies) {
    LOG_INFO("begin attention forward");

    u32 seq_len = x.shape[0];
    u32 head_dim = hidden_dim / num_heads;

    // x.shape = [seq_len, hidden_dim]
    Tensor *xq = new Tensor(x.shape);
    Tensor *xk = new Tensor({x.shape[0], kv_heads * head_dim});
    Tensor *xv = new Tensor({x.shape[0], kv_heads * head_dim});

    LOG_INFO("new tensors xq, xk, xv created with shapes %d %d", xq->shape[0],
             xq->shape[1]);

    matmul(x, *this->wq, *xq);
    LOG_INFO("matmul x wq done");
    matmul(x, *this->wk, *xk);
    LOG_INFO("matmul x wk done");
    matmul(x, *this->wv, *xv);
    LOG_INFO("matmul x wv done");

    // [seq_len, num_heads, head_dim]
    xq->view({xq->shape[0], num_heads, head_dim});
    xk->view({xk->shape[0], kv_heads, head_dim});
    xv->view({xv->shape[0], kv_heads, head_dim});
    LOG_INFO("views set");

    // keys and values need repeated to match num_heads
    // i hope this doesn't add too much overhead
    // pytorch is o(1), this isn't
    Tensor *keys = new Tensor({seq_len, num_heads, head_dim});
    Tensor *values = new Tensor({seq_len, num_heads, head_dim});

    if (num_heads % kv_heads != 0) {
        throw std::invalid_argument(
            "Number of heads must be divisible by number of key-value heads, "
            "got " +
            std::to_string(num_heads) + " and " + std::to_string(kv_heads));
    }

    for (u32 s = 0; s < seq_len; s++) {
        for (u32 r = 0; r < num_heads; r++) {
            for (u32 c = 0; c < head_dim; c++) {
                keys->data[s * num_heads * head_dim + r * head_dim + c] =
                    xk->data[s * kv_heads * head_dim +
                             (r / kv_heads) * head_dim + c];

                values->data[s * num_heads * head_dim + r * head_dim + c] =
                    xv->data[s * kv_heads * head_dim +
                             (r / kv_heads) * head_dim + c];
            }
        }
    }

    apply_rotary_embeddings(*xq, *keys, frequencies);
    LOG_INFO("apply_rotary_embeddings xq done");

    xq->transpose({1, 0, 2});

    keys->transpose({1, 0, 2}); // [num_heads, seq_len, head_dim]
    keys->transpose({0, 2, 1});

    values->transpose({1, 0, 2});

    LOG_INFO("transposes done");

    Tensor *scores = new Tensor({xq->shape[0], xq->shape[1], keys->shape[2]});
    matmul(*xq, *keys, *scores);
    divide(*scores, sqrtf(head_dim));
    LOG_INFO("matmul xq keys done");

    // TODO: masking

    columnwise_softmax(*scores);
    LOG_INFO("columnwise_softmax done");

    Tensor *matched = new Tensor({num_heads, seq_len, head_dim});
    matmul(*scores, *values, *matched);
    LOG_INFO("matmul scores values done");
    matched->transpose({1, 0, 2});
    matched->view({seq_len, hidden_dim});

    matmul(*matched, *this->wo, x);
    LOG_INFO("matmul matched wo done");

    delete xq;
    delete xk;
    delete xv;
    delete scores;
    delete matched;

    LOG_INFO("end attention forward");
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

    return total * sizeof(f32);
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

void FeedForward::forward(Tensor &x) {
    LOG_INFO("begin feedforward");

    Tensor *ffn1 = new Tensor({x.shape[0], this->w1->shape[1]});
    matmul(x, *this->w1, *ffn1);
    silu(*ffn1);

    Tensor *ffn2 = new Tensor({x.shape[0], this->w3->shape[1]});
    matmul(x, *this->w3, *ffn2);

    multiply(*ffn1, *ffn2);
    matmul(*ffn1, *this->w2, x);

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

    return total * sizeof(f32);
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

RMSNorm::RMSNorm(ModelParams params, std::map<string, string> weight_map) {
    this->weight = new Tensor({params.hidden_dim}, weight_map["weight"]);
    this->epsilon = params.epsilon;
}

// https://arxiv.org/pdf/1910.07467
// see also
// https://github.com/meta-llama/llama/blob/main/llama/model.py#L34
void RMSNorm::forward(Tensor &x) {
    LOG_INFO("begin rmsnorm forward");

    u32 &rows = x.shape[x.shape.size() - 2];
    u32 &cols = x.shape[x.shape.size() - 1];

    if (this->weight->shape[0] != cols) {
        throw std::invalid_argument(
            "Weight must be 1D and match input column length, got " +
            std::to_string(this->weight->shape[0]) + " and " +
            std::to_string(cols));
    }

    for (u32 i = 0; i < rows; i++) {
        f32 sum = 0;
        for (u32 j = 0; j < cols; j++) {
            f32 &val = x.data[i * cols + j];
            sum += val * val;
        }

        f32 norm = 1 / (sqrtf(sum / cols) + this->epsilon);

        for (u32 j = 0; j < cols; j++) {
            x.data[i * cols + j] *= norm;
        }
    }

    for (u32 i = 0; i < rows; i++) {
        for (u32 j = 0; j < cols; j++) {
            x.data[i * cols + j] *= this->weight->data[j];
        }
    }

    LOG_INFO("end rmsnorm forward");
}

RMSNorm::~RMSNorm() { delete this->weight; }

RMSNorm load_rms_norm(int block_number, string type, ModelParams params) {
    string root = Environment::getInstance().root;
    vector<LayerMeta> layers = get_layers(root + "layers");
    vector<LayerMeta> block_layers;
    for (LayerMeta layer : layers) {
        if (get_layer_block(layer.name) ==
                "layers." + std::to_string(block_number) &&
            layer.name.find(type + "_norm") != std::string::npos) {
            block_layers.push_back(layer);
        }
    }

    std::map<string, string> weight_map;

    for (u32 i = 0; i < block_layers.size(); i++) {
        LayerMeta layer = block_layers[i];
        if (layer.name.find("_norm.weight") != std::string::npos) {
            LOG_INFO("Found weight: %s", layer.filename.c_str());
            weight_map["weight"] = root + "data/" + layer.filename;
        }
    }

    if (weight_map.size() != 1) {
        throw std::invalid_argument("Missing weights for " + type + "_norm " +
                                    std::to_string(block_number) +
                                    "--is everything there?");
    }

    RMSNorm r(params, weight_map);
    LOG_INFO("Loaded rmsnorm block %d type %s", block_number, type.c_str());

    return r;
}

TransformerBlock::TransformerBlock(int block_number, ModelParams params) {
    this->attention = new Attention(load_attention(block_number, params));
    this->feed_forward =
        new FeedForward(load_feed_forward(block_number, params));

    this->attention_norm =
        new RMSNorm(load_rms_norm(block_number, "attention", params));
    this->feed_forward_norm =
        new RMSNorm(load_rms_norm(block_number, "ffn", params));
}

TransformerBlock::~TransformerBlock() {
    delete this->attention;
    delete this->feed_forward;
    delete this->attention_norm;
    delete this->feed_forward_norm;
}

void TransformerBlock::forward(Tensor &x, Tensor &frequencies) {
    LOG_INFO("begin transformer block forward");

    Tensor *intermediate = new Tensor(x.shape);

    LOG_INFO("intermediate tensor created with shape %d %d",
             intermediate->shape[0], intermediate->shape[1]);

    size_t x_size = 1;
    for (u32 i = 0; i < x.shape.size(); i++) {
        x_size *= x.shape[i];
    }

    for (u32 i = 0; i < x_size; i++) {
        intermediate->data[i] = x.data[i];
    }

    LOG_INFO("intermediate tensor created");

    this->attention_norm->forward(*intermediate);
    LOG_INFO("attention norm done");
    this->attention->forward(*intermediate, frequencies);
    LOG_INFO("attention forward done");
    add(*intermediate, x);
    LOG_INFO("attention residual done");

    this->feed_forward_norm->forward(*intermediate);
    this->feed_forward->forward(*intermediate);
    add(x, *intermediate);

    delete intermediate;

    LOG_INFO("end transformer block forward");
}

TransformerBlock load_transformer_block(int block_number, ModelParams params) {
    return TransformerBlock(block_number, params);
}
