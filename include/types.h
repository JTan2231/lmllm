#ifndef TYPES_H
#define TYPES_H

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fcntl.h>
#include <iostream>
#include <map>
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

enum class ByteUnit {
    B = 1,
    KB = 1024,
    MB = 1024 * 1024,
    GB = 1024 * 1024 * 1024
};

string byte_unit_to_string(ByteUnit unit);

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
    Environment();
    Environment(Environment const &) = delete;
    void operator=(Environment const &) = delete;
};

struct LayerMeta {
    string name;
    string filename;
    u32 input_size;
    u32 output_size;
};

vector<LayerMeta> get_layers(string filename);
string get_layer_block(string layer_name);

struct ModelParams {
    u32 hidden_dim;
    float ffn_dim_multiplier;
    bf16 epsilon;
    u32 multiple_of;
    u32 num_heads;
    u32 max_seq_len;
};

ModelParams get_llama3_1_params();

// tensors are structured as [batch_sizes..., rows, columns]
// in memory, the column values are contiguous,
// and rows are in order
//
// `Tensor` is a misnomer--here it's just being treated as batched matrices
struct Tensor {
    bf16 *data;
    u32 batches;
    vector<u32> shape;
    size_t mmapped;

    Tensor(vector<u32> shape);
    Tensor(vector<u32> shape, string filename);

    void print();

    void transpose(vector<u32> permutation);
    void view(vector<u32> shape);

    ~Tensor();

    inline bf16 get(u64 index) const { return this->data[index]; }
};

Tensor random_tensor(vector<u32> shape);

// TODO: query + key caching
struct Attention {
    Tensor *wq;
    Tensor *wk;
    Tensor *wv;
    Tensor *wo;

    u32 hidden_dim;
    u32 num_heads;

    Attention(ModelParams params, std::map<string, string> weight_map);

    ~Attention();

    void forward(Tensor &x, Tensor &frequencies);
    u64 memory_requirements(u32 seq_len);
};

Attention load_attention(int block_number, ModelParams params);

struct FeedForward {
    Tensor *w1;
    Tensor *w2;
    Tensor *w3;

    u32 hidden_dim;

    FeedForward(ModelParams params, std::map<string, string> weight_map);

    ~FeedForward();

    void forward(Tensor &x);
    u64 memory_requirements(u32 seq_len);
};

FeedForward load_feed_forward(int block_number, ModelParams params);

struct RMSNorm {
    Tensor *weight;
    bf16 epsilon;

    RMSNorm(ModelParams params, std::map<string, string> weight_map);

    ~RMSNorm();

    void forward(Tensor &x);
};

RMSNorm load_rms_norm(int block_number, string type, ModelParams params);

struct TransformerBlock {
    Attention *attention;
    FeedForward *feed_forward;

    RMSNorm *attention_norm;
    RMSNorm *feed_forward_norm;

    TransformerBlock(int block_number, ModelParams params);

    ~TransformerBlock();

    void forward(Tensor &x, Tensor &frequencies);
    u64 memory_requirements(u32 seq_len);
};

TransformerBlock load_transformer_block(int block_number, ModelParams params);

#endif // TYPES_H
