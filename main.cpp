#include <cmath>
#include <iostream>
#include <sstream>

#include "include/kernels.h"
#include "include/types.h"

static const u32 seq_len = 5;

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

int apply_rotary_embeddings_test() {
    try {
        u32 seq_len = 5;
        u32 num_heads = 4;
        u32 hidden_size = 16;

        // [seq_len, num_heads, hidden_size / num_heads]
        Tensor q = random_tensor({seq_len, num_heads, hidden_size / num_heads});
        Tensor k = random_tensor({seq_len, num_heads, hidden_size / num_heads});

        cout << "q:" << endl;
        q.print();
        cout << endl;

        cout << "k:" << endl;
        k.print();
        cout << endl;

        Tensor frequencies = get_frequency_tensor(hidden_size / num_heads, 20);

        cout << "frequencies:" << endl;
        frequencies.print();
        cout << endl;

        apply_rotary_embeddings(q, k, frequencies);

        cout << "q:" << endl;
        q.print();
        cout << endl;

        cout << "k:" << endl;
        k.print();
        cout << endl;
    } catch (std::exception &e) {
        std::cerr << e.what() << endl;
        return 1;
    }

    return 0;
}

int attention_test() {
    try {
        ModelParams params = get_llama3_1_params();
        Attention attention = load_attention(1, params);

        Tensor x = random_tensor({seq_len, params.hidden_dim});
        Tensor frequencies =
            get_frequency_tensor(params.hidden_dim, params.max_seq_len);

        attention.forward(x, frequencies);

        cout << "output:" << endl;
        x.print();
    } catch (std::exception &e) {
        std::cerr << e.what() << endl;
        return 1;
    }

    return 0;
}

int feedforward_test() {
    try {
        ModelParams params = get_llama3_1_params();
        FeedForward feed_forward = load_feed_forward(1, params);

        Tensor x = random_tensor({seq_len, params.hidden_dim});

        feed_forward.forward(x);

        cout << "output:" << endl;
        x.print();
    } catch (std::exception &e) {
        std::cerr << e.what() << endl;
        return 1;
    }

    return 0;
}

int rms_norm_test(string type) {
    try {
        ModelParams params = get_llama3_1_params();
        RMSNorm rms_norm = load_rms_norm(1, type, params);

        Tensor x = random_tensor({seq_len, params.hidden_dim});

        rms_norm.forward(x);

        cout << "output:" << endl;
        x.print();
    } catch (std::exception &e) {
        std::cerr << e.what() << endl;
        return 1;
    }

    return 0;
}

int transformer_block_test() {
    try {
        ModelParams params = get_llama3_1_params();
        TransformerBlock block = load_transformer_block(1, params);

        Tensor x = random_tensor({seq_len, params.hidden_dim});
        Tensor frequencies =
            get_frequency_tensor(params.hidden_dim, params.max_seq_len);

        block.forward(x, frequencies);

        cout << "output:" << endl;
        x.print();
    } catch (std::exception &e) {
        std::cerr << e.what() << endl;
        return 1;
    }

    return 0;
}

int full_block_test() {
    ModelParams params = get_llama3_1_params();

    try {
        for (int i = 0; i < 32; i++) {
            cout << "block " << i << endl;

            TransformerBlock block = load_transformer_block(i, params);

            Tensor x = random_tensor({seq_len, params.hidden_dim});
            Tensor frequencies =
                get_frequency_tensor(params.hidden_dim, params.max_seq_len);

            block.forward(x, frequencies);
        }
    } catch (std::exception &e) {
        std::cerr << e.what() << endl;
        return 1;
    }

    return 0;
}

// only works for llama3.1-8b
void total_memory_requirements(u32 seq_len, ByteUnit scale) {
    ModelParams params = get_llama3_1_params();

    u64 total = 0;
    for (u32 i = 0; i < 32; i++) {
        Attention attention = load_attention(i, params);
        FeedForward feed_forward = load_feed_forward(i, params);

        u64 mem = attention.memory_requirements(seq_len) +
                  feed_forward.memory_requirements(seq_len);
        cout << "block " << i << ": " << mem / static_cast<u64>(scale) << " "
             << byte_unit_to_string(scale) << endl;

        total += mem;
    }

    cout << "total: " << total / static_cast<u64>(scale) << " "
         << byte_unit_to_string(scale) << endl;
}

int main() {
    full_block_test();

    return 0;
}
