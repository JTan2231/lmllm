#include "include/kernels.h"

#include <cmath>
#include <cstdlib>
#include <fcntl.h>
#include <numeric>
#include <stdexcept>
#include <stdfloat>
#include <sys/mman.h>
#include <unistd.h>

#include "include/logger.h"
#include "include/types.h"

string shape_to_string(vector<u32> &shape) {
    string s = "(";
    for (u32 i = 0; i < shape.size(); i++) {
        s += std::to_string(shape[i]);
        if (i < shape.size() - 1) {
            s += ", ";
        }
    }

    s += ")";

    return s;
}

// TODO: broadcasting
void matmul(Tensor &a, Tensor &b, Tensor &out) {
    LOG_INFO("matmul %s %s %s", shape_to_string(a.shape).c_str(),
             shape_to_string(b.shape).c_str(),
             shape_to_string(out.shape).c_str());

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

    std::vector<u32> a_shape(a.shape.begin(), a.shape.end() - 2);
    std::vector<u32> b_shape(b.shape.begin(), b.shape.end() - 2);

    std::vector<u32> broadcast_shape;
    size_t max_dim = std::max(a_shape.size(), b_shape.size());
    a_shape.insert(a_shape.begin(), max_dim - a_shape.size(), 1);
    b_shape.insert(b_shape.begin(), max_dim - b_shape.size(), 1);

    for (u32 i = 0; i < a_shape.size(); i++) {
        if (a_shape[i] != b_shape[i] && a_shape[i] != 1 && b_shape[i] != 1) {
            throw std::invalid_argument("Invalid broadcast dimensions, got " +
                                        std::to_string(a_shape[i]) + " and " +
                                        std::to_string(b_shape[i]));
        }
    }

    for (size_t i = 0; i < max_dim; ++i) {
        broadcast_shape.push_back(std::max(a_shape[i], b_shape[i]));
    }

    std::vector<u32> &out_shape = out.shape;
    u32 out_size = std::accumulate(out_shape.begin(), out_shape.end(), 1,
                                   std::multiplies<u32>());

    u32 m = a.shape[a.shape.size() - 2];
    u32 n = b.shape.back();
    u32 k = a.shape.back();
    u32 batch_size = out_size / (m * n);

    std::vector<u32> a_strides(max_dim + 1, 1);
    std::vector<u32> b_strides(max_dim + 1, 1);
    for (int i = (int)max_dim - 1; i >= 0; --i) {
        u32 _i = (u32)i;
        a_strides[_i] =
            a_strides[_i + 1] * (_i < a_shape.size() ? a_shape[_i] : 1);
        b_strides[_i] =
            b_strides[_i + 1] * (_i < b_shape.size() ? b_shape[_i] : 1);
    }

    for (u32 batch = 0; batch < batch_size; ++batch) {
        LOG_INFO("starting batch %d", batch);
        std::vector<u32> batch_indices(max_dim, 0);
        u32 temp = batch;
        for (int i = max_dim - 1; i >= 0; --i) {
            batch_indices[(u32)i] = temp % broadcast_shape[(u32)i];
            temp /= broadcast_shape[(u32)i];
        }

        u32 a_offset = 0, b_offset = 0;
        for (size_t i = 0; i < max_dim; ++i) {
            a_offset += (batch_indices[i] % a_shape[i]) * a_strides[i];
            b_offset += (batch_indices[i] % b_shape[i]) * b_strides[i];
        }

        for (u32 i = 0; i < m; ++i) {
            LOG_INFO("%d / %d", i, m);
            for (u32 j = 0; j < n; ++j) {
                bf16 sum = 0;
                for (u32 p = 0; p < k; ++p) {
                    u32 a_idx = a_offset + i * k + p;
                    u32 b_idx = b_offset + p * n + j;
                    sum += a.data[a_idx] * b.data[b_idx];
                }

                out.data[batch * m * n + i * n + j] = sum;
            }
        }
    }

    LOG_INFO("matmui done");
}

void sqrt(Tensor &t) {
    u32 size = 1;
    for (u32 i = 0; i < t.shape.size(); i++) {
        size *= t.shape[i];
    }

    for (u64 i = 0; i < size; i++) {
        t.data[i] = float_to_bf16(sqrt(bf16_to_float(t.data[i])));
    }
}

void divide(Tensor &a, bf16 b) {
    u32 size = 1;
    for (u32 i = 0; i < a.shape.size(); i++) {
        size *= a.shape[i];
    }

    for (u64 i = 0; i < size; i++) {
        a.data[i] /= b;
    }
}

void multiply(Tensor &a, Tensor &b) {
    if (a.shape.size() != b.shape.size()) {
        throw std::invalid_argument(
            "Tensors must have the same number of dimensions, got " +
            std::to_string(a.shape.size()) + " and " +
            std::to_string(b.shape.size()));
    }

    for (u32 i = 0; i < a.shape.size(); i++) {
        if (a.shape[i] != b.shape[i]) {
            throw std::invalid_argument(
                "Tensors must have the same shape, got " +
                std::to_string(a.shape[i]) + " and " +
                std::to_string(b.shape[i]));
        }
    }

    u32 size = 1;
    for (u32 i = 0; i < a.shape.size(); i++) {
        size *= a.shape[i];
    }

    for (u64 i = 0; i < size; i++) {
        a.data[i] *= b.data[i];
    }
}

void columnwise_softmax(Tensor &t) {
    u32 &rows = t.shape[t.shape.size() - 2];
    u32 &cols = t.shape[t.shape.size() - 1];
    u32 mat_size = rows * cols;

    for (u32 b = 0; b < t.batches; b++) {
        for (u32 i = 0; i < rows; i++) {
            bf16 sum = 0;
            for (u32 j = 0; j < cols; j++) {
                bf16 &val = t.data[b * mat_size + i * cols + j];
                val = float_to_bf16(exp(val));
                sum += val;
            }

            for (u32 j = 0; j < cols; j++) {
                t.data[b * mat_size + i * cols + j] /= sum;
            }
        }
    }
}

void silu(Tensor &t) {
    u32 size = 1;
    for (u32 i = 0; i < t.shape.size(); i++) {
        size *= t.shape[i];
    }

    for (u64 i = 0; i < size; i++) {
        t.data[i] =
            t.data[i] * float_to_bf16(1 / (1 + exp(-bf16_to_float(t.data[i]))));
    }
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
