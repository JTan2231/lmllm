#include "include/kernels.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <fcntl.h>
#include <immintrin.h>
#include <numeric>
#include <stdexcept>
#include <stdfloat>
#include <sys/mman.h>
#include <unistd.h>

#include "include/logger.h"
#include "include/types.h"

#ifdef NUMERIC_DEBUG
#define NAN_CHECK(tensor)                                                      \
    if (nan_check(tensor)) {                                                   \
        LOG_ERROR("NaNs in function: %s", __func__);                           \
        throw std::runtime_error(std::string("NaNs in function: ") +           \
                                 __func__);                                    \
    }

#define BOUND_CHECK(tensor)                                                    \
    if (bound_check(tensor, 0, 1)) {                                           \
        LOG_ERROR("bound breached in function: %s", __func__);                 \
        throw std::runtime_error(std::string("bound breached in function: ") + \
                                 __func__);                                    \
    }

#else
#define NAN_CHECK(tensor)
#define BOUND_CHECK(tensor)
#endif

#define CHECK_OUTPUT_NAN NAN_CHECK(out)

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

bool nan_check(Tensor &t) {
    u64 size = 1;
    for (u32 i = 0; i < t.shape.size(); i++) {
        size *= t.shape[i];
    }

    for (u32 i = 0; i < size; i++) {
        if (std::isnan(t.data[i])) {
            return true;
        }
    }

    return false;
}

bool bound_check(Tensor &t, f32 lower, f32 upper) {
    u64 size = 1;
    for (u32 i = 0; i < t.shape.size(); i++) {
        size *= t.shape[i];
    }

    for (u32 i = 0; i < size; i++) {
        if (t.data[i] < lower || t.data[i] > upper) {
            return true;
        }
    }

    return false;
}

bool bound_check(f32 val, f32 lower, f32 upper) {
    return val < lower || val > upper;
}

bool nan_check(f32 val) { return std::isnan(val); }

int getTotalSIMDRegisters() {
    std::array<int, 4> cpuInfo;
    int totalRegisters = 0;

    // Execute CPUID to get CPU features
    __asm__ __volatile__("cpuid"
                         : "=a"(cpuInfo[0]), "=b"(cpuInfo[1]), "=c"(cpuInfo[2]),
                           "=d"(cpuInfo[3])
                         : "a"(1)); // Requesting basic info with EAX=1

    // Check for SSE
    if (cpuInfo[3] & (1 << 25)) {
        totalRegisters = 8; // SSE has 8 registers (XMM0 to XMM7)
    }

    // Check for AVX
    if (cpuInfo[2] & (1 << 28)) {
        totalRegisters = 16; // AVX supports 16 registers (YMM0 to YMM15)
    }

    // Check for AVX-512
    __asm__ __volatile__(
        "cpuid"
        : "=a"(cpuInfo[0]), "=b"(cpuInfo[1]), "=c"(cpuInfo[2]), "=d"(cpuInfo[3])
        : "a"(7), "c"(0)); // EAX=7, ECX=0 for extended features

    if (cpuInfo[1] & (1 << 16)) {
        totalRegisters = 32; // AVX-512 supports 32 registers (ZMM0 to ZMM31)
    }

    return totalRegisters; // Return total number of SIMD registers available
}

void matmul_naive(Tensor &a, Tensor &b, Tensor &out) {
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

    u32 ar = a.shape[a.shape.size() - 2];
    u32 ac = a.shape[a.shape.size() - 1];
    u32 br = b.shape[b.shape.size() - 2];
    u32 bc = b.shape[b.shape.size() - 1];

    u32 batch_size = out_size / (ar * bc);

    std::vector<u32> a_strides(max_dim, 1);
    std::vector<u32> b_strides(max_dim, 1);
    for (int i = (int)max_dim - 2; i >= 0; --i) {
        u32 _i = (u32)i;
        a_strides[_i] =
            a_strides[_i + 1] * (_i < a_shape.size() ? a_shape[_i] : 1);
        b_strides[_i] =
            b_strides[_i + 1] * (_i < b_shape.size() ? b_shape[_i] : 1);
    }

    for (u32 batch = 0; batch < batch_size; ++batch) {
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

        a_offset *= ar * ac;
        b_offset *= br * bc;

        for (u32 i = 0; i < ar; ++i) {
            for (u32 j = 0; j < bc; ++j) {
                f32 sum = 0;
                for (u32 p = 0; p < ac; ++p) {
                    u32 a_idx = a_offset + i * ac + p;
                    u32 b_idx = b_offset + p * bc + j;
                    f32 s = a.data[a_idx] * b.data[b_idx];
                    sum += s;
                }

                out.data[batch * ar * bc + i * bc + j] = sum;
            }
        }
    }
}

void matmul_block(Tensor &a, Tensor &b, Tensor &out) {
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

    for (u32 i = 0; i < out_size; i++) {
        out.data[i] = 0;
    }

    u32 ar = a.shape[a.shape.size() - 2];
    u32 ac = a.shape[a.shape.size() - 1];
    u32 br = b.shape[b.shape.size() - 2];
    u32 bc = b.shape[b.shape.size() - 1];

    u32 batch_size = out_size / (ar * bc);

    std::vector<u32> a_strides(max_dim, 1);
    std::vector<u32> b_strides(max_dim, 1);
    for (int i = (int)max_dim - 2; i >= 0; --i) {
        u32 _i = (u32)i;
        a_strides[_i] =
            a_strides[_i + 1] * (_i < a_shape.size() ? a_shape[_i] : 1);
        b_strides[_i] =
            b_strides[_i + 1] * (_i < b_shape.size() ? b_shape[_i] : 1);
    }

    for (u32 batch = 0; batch < batch_size; ++batch) {
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

        a_offset *= ar * ac;
        b_offset *= br * bc;

        for (u32 col_chunk = 0; col_chunk < bc; col_chunk += block_size) {
            for (u32 row = 0; row < ar; row++) {
                for (u32 tile = 0; tile < br; tile += block_size) {
                    for (u32 tile_row = 0;
                         tile_row < std::min(ar - tile, block_size);
                         tile_row++) {
                        for (u32 idx = 0;
                             idx < std::min(bc - col_chunk, block_size);
                             idx += 8) {
                            __m256 a_vec = _mm256_loadu_ps(
                                &a.data[a_offset + row * ac + tile + tile_row]);
                            __m256 b_vec = _mm256_loadu_ps(
                                &b.data[b_offset + tile * bc + tile_row * bc +
                                        col_chunk + idx]);

                            __m256 prod = _mm256_mul_ps(a_vec, b_vec);

                            __m256 out_vec = _mm256_loadu_ps(
                                &out.data[batch * ar * bc + row * bc +
                                          col_chunk + idx]);

                            out_vec = _mm256_add_ps(out_vec, prod);
                            _mm256_storeu_ps(
                                &out.data[batch * ar * bc + row * bc +
                                          col_chunk + idx],
                                out_vec);
                        }
                    }
                }
            }
        }
    }
}

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

#ifdef MATMUL_BLOCK
    matmul_block(a, b, out);
#else
    matmul_naive(a, b, out);
#endif

    NAN_CHECK(out);

    LOG_INFO("matmui done");
}

void add(Tensor &a, Tensor &b) {
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
        a.data[i] += b.data[i];
    }

    NAN_CHECK(a);
}

void sqrt(Tensor &t) {
    u32 size = 1;
    for (u32 i = 0; i < t.shape.size(); i++) {
        size *= t.shape[i];
    }

    for (u64 i = 0; i < size; i++) {
        t.data[i] = (sqrtf((t.data[i])));
    }

    NAN_CHECK(t);
}

void divide(Tensor &a, f32 b) {
    u32 size = 1;
    for (u32 i = 0; i < a.shape.size(); i++) {
        size *= a.shape[i];
    }

    for (u64 i = 0; i < size; i++) {
        a.data[i] /= b;
    }

    NAN_CHECK(a);
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

    NAN_CHECK(a);
}

void columnwise_softmax(Tensor &t) {
    u32 &rows = t.shape[t.shape.size() - 2];
    u32 &cols = t.shape[t.shape.size() - 1];
    u32 mat_size = rows * cols;

    for (u32 b = 0; b < t.batches; b++) {
        for (u32 i = 0; i < rows; i++) {
            f32 sum = 0;
            for (u32 j = 0; j < cols; j++) {
                f32 &val = t.data[b * mat_size + i * cols + j];
                val = (expf(val));
                sum += val;
            }

            for (u32 j = 0; j < cols; j++) {
                t.data[b * mat_size + i * cols + j] /= sum;
                BOUND_CHECK(t.data[b * mat_size + i * cols + j]);
            }
        }
    }

    NAN_CHECK(t);
    BOUND_CHECK(t);
}

void silu(Tensor &t) {
    u32 size = 1;
    for (u32 i = 0; i < t.shape.size(); i++) {
        size *= t.shape[i];
    }

    for (u64 i = 0; i < size; i++) {
        t.data[i] = t.data[i] * (1 / (1 + expf(-(t.data[i]))));
    }

    NAN_CHECK(t);
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
                f32 fr = frequencies.data[b * cols * 2 + c * 2];
                f32 fi = frequencies.data[b * cols * 2 + c * 2 + 1];

                q.data[b * rows * cols + r * cols + c] *= fr;
                q.data[b * rows * cols + r * cols + c + 1] *= fi;

                k.data[b * rows * cols + r * cols + c] *= fr;
                k.data[b * rows * cols + r * cols + c + 1] *= fi;
            }
        }
    }

    NAN_CHECK(q);
}

// https://github.com/meta-llama/llama/blob/main/llama/model.py#L80
// returns tensor of shape [end, dim / 2, 2]
Tensor get_frequency_tensor(u32 dim, u32 end) {
    const float theta = 10000;

    Tensor angles({dim / 2});
    for (u32 i = 0; i < dim; i += 2) {
        angles.data[i / 2] = 1 / (powf(theta, float(i) / dim));
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
            freq_polar.data[i * cols * 2 + j * 2] =
                sinf(frequencies.data[i * cols + j]);

            freq_polar.data[i * cols * 2 + j * 2 + 1] =
                cosf(frequencies.data[i * cols + j]);
        }
    }

    NAN_CHECK(freq_polar);

    return freq_polar;
}
