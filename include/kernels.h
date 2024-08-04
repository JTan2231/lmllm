#ifndef KERNELS_H
#define KERNELS_H

#include "types.h"

void matmul(Tensor &a, Tensor &b, Tensor &out);
// this is so unorganized lmfao please add some structure
void add(Tensor &a, Tensor &b);
void sqrt(Tensor &t);
void divide(Tensor &a, bf16 b);
void multiply(Tensor &a, Tensor &b);
void columnwise_softmax(Tensor &t);
void silu(Tensor &t);

bool nan_check(Tensor &t);

void apply_rotary_embeddings(Tensor &q, Tensor &k, Tensor &frequencies);
Tensor get_frequency_tensor(u32 dim, u32 end);

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

#endif // KERNELS_H
