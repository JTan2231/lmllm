#ifndef KERNELS_H
#define KERNELS_H

#include "types.h"

void matmul_naive(Tensor &a, Tensor &b, Tensor &out);
void matmul_block(Tensor &a, Tensor &b, Tensor &out);
void matmul(Tensor &a, Tensor &b, Tensor &out);
// this is so unorganized lmfao please add some structure
void add(Tensor &a, Tensor &b);
void sqrt(Tensor &t);
void divide(Tensor &a, f32 b);
void multiply(Tensor &a, Tensor &b);
void columnwise_softmax(Tensor &t);
void silu(Tensor &t);

bool nan_check(Tensor &t);

void apply_rotary_embeddings(Tensor &q, Tensor &k, Tensor &frequencies);
Tensor get_frequency_tensor(u32 dim, u32 end);

void row_col_major_switch(Tensor &t, u32 &row, u32 &col);

// what even is this lol
// https://github.com/ggerganov/llama.cpp/blob/3855416027cb25d9a708ffa5581cf503a87856a6/ggml-impl.h#L90
static inline f32 bf16_to_float(bf16 v) {
    union {
        bf16 b;
        u16 i;
    } conv;
    conv.b = v;

    union {
        f32 f;
        u32 i;
    } out;
    out.i = (u32(conv.i) << 16);

    return out.f;
}

#endif // KERNELS_H
