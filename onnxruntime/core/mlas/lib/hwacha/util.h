#ifndef util_h
#define util_h

#include <stdint.h>

#define MAX(a, b)  ((a>b)?a:b)
void hwacha_init();

struct ConvParams {
    int batch_size;
    int in_dim, out_dim;
    int kernel_size;
    int in_channels;
    int out_channels;
    int stride;
    int padding;
    bool bias;
    bool depthwise;
    int n_patches;
    int patch_size;
    int output_scale;
    int res_scale;
    int pool_size, pool_stride, pool_padding, out_dim_pooled;

    int I, J, K;
};

// int rdcycle();
// int rdinstret();
// void* safe_malloc(int size);
// void printfloatmatrix(int channels, int width, int height, float* M);
// void printintmatrix(int channels, int width, int height, int* M);
// void printint16matrix(int channels, int width, int height, int16_t* M);
// void fill_seq_32(float* p, int n, int mode);
// void fill_seq_16(int16_t* p, int n, int mode);
// void setvcfg(int nd, int nw, int nh, int np);
// int setvlen(int vlen);
//
// void memcpy_16(int16_t* src, int16_t* dest, int len);
// void memcpy_32(float* src, float* dest, int len);
//
// void cvt_32_16(float* src, int16_t* dest, int len);
// void cvt_16_32(int16_t* src, float* dest, int len);
//
// void gather_16(const int* id, const int16_t* src, int16_t* dest, int len);
// void gather_32(const int* id, const float* src, float* dest, int len);
// void fill_16(int N, float ALPHA, int16_t* X);
// void fill_32(int N, float ALPHA, float* X);
// void normalize_16(int16_t *x, int16_t *mean, int16_t *variance, int filters, int spatial);
// void normalize_32(float *x, float *mean, float *variance, int filters, int spatial);
// void scale_16(int16_t* x, int16_t scale, int size);
// void scale_32(float* x, float scale, int size);
// void add_16(int16_t* x, int16_t y, int size);
// void add_32(float* x, float y, int size);
//
// void square_32(int N, float* x, float* dest);
//
// void axpy_32(int N, float A, float* X, float* Y);
//
// void mul_32(int N, float* X, float* Y);
#endif
