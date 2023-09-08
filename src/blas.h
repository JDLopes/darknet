#ifndef BLAS_H
#define BLAS_H
#include "darknet.h"

void flatten(Unum4 *x, int size, int layers, int batch, int forward);
void pm(int M, int N, Unum4 *A);
Unum4 *random_matrix(int rows, int cols);
void time_random_matrix(int TA, int TB, int m, int k, int n);
void reorg_cpu(Unum4 *x, int w, int h, int c, int batch, int stride, int forward, Unum4 *out);

void test_blas();

void inter_cpu(int NX, Unum4 *X, int NY, Unum4 *Y, int B, Unum4 *OUT);
void deinter_cpu(int NX, Unum4 *X, int NY, Unum4 *Y, int B, Unum4 *OUT);
void mult_add_into_cpu(int N, Unum4 *X, Unum4 *Y, Unum4 *Z);

void const_cpu(int N, Unum4 ALPHA, Unum4 *X, int INCX);
void constrain_gpu(int N, Unum4 ALPHA, Unum4 * X, int INCX);
void pow_cpu(int N, Unum4 ALPHA, Unum4 *X, int INCX, Unum4 *Y, int INCY);
void mul_cpu(int N, Unum4 *X, int INCX, Unum4 *Y, int INCY);

int test_gpu_blas();
void shortcut_cpu(int batch, int w1, int h1, int c1, Unum4 *add, int w2, int h2, int c2, Unum4 s1, Unum4 s2, Unum4 *out);

void mean_cpu(Unum4 *x, int batch, int filters, int spatial, Unum4 *mean);
void variance_cpu(Unum4 *x, Unum4 *mean, int batch, int filters, int spatial, Unum4 *variance);

void scale_bias(Unum4 *output, Unum4 *scales, int batch, int n, int size);
void backward_scale_cpu(Unum4 *x_norm, Unum4 *delta, int batch, int n, int size, Unum4 *scale_updates);
void mean_delta_cpu(Unum4 *delta, Unum4 *variance, int batch, int filters, int spatial, Unum4 *mean_delta);
void  variance_delta_cpu(Unum4 *x, Unum4 *delta, Unum4 *mean, Unum4 *variance, int batch, int filters, int spatial, Unum4 *variance_delta);
void normalize_delta_cpu(Unum4 *x, Unum4 *mean, Unum4 *variance, Unum4 *mean_delta, Unum4 *variance_delta, int batch, int filters, int spatial, Unum4 *delta);
void l2normalize_cpu(Unum4 *x, Unum4 *dx, int batch, int filters, int spatial);

void smooth_l1_cpu(int n, Unum4 *pred, Unum4 *truth, Unum4 *delta, Unum4 *error);
void l2_cpu(int n, Unum4 *pred, Unum4 *truth, Unum4 *delta, Unum4 *error);
void l1_cpu(int n, Unum4 *pred, Unum4 *truth, Unum4 *delta, Unum4 *error);
void logistic_x_ent_cpu(int n, Unum4 *pred, Unum4 *truth, Unum4 *delta, Unum4 *error);
void softmax_x_ent_cpu(int n, Unum4 *pred, Unum4 *truth, Unum4 *delta, Unum4 *error);
void weighted_sum_cpu(Unum4 *a, Unum4 *b, Unum4 *s, int num, Unum4 *c);
void weighted_delta_cpu(Unum4 *a, Unum4 *b, Unum4 *s, Unum4 *da, Unum4 *db, Unum4 *ds, int n, Unum4 *dc);

void softmax(Unum4 *input, int n, Unum4 temp, int stride, Unum4 *output);
void softmax_cpu(Unum4 *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, Unum4 temp, Unum4 *output);
void upsample_cpu(Unum4 *in, int w, int h, int c, int batch, int stride, int forward, Unum4 scale, Unum4 *out);

#ifdef GPU
#include "cuda.h"
#include "tree.h"

void axpy_gpu(int N, Unum4 ALPHA, Unum4 * X, int INCX, Unum4 * Y, int INCY);
void axpy_gpu_offset(int N, Unum4 ALPHA, Unum4 * X, int OFFX, int INCX, Unum4 * Y, int OFFY, int INCY);
void copy_gpu(int N, Unum4 * X, int INCX, Unum4 * Y, int INCY);
void copy_gpu_offset(int N, Unum4 * X, int OFFX, int INCX, Unum4 * Y, int OFFY, int INCY);
void add_gpu(int N, Unum4 ALPHA, Unum4 * X, int INCX);
void supp_gpu(int N, Unum4 ALPHA, Unum4 * X, int INCX);
void mask_gpu(int N, Unum4 * X, Unum4 mask_num, Unum4 * mask, Unum4 val);
void scale_mask_gpu(int N, Unum4 * X, Unum4 mask_num, Unum4 * mask, Unum4 scale);
void const_gpu(int N, Unum4 ALPHA, Unum4 *X, int INCX);
void pow_gpu(int N, Unum4 ALPHA, Unum4 *X, int INCX, Unum4 *Y, int INCY);
void mul_gpu(int N, Unum4 *X, int INCX, Unum4 *Y, int INCY);

void mean_gpu(Unum4 *x, int batch, int filters, int spatial, Unum4 *mean);
void variance_gpu(Unum4 *x, Unum4 *mean, int batch, int filters, int spatial, Unum4 *variance);
void normalize_gpu(Unum4 *x, Unum4 *mean, Unum4 *variance, int batch, int filters, int spatial);
void l2normalize_gpu(Unum4 *x, Unum4 *dx, int batch, int filters, int spatial);

void normalize_delta_gpu(Unum4 *x, Unum4 *mean, Unum4 *variance, Unum4 *mean_delta, Unum4 *variance_delta, int batch, int filters, int spatial, Unum4 *delta);

void fast_mean_delta_gpu(Unum4 *delta, Unum4 *variance, int batch, int filters, int spatial, Unum4 *mean_delta);
void fast_variance_delta_gpu(Unum4 *x, Unum4 *delta, Unum4 *mean, Unum4 *variance, int batch, int filters, int spatial, Unum4 *variance_delta);

void fast_variance_gpu(Unum4 *x, Unum4 *mean, int batch, int filters, int spatial, Unum4 *variance);
void fast_mean_gpu(Unum4 *x, int batch, int filters, int spatial, Unum4 *mean);
void shortcut_gpu(int batch, int w1, int h1, int c1, Unum4 *add, int w2, int h2, int c2, Unum4 s1, Unum4 s2, Unum4 *out);
void scale_bias_gpu(Unum4 *output, Unum4 *biases, int batch, int n, int size);
void backward_scale_gpu(Unum4 *x_norm, Unum4 *delta, int batch, int n, int size, Unum4 *scale_updates);
void scale_bias_gpu(Unum4 *output, Unum4 *biases, int batch, int n, int size);
void add_bias_gpu(Unum4 *output, Unum4 *biases, int batch, int n, int size);
void backward_bias_gpu(Unum4 *bias_updates, Unum4 *delta, int batch, int n, int size);

void logistic_x_ent_gpu(int n, Unum4 *pred, Unum4 *truth, Unum4 *delta, Unum4 *error);
void softmax_x_ent_gpu(int n, Unum4 *pred, Unum4 *truth, Unum4 *delta, Unum4 *error);
void smooth_l1_gpu(int n, Unum4 *pred, Unum4 *truth, Unum4 *delta, Unum4 *error);
void l2_gpu(int n, Unum4 *pred, Unum4 *truth, Unum4 *delta, Unum4 *error);
void l1_gpu(int n, Unum4 *pred, Unum4 *truth, Unum4 *delta, Unum4 *error);
void wgan_gpu(int n, Unum4 *pred, Unum4 *truth, Unum4 *delta, Unum4 *error);
void weighted_delta_gpu(Unum4 *a, Unum4 *b, Unum4 *s, Unum4 *da, Unum4 *db, Unum4 *ds, int num, Unum4 *dc);
void weighted_sum_gpu(Unum4 *a, Unum4 *b, Unum4 *s, int num, Unum4 *c);
void mult_add_into_gpu(int num, Unum4 *a, Unum4 *b, Unum4 *c);
void inter_gpu(int NX, Unum4 *X, int NY, Unum4 *Y, int B, Unum4 *OUT);
void deinter_gpu(int NX, Unum4 *X, int NY, Unum4 *Y, int B, Unum4 *OUT);

void reorg_gpu(Unum4 *x, int w, int h, int c, int batch, int stride, int forward, Unum4 *out);

void softmax_gpu(Unum4 *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, Unum4 temp, Unum4 *output);
void adam_update_gpu(Unum4 *w, Unum4 *d, Unum4 *m, Unum4 *v, Unum4 B1, Unum4 B2, Unum4 eps, Unum4 decay, Unum4 rate, int n, int batch, int t);
void adam_gpu(int n, Unum4 *x, Unum4 *m, Unum4 *v, Unum4 B1, Unum4 B2, Unum4 rate, Unum4 eps, int t);

void flatten_gpu(Unum4 *x, int spatial, int layers, int batch, int forward, Unum4 *out);
void softmax_tree(Unum4 *input, int spatial, int batch, int stride, Unum4 temp, Unum4 *output, tree hier);
void upsample_gpu(Unum4 *in, int w, int h, int c, int batch, int stride, int forward, Unum4 scale, Unum4 *out);

#endif
#endif
