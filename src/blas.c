#include "blas.h"
#include "format.h"

#include <math.h>
#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void reorg_cpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out)
{
    int b,i,j,k;
    int out_c = c/(stride*stride);

    for(b = 0; b < batch; ++b){
        for(k = 0; k < c; ++k){
            for(j = 0; j < h; ++j){
                for(i = 0; i < w; ++i){
                    int in_index  = i + w*(j + h*(k + c*b));
                    int c2 = k % out_c;
                    int offset = k / out_c;
                    int w2 = i*stride + offset % stride;
                    int h2 = j*stride + offset / stride;
                    int out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));
                    if(forward) out[out_index] = x[in_index];
                    else out[in_index] = x[out_index];
                }
            }
        }
    }
}

void flatten(float *x, int size, int layers, int batch, int forward)
{
    float *swap = calloc(size*layers*batch, sizeof(float));
    int i,c,b;
    for(b = 0; b < batch; ++b){
        for(c = 0; c < layers; ++c){
            for(i = 0; i < size; ++i){
                int i1 = b*layers*size + c*size + i;
                int i2 = b*layers*size + i*layers + c;
                if (forward) swap[i2] = x[i1];
                else swap[i1] = x[i2];
            }
        }
    }
    memcpy(x, swap, size*layers*batch*sizeof(float));
    free(swap);
}

void weighted_sum_cpu(float *a, float *b, float *s, int n, float *c)
{
    int i;
    for(i = 0; i < n; ++i){
        //c[i] = s[i]*a[i] + (1-s[i])*(b ? b[i] : 0);
        c[i] = add(mul(s[i], a[i]), mul(sub(ONE, s[i]), (b ? b[i] : ZERO)));
    }
}

void weighted_delta_cpu(float *a, float *b, float *s, float *da, float *db, float *ds, int n, float *dc)
{
    int i;
    for(i = 0; i < n; ++i){
        //if(da) da[i] += dc[i] * s[i];
        if(da) da[i] = add(da[i], mul(dc[i], s[i]));
        //if(db) db[i] += dc[i] * (1-s[i]);
        if(db) db[i] = add(db[i], mul(dc[i], sub(ONE, s[i])));
        //ds[i] += dc[i] * (a[i] - b[i]);
        ds[i] = add(ds[i], mul(dc[i], sub(a[i], b[i])));
    }
}

void shortcut_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out)
{
    int stride = w1/w2;
    int sample = w2/w1;
    assert(stride == h1/h2);
    assert(sample == h2/h1);
    if(stride < 1) stride = 1;
    if(sample < 1) sample = 1;
    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;

    int i,j,k,b;
    for(b = 0; b < batch; ++b){
        for(k = 0; k < minc; ++k){
            for(j = 0; j < minh; ++j){
                for(i = 0; i < minw; ++i){
                    int out_index = i*sample + w2*(j*sample + h2*(k + c2*b));
                    int add_index = i*stride + w1*(j*stride + h1*(k + c1*b));
                    //out[out_index] = s1*out[out_index] + s2*add[add_index];
                    out[out_index] = add(mul(s1, out[out_index]), mul(s2, add[add_index]));
                }
            }
        }
    }
}

void mean_cpu(float *x, int batch, int filters, int spatial, float *mean)
{
    //float scale = 1./(batch * spatial);
    float scale = float2type(1./(batch * spatial));
    int i,j,k;
    for(i = 0; i < filters; ++i){
        //mean[i] = 0;
        mean[i] = ZERO;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                //mean[i] += x[index];
                mean[i] = add(mean[i], x[index]);
            }
        }
        //mean[i] *= scale;
        mean[i] = mul(mean[i], scale);
    }
}

void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    //float scale = 1./(batch * spatial - 1);
    float scale = float2type(1./(batch * spatial - 1));
    int i,j,k;
    for(i = 0; i < filters; ++i){
        //variance[i] = 0;
        variance[i] = ZERO;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                //variance[i] += pow((x[index] - mean[i]), 2);
                variance[i] = add(variance[i], pow(sub(x[index], mean[i]), TWO));
            }
        }
        //variance[i] *= scale;
        variance[i] = mul(variance[i], scale);
    }
}

void l2normalize_cpu(float *x, float *dx, int batch, int filters, int spatial)
{
    int b,f,i;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < spatial; ++i){
            //float sum = 0;
            float sum = ZERO;
            for(f = 0; f < filters; ++f){
                int index = b*filters*spatial + f*spatial + i;
                //sum += powf(x[index], 2);
                sum = add(sum, powf(x[index], 2));
            }
            sum = sqrtf(sum);
            for(f = 0; f < filters; ++f){
                int index = b*filters*spatial + f*spatial + i;
                //x[index] /= sum;
                x[index] = div(x[index], sum);
                //dx[index] = (1 - x[index]) / sum;
                dx[index] = div(sub(1, x[index]), sum);
            }
        }
    }
}


void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    int b, f, i;
    for(b = 0; b < batch; ++b){
        for(f = 0; f < filters; ++f){
            for(i = 0; i < spatial; ++i){
                int index = b*filters*spatial + f*spatial + i;
                //x[index] = (x[index] - mean[f])/(sqrt(variance[f]) + .000001f);
                x[index] = div(sub(x[index], mean[f]), add(sqrt(variance[f]), float2type(.000001f)));
            }
        }
    }
}

void const_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}

void mul_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    //for(i = 0; i < N; ++i) Y[i*INCY] *= X[i*INCX];
    for(i = 0; i < N; ++i) Y[i*INCY] = mul(Y[i*INCY], X[i*INCX]);
}

void pow_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] = pow(X[i*INCX], ALPHA);
}

void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i;
    //for(i = 0; i < N; ++i) Y[i*INCY] += ALPHA*X[i*INCX];
    for(i = 0; i < N; ++i) Y[i*INCY] = add(Y[i*INCY], mul(ALPHA, X[i*INCX]));
}

void scal_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    //for(i = 0; i < N; ++i) X[i*INCX] *= ALPHA;
    for(i = 0; i < N; ++i) X[i*INCX] = mul(X[i*INCX], ALPHA);
}

void fill_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}

void deinter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT)
{
    int i, j;
    int index = 0;
    for(j = 0; j < B; ++j) {
        for(i = 0; i < NX; ++i){
            //if(X) X[j*NX + i] += OUT[index];
            if(X) X[j*NX + i] = add(X[j*NX + i], OUT[index]);
            ++index;
        }
        for(i = 0; i < NY; ++i){
            //if(Y) Y[j*NY + i] += OUT[index];
            if(Y) Y[j*NY + i] = add(Y[j*NY + i], OUT[index]);
            ++index;
        }
    }
}

void inter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT)
{
    int i, j;
    int index = 0;
    for(j = 0; j < B; ++j) {
        for(i = 0; i < NX; ++i){
            OUT[index++] = X[j*NX + i];
        }
        for(i = 0; i < NY; ++i){
            OUT[index++] = Y[j*NY + i];
        }
    }
}

void copy_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] = X[i*INCX];
}

void mult_add_into_cpu(int N, float *X, float *Y, float *Z)
{
    int i;
    //for(i = 0; i < N; ++i) Z[i] += X[i]*Y[i];
    for(i = 0; i < N; ++i) Z[i] = add(Z[i], mul(X[i], Y[i]));
}

void smooth_l1_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        //float diff = truth[i] - pred[i];
        float diff = sub(truth[i], pred[i]);
        float abs_val = fabs(diff);
        //if(abs_val < 1) {
        if(lt(abs_val, ONE)) {
            //error[i] = diff * diff;
            error[i] = mul(diff, diff);
            delta[i] = diff;
        }
        else {
            //error[i] = 2*abs_val - 1;
            error[i] = sub(mul(TWO, abs_val), ONE);
            //delta[i] = (diff < 0) ? 1 : -1;
            delta[i] = lt(diff, ZERO) ? ONE : neg(ONE);
        }
    }
}

void l1_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        //float diff = truth[i] - pred[i];
        float diff = sub(truth[i], pred[i]);
        error[i] = fabs(diff);
        //delta[i] = diff > 0 ? 1 : -1;
        delta[i] = gt(diff, 0)? ONE : neg(ONE);
    }
}

void softmax_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float t = truth[i];
        float p = pred[i];
        //error[i] = (t) ? -log(p) : 0;
        error[i] = !eq(t, ZERO) ? neg(log(p)) : ZERO;
        //delta[i] = t-p;
        delta[i] = sub(t, p);
    }
}

void logistic_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        float t = truth[i];
        float p = pred[i];
        //error[i] = -t*log(p) - (1-t)*log(1-p);
        error[i] = sub(mul(neg(t), log(p)), mul(sub(ONE, t), log(sub(ONE, p))));
        //delta[i] = t-p;
        delta[i] = sub(t, p);
    }
}

void l2_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    int i;
    for(i = 0; i < n; ++i){
        //float diff = truth[i] - pred[i];
        float diff = sub(truth[i], pred[i]);
        //error[i] = diff * diff;
        error[i] = mul(diff, diff);
        delta[i] = diff;
    }
}

float dot_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    //float dot = 0;
    float dot = ZERO;
    //for(i = 0; i < N; ++i) dot += X[i*INCX] * Y[i*INCY];
    for(i = 0; i < N; ++i) dot = add(dot, mul(X[i*INCX], Y[i*INCY]));
    return dot;
}

void softmax(float *input, int n, float temp, int stride, float *output)
{
    int i;
    //float sum = 0;
    float sum = ZERO;
    float largest = float2type(-FLT_MAX);
    for(i = 0; i < n; ++i){
        //if(input[i*stride] > largest) largest = input[i*stride];
        if(gt(input[i*stride], largest)) largest = input[i*stride];
    }
    for(i = 0; i < n; ++i){
        //float e = exp(input[i*stride]/temp - largest/temp);
        float e = exp(sub(div(input[i*stride], temp), div(largest,temp)));
        //sum += e;
        sum = add(sum, e);
        output[i*stride] = e;
    }
    for(i = 0; i < n; ++i){
        //output[i*stride] /= sum;
        output[i*stride] = div(output[i*stride], sum);
    }
}


void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
    int g, b;
    for(b = 0; b < batch; ++b){
        for(g = 0; g < groups; ++g){
            softmax(input + b*batch_offset + g*group_offset, n, temp, stride, output + b*batch_offset + g*group_offset);
        }
    }
}

void upsample_cpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out)
{
    int i, j, k, b;
    for(b = 0; b < batch; ++b){
        for(k = 0; k < c; ++k){
            for(j = 0; j < h*stride; ++j){
                for(i = 0; i < w*stride; ++i){
                    int in_index = b*w*h*c + k*w*h + (j/stride)*w + i/stride;
                    int out_index = b*w*h*c*stride*stride + k*w*h*stride*stride + j*w*stride + i;
                    //if(forward) out[out_index] = scale*in[in_index];
                    if(forward) out[out_index] = mul(scale, in[in_index]);
                    //else in[in_index] += scale*out[out_index];
                    else in[in_index] = add(in[in_index], mul(scale, out[out_index]));
                }
            }
        }
    }
}


