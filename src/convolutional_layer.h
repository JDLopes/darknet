#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "cuda.h"
#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"

typedef layer convolutional_layer;

#ifdef GPU
void forward_convolutional_layer_gpu(convolutional_layer layer, network net);
void backward_convolutional_layer_gpu(convolutional_layer layer, network net);
void update_convolutional_layer_gpu(convolutional_layer layer, update_args a);

void push_convolutional_layer(convolutional_layer layer);
void pull_convolutional_layer(convolutional_layer layer);

void add_bias_gpu(Unum4 *output, Unum4 *biases, int batch, int n, int size);
void backward_bias_gpu(Unum4 *bias_updates, Unum4 *delta, int batch, int n, int size);
void adam_update_gpu(Unum4 *w, Unum4 *d, Unum4 *m, Unum4 *v, Unum4 B1, Unum4 B2, Unum4 eps, Unum4 decay, Unum4 rate, int n, int batch, int t);
#ifdef CUDNN
void cudnn_convolutional_setup(layer *l);
#endif
#endif

convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam);
void resize_convolutional_layer(convolutional_layer *layer, int w, int h);
void forward_convolutional_layer(const convolutional_layer layer, network net);
void update_convolutional_layer(convolutional_layer layer, update_args a);
image *visualize_convolutional_layer(convolutional_layer layer, char *window, image *prev_weights);
void binarize_weights(Unum4 *weights, int n, int size, Unum4 *binary);
void swap_binary(convolutional_layer *l);
void binarize_weights2(Unum4 *weights, int n, int size, char *binary, Unum4 *scales);

void backward_convolutional_layer(convolutional_layer layer, network net);

void add_bias(Unum4 *output, Unum4 *biases, int batch, int n, int size);
void backward_bias(Unum4 *bias_updates, Unum4 *delta, int batch, int n, int size);

image get_convolutional_image(convolutional_layer layer);
image get_convolutional_delta(convolutional_layer layer);
image get_convolutional_weight(convolutional_layer layer, int i);

int convolutional_out_height(convolutional_layer layer);
int convolutional_out_width(convolutional_layer layer);

#endif

