#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H
#include "layer.h"
#include "network.h"

typedef layer softmax_layer;

void softmax_array(Unum4 *input, int n, Unum4 temp, Unum4 *output);
softmax_layer make_softmax_layer(int batch, int inputs, int groups);
void forward_softmax_layer(const softmax_layer l, network net);
void backward_softmax_layer(const softmax_layer l, network net);

#ifdef GPU
void pull_softmax_layer_output(const softmax_layer l);
void forward_softmax_layer_gpu(const softmax_layer l, network net);
void backward_softmax_layer_gpu(const softmax_layer l, network net);
#endif

#endif
