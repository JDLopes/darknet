#include "detection_layer.h"
#include "activations.h"
#include "softmax_layer.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"
#include "format.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

detection_layer make_detection_layer(int batch, int inputs, int n, int side, int classes, int coords, int rescore)
{
    detection_layer l = {0};
    l.type = DETECTION;

    l.n = n;
    l.batch = batch;
    l.inputs = inputs;
    l.classes = classes;
    l.coords = coords;
    l.rescore = rescore;
    l.side = side;
    l.w = side;
    l.h = side;
    assert(side*side*((1 + l.coords)*l.n + l.classes) == inputs);
    l.cost = calloc(1, sizeof(float));
    l.outputs = l.inputs;
    l.truths = l.side*l.side*(1+l.coords+l.classes);
    l.output = calloc(batch*l.outputs, sizeof(float));
    l.delta = calloc(batch*l.outputs, sizeof(float));

    l.forward = forward_detection_layer;
    l.backward = backward_detection_layer;
#ifdef GPU
    l.forward_gpu = forward_detection_layer_gpu;
    l.backward_gpu = backward_detection_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "Detection Layer\n");
    srand(0);

    return l;
}

void forward_detection_layer(const detection_layer l, network net)
{
    int locations = l.side*l.side;
    int i,j;
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));
    //if(l.reorg) reorg(l.output, l.w*l.h, size*l.n, l.batch, 1);
    int b;
    if (l.softmax){
        for(b = 0; b < l.batch; ++b){
            int index = b*l.inputs;
            for (i = 0; i < locations; ++i) {
                int offset = i*l.classes;
                softmax(l.output + index + offset, l.classes, 1, 1,
                        l.output + index + offset);
            }
        }
    }
    if(net.train){
        //float avg_iou = 0;
        float avg_iou = ZERO;
        //float avg_cat = 0;
        float avg_cat = ZERO;
        //float avg_allcat = 0;
        float avg_allcat = ZERO;
        //float avg_obj = 0;
        float avg_obj = ZERO;
        //float avg_anyobj = 0;
        float avg_anyobj = ZERO;
        int count = 0;
        //*(l.cost) = 0;
        *(l.cost) = ZERO;
        int size = l.inputs * l.batch;
        memset(l.delta, 0, size * sizeof(float));
        for (b = 0; b < l.batch; ++b){
            int index = b*l.inputs;
            for (i = 0; i < locations; ++i) {
                int truth_index = (b*locations + i)*(1+l.coords+l.classes);
                int is_obj = net.truth[truth_index];
                for (j = 0; j < l.n; ++j) {
                    int p_index = index + locations*l.classes + i*l.n + j;
                    //l.delta[p_index] = l.noobject_scale*(0 - l.output[p_index]);
                    l.delta[p_index] = mul(l.noobject_scale, sub(ZERO, l.output[p_index]));
                    //*(l.cost) += l.noobject_scale*pow(l.output[p_index], 2);
                    *(l.cost) = add(*(l.cost), mul(l.noobject_scale, pow(l.output[p_index], TWO)));
                    //avg_anyobj += l.output[p_index];
                    avg_anyobj = add(avg_anyobj, l.output[p_index]);
                }

                int best_index = -1;
                //float best_iou = 0;
                float best_iou = ZERO;
                //float best_rmse = 20;
                float best_rmse = float2type(20);

                if (!is_obj){
                    continue;
                }

                int class_index = index + i*l.classes;
                for(j = 0; j < l.classes; ++j) {
                    //l.delta[class_index+j] = l.class_scale * (net.truth[truth_index+1+j] - l.output[class_index+j]);
                    l.delta[class_index+j] = mul(l.class_scale, sub(net.truth[truth_index+1+j], l.output[class_index+j]));
                    //*(l.cost) += l.class_scale * pow(net.truth[truth_index+1+j] - l.output[class_index+j], 2);
                    *(l.cost) = add(*(l.cost), mul(l.class_scale, pow(sub(net.truth[truth_index+1+j], l.output[class_index+j]), TWO)));
                    //if(net.truth[truth_index + 1 + j]) avg_cat += l.output[class_index+j];
                    if(!eq(net.truth[truth_index + 1 + j], ZERO)) avg_cat = add(avg_cat, l.output[class_index+j]);
                    //avg_allcat += l.output[class_index+j];
                    avg_allcat = add(avg_allcat, l.output[class_index+j]);
                }

                box truth = float_to_box(net.truth + truth_index + 1 + l.classes, 1);
                truth.x /= l.side;
                truth.y /= l.side;

                for(j = 0; j < l.n; ++j){
                    int box_index = index + locations*(l.classes + l.n) + (i*l.n + j) * l.coords;
                    box out = float_to_box(l.output + box_index, 1);
                    out.x /= l.side;
                    out.y /= l.side;

                    if (l.sqrt){
                        out.w = out.w*out.w;
                        out.h = out.h*out.h;
                    }

                    float iou  = box_iou(out, truth);
                    //iou = 0;
                    float rmse = box_rmse(out, truth);
                    //if(best_iou > 0 || iou > 0){
                    if(gt(best_iou, ZERO) || gt(iou, ZERO)){
                        //if(iou > best_iou){
                        if(gt(iou, best_iou)){
                            best_iou = iou;
                            best_index = j;
                        }
                    }else{
                        //if(rmse < best_rmse){
                      if(lt(rmse, best_rmse)){
                            best_rmse = rmse;
                            best_index = j;
                        }
                    }
                }

                if(l.forced){
                    if(truth.w*truth.h < .1){
                        best_index = 1;
                    }else{
                        best_index = 0;
                    }
                }
                if(l.random && *(net.seen) < 64000){
                    best_index = rand()%l.n;
                }

                int box_index = index + locations*(l.classes + l.n) + (i*l.n + best_index) * l.coords;
                int tbox_index = truth_index + 1 + l.classes;

                box out = float_to_box(l.output + box_index, 1);
                out.x /= l.side;
                out.y /= l.side;
                if (l.sqrt) {
                    out.w = out.w*out.w;
                    out.h = out.h*out.h;
                }
                float iou  = box_iou(out, truth);

                //printf("%d,", best_index);
                int p_index = index + locations*l.classes + i*l.n + best_index;
                //*(l.cost) -= l.noobject_scale * pow(l.output[p_index], 2);
                *(l.cost) = sub(*(l.cost), mul(l.noobject_scale, pow(l.output[p_index], TWO)));
                //*(l.cost) += l.object_scale * pow(1-l.output[p_index], 2);
                *(l.cost) = add(*(l.cost), mul(l.object_scale, pow(sub(ONE, l.output[p_index]), TWO)));
                //avg_obj += l.output[p_index];
                avg_obj = add(avg_obj, l.output[p_index]);
                //l.delta[p_index] = l.object_scale * (1.-l.output[p_index]);
                l.delta[p_index] = mul(l.object_scale, sub(ONE, l.output[p_index]));

                if(l.rescore){
                    //l.delta[p_index] = l.object_scale * (iou - l.output[p_index]);
                    l.delta[p_index] = mul(l.object_scale, sub(iou, l.output[p_index]));
                }

                //l.delta[box_index+0] = l.coord_scale*(net.truth[tbox_index + 0] - l.output[box_index + 0]);
                l.delta[box_index+0] = mul(l.coord_scale, sub(net.truth[tbox_index + 0], l.output[box_index + 0]));
                //l.delta[box_index+1] = l.coord_scale*(net.truth[tbox_index + 1] - l.output[box_index + 1]);
                l.delta[box_index+1] = mul(l.coord_scale, sub(net.truth[tbox_index + 1], l.output[box_index + 1]));
                //l.delta[box_index+2] = l.coord_scale*(net.truth[tbox_index + 2] - l.output[box_index + 2]);
                l.delta[box_index+2] = mul(l.coord_scale, sub(net.truth[tbox_index + 2], l.output[box_index + 2]));
                //l.delta[box_index+3] = l.coord_scale*(net.truth[tbox_index + 3] - l.output[box_index + 3]);
                l.delta[box_index+3] = mul(l.coord_scale, sub(net.truth[tbox_index + 3], l.output[box_index + 3]));
                if(l.sqrt){
                    //l.delta[box_index+2] = l.coord_scale*(sqrt(net.truth[tbox_index + 2]) - l.output[box_index + 2]);
                    l.delta[box_index+2] = mul(l.coord_scale, sub(sqrt(net.truth[tbox_index + 2]), l.output[box_index + 2]));
                    //l.delta[box_index+3] = l.coord_scale*(sqrt(net.truth[tbox_index + 3]) - l.output[box_index + 3]);
                    l.delta[box_index+3] = mul(l.coord_scale, sub(sqrt(net.truth[tbox_index + 3]), l.output[box_index + 3]));
                }

                //*(l.cost) += pow(1-iou, 2);
                *(l.cost) = add(*(l.cost), pow(sub(ONE, iou), TWO));
                //avg_iou += iou;
                avg_iou = add(avg_iou, iou);
                ++count;
            }
        }

        if(0){
            float *costs = calloc(l.batch*locations*l.n, sizeof(float));
            for (b = 0; b < l.batch; ++b) {
                int index = b*l.inputs;
                for (i = 0; i < locations; ++i) {
                    for (j = 0; j < l.n; ++j) {
                        int p_index = index + locations*l.classes + i*l.n + j;
                        //costs[b*locations*l.n + i*l.n + j] = l.delta[p_index]*l.delta[p_index];
                        costs[b*locations*l.n + i*l.n + j] = mul(l.delta[p_index], l.delta[p_index]);
                    }
                }
            }
            int indexes[100];
            top_k(costs, l.batch*locations*l.n, 100, indexes);
            float cutoff = costs[indexes[99]];
            for (b = 0; b < l.batch; ++b) {
                int index = b*l.inputs;
                for (i = 0; i < locations; ++i) {
                    for (j = 0; j < l.n; ++j) {
                        int p_index = index + locations*l.classes + i*l.n + j;
                        //if (l.delta[p_index]*l.delta[p_index] < cutoff) l.delta[p_index] = 0;
                        if (lt(mul(l.delta[p_index], l.delta[p_index]), cutoff)) l.delta[p_index] = ZERO;
                    }
                }
            }
            free(costs);
        }


        //*(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
        *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), TWO);


        //printf("Detection Avg IOU: %f, Pos Cat: %f, All Cat: %f, Pos Obj: %f, Any Obj: %f, count: %d\n", avg_iou/count, avg_cat/count, avg_allcat/(count*l.classes), avg_obj/count, avg_anyobj/(l.batch*locations*l.n), count);
        printf("Detection Avg IOU: %f, Pos Cat: %f, All Cat: %f, Pos Obj: %f, Any Obj: %f, count: %d\n", type2float(div(avg_iou, float2type(count))), type2float(div(avg_cat, float2type(count))), type2float(div(avg_allcat, float2type(count*l.classes))), type2float(div(avg_obj, float2type(count))), type2float(div(avg_anyobj, float2type(l.batch*locations*l.n))), count);
        //if(l.reorg) reorg(l.delta, l.w*l.h, size*l.n, l.batch, 0);
    }
}

void backward_detection_layer(const detection_layer l, network net)
{
    axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
}

void get_detection_detections(layer l, int w, int h, float thresh, detection *dets)
{
    int i,j,n;
    float *predictions = l.output;
    //int per_cell = 5*num+classes;
    for (i = 0; i < l.side*l.side; ++i){
        int row = i / l.side;
        int col = i % l.side;
        for(n = 0; n < l.n; ++n){
            int index = i*l.n + n;
            int p_index = l.side*l.side*l.classes + i*l.n + n;
            float scale = predictions[p_index];
            int box_index = l.side*l.side*(l.classes + l.n) + (i*l.n + n)*4;
            box b;
            //b.x = (predictions[box_index + 0] + col) / l.side * w;
            b.x = mul(div(add(predictions[box_index + 0], float2type(col)), float2type(l.side)), float2type(w));
            //b.y = (predictions[box_index + 1] + row) / l.side * h;
            b.y = mul(div(add(predictions[box_index + 1], float2type(row)), float2type(l.side)), float2type(h));
            //b.w = pow(predictions[box_index + 2], (l.sqrt?2:1)) * w;
            b.w = mul(pow(predictions[box_index + 2], float2type((l.sqrt?2:1))), w);
            //b.h = pow(predictions[box_index + 3], (l.sqrt?2:1)) * h;
            b.h = mul(pow(predictions[box_index + 3], float2type((l.sqrt?2:1))), h);
            dets[index].bbox = b;
            dets[index].objectness = scale;
            for(j = 0; j < l.classes; ++j){
                int class_index = i*l.classes;
                //float prob = scale*predictions[class_index+j];
                float prob = mul(scale, predictions[class_index+j]);
                //dets[index].prob[j] = (prob > thresh) ? prob : 0;
                dets[index].prob[j] = gt(prob, thresh) ? prob : ZERO;
            }
        }
    }
}

#ifdef GPU

void forward_detection_layer_gpu(const detection_layer l, network net)
{
    if(!net.train){
        copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
        return;
    }

    cuda_pull_array(net.input_gpu, net.input, l.batch*l.inputs);
    forward_detection_layer(l, net);
    cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.inputs);
}

void backward_detection_layer_gpu(detection_layer l, network net)
{
    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
    //copy_gpu(l.batch*l.inputs, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif

