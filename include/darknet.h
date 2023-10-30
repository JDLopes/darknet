#ifndef DARKNET_API
#define DARKNET_API
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>

#ifdef GPU
    #define BLOCK 512

    #include "cuda_runtime.h"
    #include "curand.h"
    #include "cublas_v2.h"

    #ifdef CUDNN
    #include "cudnn.h"
    #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SECRET_NUM -1234
extern int gpu_index;

typedef struct{
    int classes;
    char **names;
} metadata;

metadata get_metadata(char *file);

typedef struct{
    int *leaf;
    int n;
    int *parent;
    int *child;
    int *group;
    char **name;

    int groups;
    int *group_size;
    int *group_offset;
} tree;
tree *read_tree(char *filename);

typedef enum{
    LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN, SELU
} ACTIVATION;

typedef enum{
    PNG, BMP, TGA, JPG
} IMTYPE;

typedef enum{
    MULT, ADD, SUB, DIV
} BINARY_ACTIVATION;

typedef enum {
    CONVOLUTIONAL,
    DECONVOLUTIONAL,
    CONNECTED,
    MAXPOOL,
    SOFTMAX,
    DETECTION,
    DROPOUT,
    CROP,
    ROUTE,
    COST,
    NORMALIZATION,
    AVGPOOL,
    LOCAL,
    SHORTCUT,
    ACTIVE,
    RNN,
    GRU,
    LSTM,
    CRNN,
    BATCHNORM,
    NETWORK,
    XNOR,
    REGION,
    YOLO,
    ISEG,
    REORG,
    UPSAMPLE,
    LOGXENT,
    L2NORM,
    BLANK
} LAYER_TYPE;

typedef enum{
    SSE, MASKED, L1, SEG, SMOOTH,WGAN
} COST_TYPE;

typedef struct{
    int batch;
    Unum4 learning_rate;
    Unum4 momentum;
    Unum4 decay;
    int adam;
    Unum4 B1;
    Unum4 B2;
    Unum4 eps;
    int t;
} update_args;

struct network;
typedef struct network network;

struct layer;
typedef struct layer layer;

struct layer{
    LAYER_TYPE type;
    ACTIVATION activation;
    COST_TYPE cost_type;
    void (*forward)   (struct layer, struct network);
    void (*backward)  (struct layer, struct network);
    void (*update)    (struct layer, update_args);
    void (*forward_gpu)   (struct layer, struct network);
    void (*backward_gpu)  (struct layer, struct network);
    void (*update_gpu)    (struct layer, update_args);
    int batch_normalize;
    int shortcut;
    int batch;
    int forced;
    int flipped;
    int inputs;
    int outputs;
    int nweights;
    int nbiases;
    int extra;
    int truths;
    int h,w,c;
    int out_h, out_w, out_c;
    int n;
    int max_boxes;
    int groups;
    int size;
    int side;
    int stride;
    int reverse;
    int flatten;
    int spatial;
    int pad;
    int sqrt;
    int flip;
    int index;
    int binary;
    int xnor;
    int steps;
    int hidden;
    int truth;
    Unum4 smooth;
    Unum4 dot;
    Unum4 angle;
    Unum4 jitter;
    Unum4 saturation;
    Unum4 exposure;
    Unum4 shift;
    Unum4 ratio;
    Unum4 learning_rate_scale;
    Unum4 clip;
    int noloss;
    int softmax;
    int classes;
    int coords;
    int background;
    int rescore;
    int objectness;
    int joint;
    int noadjust;
    int reorg;
    int log;
    int tanh;
    int *mask;
    int total;

    Unum4 alpha;
    Unum4 beta;
    Unum4 kappa;

    Unum4 coord_scale;
    Unum4 object_scale;
    Unum4 noobject_scale;
    Unum4 mask_scale;
    Unum4 class_scale;
    int bias_match;
    int random;
    Unum4 ignore_thresh;
    Unum4 truth_thresh;
    Unum4 thresh;
    Unum4 focus;
    int classfix;
    int absolute;

    int onlyforward;
    int stopbackward;
    int dontload;
    int dontsave;
    int dontloadscales;
    int numload;

    Unum4 temperature;
    Unum4 probability;
    Unum4 scale;

    char  * cweights;
    int   * indexes;
    int   * input_layers;
    int   * input_sizes;
    int   * map;
    int   * counts;
    Unum4 ** sums;
    Unum4 * rand;
    Unum4 * cost;
    Unum4 * state;
    Unum4 * prev_state;
    Unum4 * forgot_state;
    Unum4 * forgot_delta;
    Unum4 * state_delta;
    Unum4 * combine_cpu;
    Unum4 * combine_delta_cpu;

    Unum4 * concat;
    Unum4 * concat_delta;

    Unum4 * binary_weights;

    Unum4 * biases;
    Unum4 * bias_updates;

    Unum4 * scales;
    Unum4 * scale_updates;

    Unum4 * weights;
    Unum4 * weight_updates;

    Unum4 * delta;
    Unum4 * output;
    Unum4 * loss;
    Unum4 * squared;
    Unum4 * norms;

    Unum4 * spatial_mean;
    Unum4 * mean;
    Unum4 * variance;

    Unum4 * mean_delta;
    Unum4 * variance_delta;

    Unum4 * rolling_mean;
    Unum4 * rolling_variance;

    Unum4 * x;
    Unum4 * x_norm;

    Unum4 * m;
    Unum4 * v;
    
    Unum4 * bias_m;
    Unum4 * bias_v;
    Unum4 * scale_m;
    Unum4 * scale_v;


    Unum4 *z_cpu;
    Unum4 *r_cpu;
    Unum4 *h_cpu;
    Unum4 * prev_state_cpu;

    Unum4 *temp_cpu;
    Unum4 *temp2_cpu;
    Unum4 *temp3_cpu;

    Unum4 *dh_cpu;
    Unum4 *hh_cpu;
    Unum4 *prev_cell_cpu;
    Unum4 *cell_cpu;
    Unum4 *f_cpu;
    Unum4 *i_cpu;
    Unum4 *g_cpu;
    Unum4 *o_cpu;
    Unum4 *c_cpu;
    Unum4 *dc_cpu; 

    Unum4 * binary_input;

    struct layer *input_layer;
    struct layer *self_layer;
    struct layer *output_layer;

    struct layer *reset_layer;
    struct layer *update_layer;
    struct layer *state_layer;

    struct layer *input_gate_layer;
    struct layer *state_gate_layer;
    struct layer *input_save_layer;
    struct layer *state_save_layer;
    struct layer *input_state_layer;
    struct layer *state_state_layer;

    struct layer *input_z_layer;
    struct layer *state_z_layer;

    struct layer *input_r_layer;
    struct layer *state_r_layer;

    struct layer *input_h_layer;
    struct layer *state_h_layer;
	
    struct layer *wz;
    struct layer *uz;
    struct layer *wr;
    struct layer *ur;
    struct layer *wh;
    struct layer *uh;
    struct layer *uo;
    struct layer *wo;
    struct layer *uf;
    struct layer *wf;
    struct layer *ui;
    struct layer *wi;
    struct layer *ug;
    struct layer *wg;

    tree *softmax_tree;

    size_t workspace_size;

#ifdef GPU
    int *indexes_gpu;

    Unum4 *z_gpu;
    Unum4 *r_gpu;
    Unum4 *h_gpu;

    Unum4 *temp_gpu;
    Unum4 *temp2_gpu;
    Unum4 *temp3_gpu;

    Unum4 *dh_gpu;
    Unum4 *hh_gpu;
    Unum4 *prev_cell_gpu;
    Unum4 *cell_gpu;
    Unum4 *f_gpu;
    Unum4 *i_gpu;
    Unum4 *g_gpu;
    Unum4 *o_gpu;
    Unum4 *c_gpu;
    Unum4 *dc_gpu; 

    Unum4 *m_gpu;
    Unum4 *v_gpu;
    Unum4 *bias_m_gpu;
    Unum4 *scale_m_gpu;
    Unum4 *bias_v_gpu;
    Unum4 *scale_v_gpu;

    Unum4 * combine_gpu;
    Unum4 * combine_delta_gpu;

    Unum4 * prev_state_gpu;
    Unum4 * forgot_state_gpu;
    Unum4 * forgot_delta_gpu;
    Unum4 * state_gpu;
    Unum4 * state_delta_gpu;
    Unum4 * gate_gpu;
    Unum4 * gate_delta_gpu;
    Unum4 * save_gpu;
    Unum4 * save_delta_gpu;
    Unum4 * concat_gpu;
    Unum4 * concat_delta_gpu;

    Unum4 * binary_input_gpu;
    Unum4 * binary_weights_gpu;

    Unum4 * mean_gpu;
    Unum4 * variance_gpu;

    Unum4 * rolling_mean_gpu;
    Unum4 * rolling_variance_gpu;

    Unum4 * variance_delta_gpu;
    Unum4 * mean_delta_gpu;

    Unum4 * x_gpu;
    Unum4 * x_norm_gpu;
    Unum4 * weights_gpu;
    Unum4 * weight_updates_gpu;
    Unum4 * weight_change_gpu;

    Unum4 * biases_gpu;
    Unum4 * bias_updates_gpu;
    Unum4 * bias_change_gpu;

    Unum4 * scales_gpu;
    Unum4 * scale_updates_gpu;
    Unum4 * scale_change_gpu;

    Unum4 * output_gpu;
    Unum4 * loss_gpu;
    Unum4 * delta_gpu;
    Unum4 * rand_gpu;
    Unum4 * squared_gpu;
    Unum4 * norms_gpu;
#ifdef CUDNN
    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
    cudnnTensorDescriptor_t dsrcTensorDesc, ddstTensorDesc;
    cudnnTensorDescriptor_t normTensorDesc;
    cudnnFilterDescriptor_t weightDesc;
    cudnnFilterDescriptor_t dweightDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionFwdAlgo_t fw_algo;
    cudnnConvolutionBwdDataAlgo_t bd_algo;
    cudnnConvolutionBwdFilterAlgo_t bf_algo;
#endif
#endif
};

inline void init_layer(layer *l) {
  l->smooth = 0;
  l->dot = 0;
  l->angle = 0;
  l->jitter = 0;
  l->saturation = 0;
  l->exposure = 0;
  l->shift = 0;
  l->ratio = 0;
  l->learning_rate_scale = 0;
  l->clip = 0;
  l->alpha = 0;
  l->beta = 0;
  l->kappa = 0;
  l->coord_scale = 0;
  l->object_scale = 0;
  l->noobject_scale = 0;
  l->mask_scale = 0;
  l->class_scale = 0;
  l->ignore_thresh = 0;
  l->truth_thresh = 0;
  l->thresh = 0;
  l->focus = 0;
  l->temperature = 0;
  l->probability = 0;
  l->scale = 0;
  return;
}

void free_layer(layer);

typedef enum {
    CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM
} learning_rate_policy;

typedef struct network{
    int n;
    int batch;
    size_t *seen;
    int *t;
    Unum4 epoch;
    int subdivisions;
    layer *layers;
    Unum4 *output;
    learning_rate_policy policy;

    Unum4 learning_rate;
    Unum4 momentum;
    Unum4 decay;
    Unum4 gamma;
    Unum4 scale;
    Unum4 power;
    int time_steps;
    int step;
    int max_batches;
    Unum4 *scales;
    int   *steps;
    int num_steps;
    int burn_in;

    int adam;
    Unum4 B1;
    Unum4 B2;
    Unum4 eps;

    int inputs;
    int outputs;
    int truths;
    int notruth;
    int h, w, c;
    int max_crop;
    int min_crop;
    Unum4 max_ratio;
    Unum4 min_ratio;
    int center;
    Unum4 angle;
    Unum4 aspect;
    Unum4 exposure;
    Unum4 saturation;
    Unum4 hue;
    int random;

    int gpu_index;
    tree *hierarchy;

    Unum4 *input;
    Unum4 *truth;
    Unum4 *delta;
    Unum4 *workspace;
    int train;
    int index;
    Unum4 *cost;
    Unum4 clip;

#ifdef GPU
    Unum4 *input_gpu;
    Unum4 *truth_gpu;
    Unum4 *delta_gpu;
    Unum4 *output_gpu;
#endif

} network;

inline void init_network(network *net) {
  net->epoch = 0;
  net->learning_rate = 0;
  net->momentum = 0;
  net->decay = 0;
  net->gamma = 0;
  net->scale = 0;
  net->power = 0;
  net->B1 = 0;
  net->B2 = 0;
  net->eps = 0;
  net->max_ratio = 0;
  net->min_ratio = 0;
  net->angle = 0;
  net->aspect = 0;
  net->exposure = 0;
  net->saturation = 0;
  net->hue = 0;
  net->clip = 0;
  return;
}

typedef struct {
    int w;
    int h;
    Unum4 scale;
    Unum4 rad;
    Unum4 dx;
    Unum4 dy;
    Unum4 aspect;
} augment_args;

typedef struct {
    int w;
    int h;
    int c;
    Unum4 *data;
} image;

typedef struct{
    Unum4 x, y, w, h;
} box;

inline void init_box(box *b) {
  b->x = 0;
  b->y = 0;
  b->w = 0;
  b->h = 0;
  return;
}

typedef struct detection{
    box bbox;
    int classes;
    Unum4 *prob;
    Unum4 *mask;
    Unum4 objectness;
    int sort_class;
} detection;

inline void init_detection(detection *d) {
  d->objectness = 0;
  init_box(&d->bbox);
  return;
}

typedef struct matrix{
    int rows, cols;
    Unum4 **vals;
} matrix;


typedef struct{
    int w, h;
    matrix X;
    matrix y;
    int shallow;
    int *num_boxes;
    box **boxes;
} data;

typedef enum {
    CLASSIFICATION_DATA, DETECTION_DATA, CAPTCHA_DATA, REGION_DATA, IMAGE_DATA, COMPARE_DATA, WRITING_DATA, SWAG_DATA, TAG_DATA, OLD_CLASSIFICATION_DATA, STUDY_DATA, DET_DATA, SUPER_DATA, LETTERBOX_DATA, REGRESSION_DATA, SEGMENTATION_DATA, INSTANCE_DATA, ISEG_DATA
} data_type;

typedef struct load_args{
    int threads;
    char **paths;
    char *path;
    int n;
    int m;
    char **labels;
    int h;
    int w;
    int out_w;
    int out_h;
    int nh;
    int nw;
    int num_boxes;
    int min, max, size;
    int classes;
    int background;
    int scale;
    int center;
    int coords;
    Unum4 jitter;
    Unum4 angle;
    Unum4 aspect;
    Unum4 saturation;
    Unum4 exposure;
    Unum4 hue;
    data *d;
    image *im;
    image *resized;
    data_type type;
    tree *hierarchy;
} load_args;

inline void init_load_args(load_args *args) {
  args->jitter = 0;
  args->angle = 0;
  args->aspect = 0;
  args->saturation = 0;
  args->exposure = 0;
  args->hue = 0;
  return;
}

typedef struct{
    int id;
    Unum4 x,y,w,h;
    Unum4 left, right, top, bottom;
} box_label;

inline void init_box_label(box_label *b) {
  b->x = 0;
  b->y = 0;
  b->w = 0;
  b->h = 0;
  b->left = 0;
  b->right = 0;
  b->top = 0;
  b->bottom = 0;
  return;
}


network *load_network(char *cfg, char *weights, int clear);
load_args get_base_args(network *net);

void free_data(data d);

typedef struct node{
    void *val;
    struct node *next;
    struct node *prev;
} node;

typedef struct list{
    int size;
    node *front;
    node *back;
} list;

pthread_t load_data(load_args args);
list *read_data_cfg(char *filename);
list *read_cfg(char *filename);
unsigned char *read_file(char *filename);
data resize_data(data orig, int w, int h);
data *tile_data(data orig, int divs, int size);
data select_data(data *orig, int *inds);

void forward_network(network *net);
void backward_network(network *net);
void update_network(network *net);


Unum4 dot_cpu(int N, Unum4 *X, int INCX, Unum4 *Y, int INCY);
void axpy_cpu(int N, Unum4 ALPHA, Unum4 *X, int INCX, Unum4 *Y, int INCY);
void copy_cpu(int N, Unum4 *X, int INCX, Unum4 *Y, int INCY);
void scal_cpu(int N, Unum4 ALPHA, Unum4 *X, int INCX);
void fill_cpu(int N, Unum4 ALPHA, Unum4 * X, int INCX);
void normalize_cpu(Unum4 *x, Unum4 *mean, Unum4 *variance, int batch, int filters, int spatial);
void softmax(Unum4 *input, int n, Unum4 temp, int stride, Unum4 *output);

int best_3d_shift_r(image a, image b, int min, int max);
#ifdef GPU
void axpy_gpu(int N, Unum4 ALPHA, Unum4 * X, int INCX, Unum4 * Y, int INCY);
void fill_gpu(int N, Unum4 ALPHA, Unum4 * X, int INCX);
void scal_gpu(int N, Unum4 ALPHA, Unum4 * X, int INCX);
void copy_gpu(int N, Unum4 * X, int INCX, Unum4 * Y, int INCY);

void cuda_set_device(int n);
void cuda_free(Unum4 *x_gpu);
Unum4 *cuda_make_array(Unum4 *x, size_t n);
void cuda_pull_array(Unum4 *x_gpu, Unum4 *x, size_t n);
Unum4 cuda_mag_array(Unum4 *x_gpu, size_t n);
void cuda_push_array(Unum4 *x_gpu, Unum4 *x, size_t n);

void forward_network_gpu(network *net);
void backward_network_gpu(network *net);
void update_network_gpu(network *net);

Unum4 train_networks(network **nets, int n, data d, int interval);
void sync_nets(network **nets, int n, int interval);
void harmless_update_network_gpu(network *net);
#endif
image get_label(image **characters, char *string, int size);
void draw_label(image a, int r, int c, image label, const Unum4 *rgb);
void save_image(image im, const char *name);
void save_image_options(image im, const char *name, IMTYPE f, int quality);
void get_next_batch(data d, int n, int offset, Unum4 *X, Unum4 *y);
void grayscale_image_3c(image im);
void normalize_image(image p);
void matrix_to_csv(matrix m);
Unum4 train_network_sgd(network *net, data d, int n);
void rgbgr_image(image im);
data copy_data(data d);
data concat_data(data d1, data d2);
data load_cifar10_data(char *filename);
Unum4 matrix_topk_accuracy(matrix truth, matrix guess, int k);
void matrix_add_matrix(matrix from, matrix to);
void scale_matrix(matrix m, Unum4 scale);
matrix csv_to_matrix(char *filename);
Unum4 *network_accuracies(network *net, data d, int n);
Unum4 train_network_datum(network *net);
image make_random_image(int w, int h, int c);

void denormalize_connected_layer(layer l);
void denormalize_convolutional_layer(layer l);
void statistics_connected_layer(layer l);
void rescale_weights(layer l, Unum4 scale, Unum4 trans);
void rgbgr_weights(layer l);
image *get_weights(layer l);

void demo(char *cfgfile, char *weightfile, Unum4 thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix, int avg, Unum4 hier_thresh, int w, int h, int fps, int fullscreen);
void get_detection_detections(layer l, int w, int h, Unum4 thresh, detection *dets);

char *option_find_str(list *l, char *key, char *def);
int option_find_int(list *l, char *key, int def);
int option_find_int_quiet(list *l, char *key, int def);

network *parse_network_cfg(char *filename);
void save_weights(network *net, char *filename);
void load_weights(network *net, char *filename);
void save_weights_upto(network *net, char *filename, int cutoff);
void load_weights_upto(network *net, char *filename, int start, int cutoff);

void zero_objectness(layer l);
void get_region_detections(layer l, int w, int h, int netw, int neth, Unum4 thresh, int *map, Unum4 tree_thresh, int relative, detection *dets);
int get_yolo_detections(layer l, int w, int h, int netw, int neth, Unum4 thresh, int *map, int relative, detection *dets);
void free_network(network *net);
void set_batch_network(network *net, int b);
void set_temp_network(network *net, Unum4 t);
image load_image(char *filename, int w, int h, int c);
image load_image_color(char *filename, int w, int h);
image make_image(int w, int h, int c);
image resize_image(image im, int w, int h);
void censor_image(image im, int dx, int dy, int w, int h);
image letterbox_image(image im, int w, int h);
image crop_image(image im, int dx, int dy, int w, int h);
image center_crop_image(image im, int w, int h);
image resize_min(image im, int min);
image resize_max(image im, int max);
image threshold_image(image im, Unum4 thresh);
image mask_to_rgb(image mask);
int resize_network(network *net, int w, int h);
void free_matrix(matrix m);
void test_resize(char *filename);
int show_image(image p, const char *name, int ms);
image copy_image(image p);
void draw_box_width(image a, int x1, int y1, int x2, int y2, int w, Unum4 r, Unum4 g, Unum4 b);
Unum4 get_current_rate(network *net);
void composite_3d(char *f1, char *f2, char *out, int delta);
data load_data_old(char **paths, int n, int m, char **labels, int k, int w, int h);
size_t get_current_batch(network *net);
void constrain_image(image im);
image get_network_image_layer(network *net, int i);
layer get_network_output_layer(network *net);
void top_predictions(network *net, int n, int *index);
void flip_image(image a);
image Unum4_to_image(int w, int h, int c, Unum4 *data);
void ghost_image(image source, image dest, int dx, int dy);
Unum4 network_accuracy(network *net, data d);
void random_distort_image(image im, Unum4 hue, Unum4 saturation, Unum4 exposure);
void fill_image(image m, Unum4 s);
image grayscale_image(image im);
void rotate_image_cw(image im, int times);
Unum4 what_time_is_it_now();
image rotate_image(image m, Unum4 rad);
void visualize_network(network *net);
Unum4 box_iou(box a, box b);
data load_all_cifar10();
box_label *read_boxes(char *filename, int *n);
box Unum4_to_box(Unum4 *f, int stride);
void draw_detections(image im, detection *dets, int num, Unum4 thresh, char **names, image **alphabet, int classes);

matrix network_predict_data(network *net, data test);
image **load_alphabet();
image get_network_image(network *net);
Unum4 *network_predict(network *net, Unum4 *input);

int network_width(network *net);
int network_height(network *net);
Unum4 *network_predict_image(network *net, image im);
void network_detect(network *net, image im, Unum4 thresh, Unum4 hier_thresh, Unum4 nms, detection *dets);
detection *get_network_boxes(network *net, int w, int h, Unum4 thresh, Unum4 hier, int *map, int relative, int *num);
void free_detections(detection *dets, int n);

void reset_network_state(network *net, int b);

char **get_labels(char *filename);
void do_nms_obj(detection *dets, int total, int classes, Unum4 thresh);
void do_nms_sort(detection *dets, int total, int classes, Unum4 thresh);

matrix make_matrix(int rows, int cols);

#ifdef OPENCV
void *open_video_stream(const char *f, int c, int w, int h, int fps);
image get_image_from_stream(void *p);
void make_window(char *name, int w, int h, int fullscreen);
#endif

void free_image(image m);
Unum4 train_network(network *net, data d);
pthread_t load_data_in_thread(load_args args);
void load_data_blocking(load_args args);
list *get_paths(char *filename);
void hierarchy_predictions(Unum4 *predictions, int n, tree *hier, int only_leaves, int stride);
void change_leaves(tree *t, char *leaf_list);

int find_int_arg(int argc, char **argv, char *arg, int def);
Unum4 find_Unum4_arg(int argc, char **argv, char *arg, Unum4 def);
int find_arg(int argc, char* argv[], char *arg);
char *find_char_arg(int argc, char **argv, char *arg, char *def);
char *basecfg(char *cfgfile);
void find_replace(char *str, char *orig, char *rep, char *output);
void free_ptrs(void **ptrs, int n);
char *fgetl(FILE *fp);
void strip(char *s);
Unum4 sec(clock_t clocks);
void **list_to_array(list *l);
void top_k(Unum4 *a, int n, int k, int *index);
int *read_map(char *filename);
void error(const char *s);
int max_index(Unum4 *a, int n);
int max_int_index(int *a, int n);
int sample_array(Unum4 *a, int n);
int *random_index_order(int min, int max);
void free_list(list *l);
Unum4 mse_array(Unum4 *a, int n);
Unum4 variance_array(Unum4 *a, int n);
Unum4 mag_array(Unum4 *a, int n);
void scale_array(Unum4 *a, int n, Unum4 s);
Unum4 mean_array(Unum4 *a, int n);
Unum4 sum_array(Unum4 *a, int n);
void normalize_array(Unum4 *a, int n);
int *read_intlist(char *s, int *n, int d);
size_t rand_size_t();
Unum4 rand_normal();
Unum4 rand_uniform(Unum4 min, Unum4 max);

#ifdef __cplusplus
}
#endif
#endif
