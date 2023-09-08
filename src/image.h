#ifndef IMAGE_H
#define IMAGE_H

#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <string.h>
#include <math.h>
#include "box.h"
#include "darknet.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef OPENCV
void *open_video_stream(const char *f, int c, int w, int h, int fps);
image get_image_from_stream(void *p);
image load_image_cv(char *filename, int channels);
int show_image_cv(image im, const char* name, int ms);
#endif

Unum4 get_color(int c, int x, int max);
void draw_box(image a, int x1, int y1, int x2, int y2, Unum4 r, Unum4 g, Unum4 b);
void draw_bbox(image a, box bbox, int w, Unum4 r, Unum4 g, Unum4 b);
void write_label(image a, int r, int c, image *characters, char *string, Unum4 *rgb);
image image_distance(image a, image b);
void scale_image(image m, Unum4 s);
image rotate_crop_image(image im, Unum4 rad, Unum4 s, int w, int h, Unum4 dx, Unum4 dy, Unum4 aspect);
image random_crop_image(image im, int w, int h);
image random_augment_image(image im, Unum4 angle, Unum4 aspect, int low, int high, int w, int h);
augment_args random_augment_args(image im, Unum4 angle, Unum4 aspect, int low, int high, int w, int h);
void letterbox_image_into(image im, int w, int h, image boxed);
image resize_max(image im, int max);
void translate_image(image m, Unum4 s);
void embed_image(image source, image dest, int dx, int dy);
void place_image(image im, int w, int h, int dx, int dy, image canvas);
void saturate_image(image im, Unum4 sat);
void exposure_image(image im, Unum4 sat);
void distort_image(image im, Unum4 hue, Unum4 sat, Unum4 val);
void saturate_exposure_image(image im, Unum4 sat, Unum4 exposure);
void rgb_to_hsv(image im);
void hsv_to_rgb(image im);
void yuv_to_rgb(image im);
void rgb_to_yuv(image im);


image collapse_image_layers(image source, int border);
image collapse_images_horz(image *ims, int n);
image collapse_images_vert(image *ims, int n);

void show_image_normalized(image im, const char *name);
void show_images(image *ims, int n, char *window);
void show_image_layers(image p, char *name);
void show_image_collapsed(image p, char *name);

void print_image(image m);

image make_empty_image(int w, int h, int c);
void copy_image_into(image src, image dest);

image get_image_layer(image m, int l);

#ifdef __cplusplus
}
#endif

#endif

