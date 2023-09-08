#ifndef BOX_H
#define BOX_H
#include "darknet.h"

typedef struct{
    Unum4 dx, dy, dw, dh;
} dbox;

Unum4 box_rmse(box a, box b);
dbox diou(box a, box b);
box decode_box(box b, box anchor);
box encode_box(box b, box anchor);

#endif
