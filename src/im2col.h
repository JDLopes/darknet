#ifndef IM2COL_H
#define IM2COL_H

void im2col_cpu(Unum4* data_im,
        int channels, int height, int width,
        int ksize, int stride, int pad, Unum4* data_col);

#ifdef GPU

void im2col_gpu(Unum4 *im,
         int channels, int height, int width,
         int ksize, int stride, int pad,Unum4 *data_col);

#endif
#endif
