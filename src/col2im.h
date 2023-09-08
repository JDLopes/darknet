#ifndef COL2IM_H
#define COL2IM_H

void col2im_cpu(Unum4* data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, Unum4* data_im);

#ifdef GPU
void col2im_gpu(Unum4 *data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, Unum4 *data_im);
#endif
#endif
