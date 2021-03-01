#pragma once

// 3-Dimension representation
typedef struct _Mat{
    int N;
    int C;
    int D;
    int H;
    int W;
    float *data;
}Mat;

// convolution parameter
typedef struct _Param{
    int stride;
    int padding;
    int groups;
    int dilation;
}Param;


