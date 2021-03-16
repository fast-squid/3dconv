#pragma once

#include "mat.h"

void cube_to_coo_cuda(Mat& input,Mat& filter, Param& p);
void coo_to_csr_cuda(Mat &input);
void coo_to_csc_cuda(Mat &input, int&, int&);
void dense_sparse_mm_cuda(Mat&, Mat&, Mat& ,int, int*);
void sparse_conv_cuda(Mat& input, Mat& filter, Param& p, Mat& output);


