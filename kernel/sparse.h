#pragma once

#include "mat.h"

void cube_to_coo_cuda(Mat& input,Mat& filter, Param& p);
void coo_to_csr_cuda(Mat &input);
