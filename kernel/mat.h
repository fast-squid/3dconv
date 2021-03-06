#pragma once

typedef struct _COO{
	int row;
	int col;
	float val;
}COO;

// 3-Dimension representation
typedef struct _Mat{
    // NCDHW format
	int N;
    int C;
    int D;
    int H;
    int W;
	int nnz;
    float *data;
	
	// coo format
	int row_num;
	int col_num;
	COO* coo;

	// csr format
	int* ptr;
	int* idx;
	float* val;

	// GPU memory
	float* data_dev;

	COO* coo_dev;
	
	int* ptr_dev;
	int* idx_dev;
	float* val_dev;
}Mat;

// convolution parameter
typedef struct _Param{
    int stride;
    int padding;
    int groups;
    int dilation;
}Param;



