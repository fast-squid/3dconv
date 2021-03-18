#pragma once
#include <iostream>
typedef struct _COO{
	int row;
	int col;
	float val;
}COO;

// 3-Dimension representation
class Mat{
public:
    // NCDHW format
	int n;
    int c;
    int d;
    int h;
    int w;
	int nnz;
    float *data;
	float *data_trans;	
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
public:
	int get_mat_size()
	{
		return n*c*d*h*w;
	}
	void set_matrix(int _n, int _c, int _d, int _h, int _w)
	{
		n = _n; c = _c; d = _d; h = _h; w = _w;
		data = new float[get_mat_size()];
	}
	void set_matrix(const int (&shape)[5])
	{
		n = shape[0];
		c = shape[1];
		d = shape[2];
		h = shape[3];
		w = shape[4];
		data = new float[get_mat_size()];
	}
	Mat(){}
	Mat(int _n, int _c, int _d, int _h, int _w)
		: n(_n), c(_c), d(_d), h(_h), w(_w)
	{
		data = new float[get_mat_size()];
	}
};

// convolution parameter
class Param
{
public:
    int stride;
    int padding;
    int groups;
	int dilation;
	Param(){}

	void set_parameter(int _stride, int _padding, int _groups, int _dilation)
	{
		stride = _stride;
		padding = _padding;
		groups = _groups;
		dilation = _dilation;
	}
	void set_parameter(const int (&shape)[4])
	{
		stride   = shape[0];
		padding  = shape[1];
		groups   = shape[2];
		dilation = shape[3];  
	}

};

void print_mat(const Mat&);
void print_coo(const Mat&);
void set_rand(Mat&);
void set_transpose(Mat&);
float* transpose(const Mat&);

