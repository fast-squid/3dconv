#include "model.h"


void Model::set_cuda()
{	
	Mat temp(1,1,1,1,1);
	forward(temp);
	/*
	set_transpose(filter);
	int size = filter.n*filter.c*filter.d*filter.h*filter.w;
	cudaMalloc((void**)&filter.data_dev, sizeof(float)*size);
	cudaMemcpy(filter.data_dev, filter.data_trans, sizeof(float)*size, cudaMemcpyHostToDevice);
	printf("filter size %d\n",size);
	*/
}

