#include "model.h"

#define ERROR_CHECK \
{\
	cudaError err = cudaGetLastError(); \
	if ( cudaSuccess != err ) \
	{\
		printf("[%s:%d]CUDA ERROR : %s\n", __FILE__, __LINE__, cudaGetErrorString(err) ); \
	}\
}

void Model::set_cuda(bool transpose)
{	
	Mat* filter_ptr = 0;
	for(int i=0;i < inner_layers.size();i++)
	{
		if(inner_layers[i]->type == CONV)
		{
			filter_ptr = &((Conv*)inner_layers[i])->filter;
			int size = filter_ptr->get_mat_size();
			float* data_host;
			if(transpose)
			{
				filter_ptr->set_transpose();
				data_host = filter_ptr->data_trans;
			}
			else
			{
				data_host = filter_ptr->data;
			}
			cudaMalloc((void**)&filter_ptr->data_dev, sizeof(float)*size);
			cudaMemcpy(filter_ptr->data_dev, data_host, sizeof(float)*size, cudaMemcpyHostToDevice); 
		}
	}
}

