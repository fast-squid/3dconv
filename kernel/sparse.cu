#include "mat.h"
#include <iostream>
#include <stdlib.h>
#define ERROR_CHECK \
{\
	cudaError err = cudaGetLastError(); \
	if ( cudaSuccess != err ) \
	{\
		printf("[%s:%d]CUDA ERROR : %s\n", __FILE__, __LINE__, cudaGetErrorString(err) ); \
		exit(-1); \
	}\
}
__device__ __constant__ float eps = 0.0001;

// implementation without convolution parameter(padding, stride, groups, dilation)
__global__ void cube_to_coo(int input_d, int input_h, int input_w, float* input_data,
		int output_d, int output_h, int output_w, COO* coo, int *nnz, int stride)
{
    // block[8] : 0 ~ 7
	//printf(".\n");	
	if(threadIdx.x <5 && threadIdx.y < 5 && threadIdx.z <5)
	{
		
		int input_w_idx = (blockIdx.x % output_w)*stride + threadIdx.x;
		int input_h_idx = ((blockIdx.x / output_w) % output_h)*stride + threadIdx.y;
		int input_d_idx = (((blockIdx.x / output_w) / output_d) % output_d)*stride + threadIdx.z;

		int input_idx =  input_d_idx*(input_h*input_w)
			+ input_h_idx*(input_w)
			+ input_w_idx;
		if(input_data[input_idx]>eps)
		{
			int global_idx = atomicAdd(nnz, 1);
			coo[global_idx].row = threadIdx.z*(25)
				+ threadIdx.y*5
				+ threadIdx.x;
			coo[global_idx].col = blockIdx.x;
			coo[global_idx].val = input_data[input_idx];
		}
	}
	
}

int compare(const void* a, const void* b)
{
	COO x = *(COO*)a;
	COO y = *(COO*)b;
	if(x.row == y.row) return x.col - y.col;
	else return x.row - y.row;
}

void cube_to_coo_cuda(Mat& input, Mat& filter, Param& p)
{
	int output_N = 1;
	int output_C = filter.N;
	int output_D = 1+(input.D - filter.D + 2*p.padding)/p.stride;
	int output_H = 1+(input.H - filter.H + 2*p.padding)/p.stride;
	int output_W = 1+(input.W - filter.W + 2*p.padding)/p.stride;

	printf("input shape  : (%d,%d,%d,%d,%d)\n",input.N, input.C, input.D, input.H, input.W);
	printf("output shape : (%d,%d,%d,%d,%d)\n",output_N, output_C, output_D, output_H, output_W);

	// coo format
	int* nnz_d;

	cudaMalloc((void**)&input.coo_dev,sizeof(COO)*400000);
	cudaMalloc((void**)&nnz_d, sizeof(int));
	cudaMemset(nnz_d, 0, sizeof(int));
	
	// 3D-input
	int input_size = input.N*input.C*input.D*input.H*input.W;
	cudaMalloc((void**)&input.data_dev, sizeof(float)*input_size);
	cudaMemcpy(input.data_dev, input.data, sizeof(float)*input_size,cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	
	int block_num = output_D*output_H*output_W;
	dim3 block_size(8,8,8);
	
	cube_to_coo<<<block_num, block_size>>>
		(input.D, input.H, input.W, input.data_dev,
		 output_D, output_H, output_W,input.coo_dev,
		 nnz_d,p.stride);
	ERROR_CHECK;	

	input.row_num = block_num +1;
	cudaMemcpy(&input.nnz,nnz_d, sizeof(int), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();


	input.coo = new COO[input.nnz];
	cudaMemcpy(input.coo, input.coo_dev, input.nnz*sizeof(COO),  cudaMemcpyDeviceToHost);
	qsort(input.coo, input.nnz, sizeof(COO),compare);
	cudaMemcpy(input.coo_dev, input.coo, input.nnz*sizeof(COO),  cudaMemcpyHostToDevice);

	// cudafree
	cudaFree(input.data_dev);
}

__global__ void coo_2_csr(COO* coo, 
		int* ptr, int *idx, float* val,
		int nnz)
{
    int global_tid = threadIdx.x+blockDim.x*blockIdx.x;
	if(global_tid<nnz)
	{
		idx[global_tid] = coo[global_tid].col;
		val[global_tid] = coo[global_tid].val;
		atomicAdd(&ptr[coo[global_tid].row+1], 1);
	}
}

void coo_to_csr_cuda(Mat& input)
{
	cudaMalloc((void**)&input.ptr_dev, sizeof(int)*(input.row_num+1));
	cudaMalloc((void**)&input.idx_dev, sizeof(int)*input.nnz);
	cudaMalloc((void**)&input.val_dev, sizeof(float)*input.nnz);
	cudaMemset(input.ptr_dev, 0, sizeof(int)*(input.row_num+1));

	int block_num = input.nnz/1024+1;
	int block_size = 1024;
	
	coo_2_csr<<<block_num, block_size>>>(input.coo_dev,
			input.ptr_dev, input.idx_dev, input.val_dev,
			input.nnz);
	ERROR_CHECK;	
	
	input.ptr = new int[input.row_num+1];
	input.idx = new int[input.nnz];
	input.val = new float[input.nnz];

	cudaMemcpy(input.ptr, input.ptr_dev, sizeof(int)*(input.row_num+1),cudaMemcpyDeviceToHost);
	cudaMemcpy(input.idx, input.idx_dev, sizeof(int)*input.nnz,cudaMemcpyDeviceToHost);
	cudaMemcpy(input.val, input.val_dev, sizeof(float)*input.nnz,cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();
	for(int i=0;i<input.row_num;i++)
	{
		input.ptr[i+1] += input.ptr[i];
	}
	printf("nnz %d %d\n",input.nnz, input.ptr[input.row_num]);
	cudaMemcpy(input.ptr_dev, input.ptr, sizeof(int)*(input.row_num+1),cudaMemcpyHostToDevice);


}

