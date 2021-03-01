#include "mat.h"
#include <iostream>
__device__ __constant__ float eps = 0.00001;

__global__ void count_row_nnz(int height, int width, float *dense_data,
        int* row_ptr, int* col_idx, float* sparse_data)
{
    int global_tid = threadIdx.x + blockDim.x*blockIdx.x;

    if(global_tid < height * width)
    {
        if(dense_data[global_tid]>eps)
        {
            int row = global_tid/width;
            int col = global_tid%width;
			printf("%d\n",row);
            atomicAdd(&row_ptr[row+1],1);
        }
    }
}

//__global__ void dense_to_csr(z(int height, int width, float *dense_data,
//        int* row_ptr, int* col_idx, float* sparse_data)

void dense_to_csr_cuda(Mat& dense)
{
    int* row_ptr_h = new int[dense.H+1];
    int* col_idx_h;
    float* sparse_data_h;

    int* row_ptr_d;
    int* col_idx_d;
    float* sparse_data_d;
	printf("==============================\n");	
	printf("dense (%d, %d)\n",dense.H, dense.W);

    cudaMalloc((void**)&row_ptr_d, sizeof(int)*(dense.H+1));
    cudaMemset(row_ptr_d, 0, sizeof(int)*(dense.H+1));
	
	float* dense_data_d;
	cudaMalloc((void**)&dense_data_d, sizeof(float)*(dense.H*dense.W));
	cudaMemcpy(dense_data_d, dense.data, sizeof(float)*(dense.H*dense.W),cudaMemcpyHostToDevice);

    int block_size = 1024;
    int block_num = (dense.H * dense.W)/block_size-1;
    
	count_row_nnz<<<block_num, block_size>>>
        (dense.H, dense.W, dense.data,
         row_ptr_d, col_idx_d, sparse_data_d);

    cudaMemcpy(row_ptr_h, row_ptr_d, sizeof(int)*(dense.H+1), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	for(int i=0;i<dense.H+1;i++)
    {
        row_ptr_h[i+1] += row_ptr_h[i];
    }
    int nnz = row_ptr_h[dense.H];
	printf("nnz : %d\n",nnz);

}

