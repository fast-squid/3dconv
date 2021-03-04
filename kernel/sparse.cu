#include "mat.h"
#include <iostream>
#define ERROR_CHECK \
{\
	cudaError err = cudaGetLastError(); \
	if ( cudaSuccess != err ) \
	{\
		printf("[%s:%d]CUDA ERROR : %s\n", __FILE__, __LINE__, cudaGetErrorString(err) ); \
		exit(-1); \
	}\
}
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

// implementation without convolution parameter(padding, stride, groups, dilation)
__global__ void cube_to_coo(int input_d, int input_h, int input_w, float* input_data,
		int output_d, int output_h, int output_w, float* output_data,
		int* row, int* col, float* val, int *nnz)
{
    // block[8] : 0 ~ 7
	//printf(".\n");	
	if(threadIdx.x <5 && threadIdx.y < 5 && threadIdx.z <5)
	{
		
		int input_w_idx = (blockIdx.x % output_w) + threadIdx.x;
		int input_h_idx = (blockIdx.x / output_w) % output_h + threadIdx.y;
		int input_d_idx = ((blockIdx.x / output_w) / output_d) % output_d + threadIdx.z;

		int input_idx =  input_d_idx*(input_h*input_w)
			+ input_h_idx*(input_w)
			+ input_w_idx;
		if(input_data[input_idx]>eps)
		{
			int global_idx = atomicAdd(nnz, 1);
			row[global_idx] = threadIdx.z*(blockDim.y*blockDim.x)
				+ threadIdx.y*blockDim.x
				+ threadIdx.x;
			col[global_idx] = blockIdx.x;
		}
	}
	
}

void cube_to_coo_cuda(Mat& input, Mat& filter, Param& p)
{
    Mat output;
	output.N = 1;
	output.C = filter.N;
	output.D = 1+(input.D - filter.D + 2*p.padding)/p.stride;
	output.H = 1+(input.H - filter.H + 2*p.padding)/p.stride;
	output.W = 1+(input.W - filter.W + 2*p.padding)/p.stride;

	printf("input shape  : (%d,%d,%d,%d,%d)\n",input.N, input.C, input.D, input.H, input.W);
	printf("output shape : (%d,%d,%d,%d,%d)\n",output.N, output.C, output.D, output.H, output.W);

	// coo format
	int *row_h;
	int *col_h;
	float *val_h;

	int *row_d = new int[3000];
	int *col_d = new int[3000];
	float *val_d = new float[3000];

	int* nnz_d;

	cudaMalloc((void**)&row_d,sizeof(int)*3000);
	cudaMalloc((void**)&col_d,sizeof(int)*3000);
	cudaMalloc((void**)&val_d,sizeof(float)*3000);
	cudaMalloc((void**)&nnz_d, sizeof(int));
	cudaMemset(nnz_d, 0, sizeof(int));
	// 3D-input
	float* input_data_d;
	int input_size = input.N*input.C*input.D*input.H*input.W;
	cudaMalloc((void**)&input_data_d, sizeof(float)*input_size);
	cudaMemcpy(input_data_d, input.data, sizeof(float)*input_size,cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	int counter = 0;
	for(int i=0;i<input.D * input.H * input.W;i++)
	{
		if(input.data[i] >eps)
		{
			counter++;
			printf("%d\n",i);
		}
	}
	printf("counter %d\n",counter);

	int block_num = output.D*output.H*output.W;
	printf("%d\n",output.D*output.H*output.W);
	dim3 block_size(8,8,8);
	//printf("block_num : %d block_size (%d %d %d)\n",block_num, block_size.x, block_size.y, block_size.z);
	cube_to_coo<<<block_num, block_size>>>
		(input.D, input.H, input.W, input_data_d,
		 output.D, output.H, output.W, NULL,
		 row_d, col_d, val_d, nnz_d);
	ERROR_CHECK;	

	cudaDeviceSynchronize();
	int nnz;
	cudaMemcpy(&nnz,nnz_d, sizeof(int), cudaMemcpyDeviceToHost);
	
}


