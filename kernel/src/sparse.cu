#include "mat.h"
#include <iostream>
#include <stdlib.h>

#define ERROR_CHECK \
{\
	cudaError err = cudaGetLastError(); \
	if ( cudaSuccess != err ) \
	{\
		printf("[%s:%d]CUDA ERROR : %s\n", __FILE__, __LINE__, cudaGetErrorString(err) ); \
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
		int input_d_idx = (((blockIdx.x / output_w) / output_h) % output_d)*stride + threadIdx.z;

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
	if(x.col == y.col) return x.row - y.row;
	else return x.col - y.col;
}

int compare2(const void* a, const void* b)
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
	
	input.row_num = filter.C*filter.D*filter.H*filter.W;
	input.col_num = output_D*output_H*output_W;

	printf("input shape  : (%d,%d,%d,%d,%d)\n",input.N, input.C, input.D, input.H, input.W);
	printf("output shape : (%d,%d,%d,%d,%d)\n",output_N, output_C, output_D, output_H, output_W);
	printf("im2col shape : (%d,%d)\n",input.row_num, input.col_num);
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

__global__ void coo_2_csc(COO* coo, 
		int* ptr, int *idx, float* val,
		int nnz)
{
    int global_tid = threadIdx.x+blockDim.x*blockIdx.x;
	if(global_tid<nnz)
	{
		idx[global_tid] = coo[global_tid].row;
		val[global_tid] = coo[global_tid].val;
		atomicAdd(&ptr[coo[global_tid].col+1], 1);
	}
}

void coo_to_csc_cuda(Mat& input, int& number_of_non_zero_vectors, int *non_zero_vectors)
{
	cudaMalloc((void**)&input.ptr_dev, sizeof(int)*(input.col_num+1));
	cudaMalloc((void**)&input.idx_dev, sizeof(int)*input.nnz);
	cudaMalloc((void**)&input.val_dev, sizeof(float)*input.nnz);
	cudaMemset(input.ptr_dev, 0, sizeof(int)*(input.col_num+1));

	int block_num = input.nnz/1024+1;
	int block_size = 1024;
	
	coo_2_csc<<<block_num, block_size>>>(input.coo_dev,
			input.ptr_dev, input.idx_dev, input.val_dev,
			input.nnz);
	ERROR_CHECK;	
	
	input.ptr = new int[input.col_num+1];
	input.idx = new int[input.nnz];
	input.val = new float[input.nnz];

	cudaMemcpy(input.ptr, input.ptr_dev, sizeof(int)*(input.col_num+1),cudaMemcpyDeviceToHost);
	cudaMemcpy(input.idx, input.idx_dev, sizeof(int)*input.nnz,cudaMemcpyDeviceToHost);
	cudaMemcpy(input.val, input.val_dev, sizeof(float)*input.nnz,cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	for(int i=0;i<input.col_num;i++)
	{
		input.ptr[i+1] += input.ptr[i];
		if(input.ptr[i+1]-input.ptr[i])
		{
			non_zero_vectors[number_of_non_zero_vectors] = i;
			number_of_non_zero_vectors++;
		}
	}

	printf("nnz %d %d\n",input.nnz, input.ptr[input.col_num]);
	cudaMemcpy(input.ptr_dev, input.ptr, sizeof(int)*(input.col_num+1),cudaMemcpyHostToDevice);
}

/*
 * Function: dense_sparse_mm
 * TODO: fix code for any integer number for a_height
 * -----------------------------------------------------------------------
 * computes matrix multiplication  C = AxB corresponding to convolution 
 * Parameters ------------------------------------------------------------
 * matrix A (dense matrix A correspoinding to filter)
 * 	a_height : height of matrix A
 *	a_width  : width of matrix A 
 * 	a_val    : value array of matrix A
 * matrix B (CSC formatted B corresponding to input feature map) 
 *	b_ptr	 : row pointer array of matrix B 
 *	b_idx	 : column index array of matrix B
 *	b_val	 : value array of matrix B
 * non_zero_vectors : non-zero column vector idx array of matrix B
 * returns ---------------------------------------------------------------
 * matrix C (CSC forammted C correspoint to output feature map)
 * 	 c_ptr, c_idx, c_val : same as B
 */
__global__ void dense_sparse_mm(const int a_height, const int a_width, const float* a_val,
		const int* b_ptr, const int* b_idx, const float* b_val,
		int* c_ptr, int* c_idx, float* c_val,
		const int* non_zero_vectors, const int number_of_non_zero_vectors)
{
	int gtid = threadIdx.x + blockIdx.x*blockDim.x;
	int gwid = gtid/32;
	int tid = gtid%blockDim.x;
	int wid = tid/32;
	if(gwid<number_of_non_zero_vectors)
	{
		//int b_cidx = gwid;
		int b_cidx = non_zero_vectors[gwid];
		int b_offset = b_ptr[b_cidx];
		int nnz = b_ptr[b_cidx + 1] - b_ptr[b_cidx];
		//if(threadIdx.x %32 == 0)
		//	printf("b_cidx %d nnz %d\n",b_cidx,nnz);
		
		//__shared__ float smem_a_val[128][32];
		
		// compute tiled matrix multiplication
		float val = 0;
		for(int i=0 ; i < nnz; i++)
		{	
			//smem_a_val[i][32*wid + tid%32] = a_val[a_width*(tid%32) + b_idx[b_offset+i]];
			val += a_val[a_width*(b_idx[b_offset+i])+wid*32+tid%32] *b_val[b_offset+i];

		}
		// compute remaining matrix multiplication
		int c_cidx = b_cidx;
		int c_height = a_width;
		c_idx[c_height*gwid + tid%32] = tid%32;
		c_val[c_height*gwid + tid%32] = val;
		c_ptr[c_cidx+1] = 32;
	}

}

void call_when_model_loaded(Mat& filter)
{	
	set_transpose(filter);
	int size = filter.N*filter.C*filter.D*filter.H*filter.W;
	cudaMalloc((void**)&filter.data_dev, sizeof(float)*size);
	cudaMemcpy(filter.data_dev, filter.data_trans, sizeof(float)*size, cudaMemcpyHostToDevice);
	printf("filter size %d\n",size);
}

void dense_sparse_mm_cuda(Mat& input, Mat& filter, Mat& output,
		int number_of_non_zero_vectors, int* non_zero_vectors)
{

	int a_height = filter.C*filter.D*filter.H*filter.W;
	int a_width = filter.N;
	int b_height = input.row_num;
	int b_width = input.col_num;

	int* non_zero_vectors_dev;
	cudaMalloc((void**)&non_zero_vectors_dev, sizeof(int)*number_of_non_zero_vectors);
	cudaMemcpy(non_zero_vectors_dev, non_zero_vectors, sizeof(int)*number_of_non_zero_vectors, cudaMemcpyHostToDevice);
	call_when_model_loaded(filter);

	output.row_num = a_width;
	output.col_num = b_width;
	output.nnz = a_width*number_of_non_zero_vectors;

	cudaMalloc((void**)&output.ptr_dev, sizeof(int)*(output.col_num+1));
	cudaMalloc((void**)&output.idx_dev, sizeof(int)*output.nnz);
	cudaMalloc((void**)&output.val_dev, sizeof(float)*output.nnz);
	cudaMemset(output.ptr_dev, 0, sizeof(int)*(output.col_num+1));
	int tmp = number_of_non_zero_vectors;
	int block_num = tmp;
	int block_size = 32;
	
	printf("%d %d\n",a_height, a_width);
	printf("%d %d\n",b_height, b_width);
	printf("%d %d %d\n",output.row_num, output.col_num, output.nnz);
	printf("bs %d bn %d\n",block_size, block_num);	
	
	dense_sparse_mm<<<block_num, block_size >>>(a_height, a_width, filter.data_dev,
		input.ptr_dev, input.idx_dev, input.val_dev,
		output.ptr_dev,  output.idx_dev, output.val_dev,
		non_zero_vectors_dev,number_of_non_zero_vectors);
	ERROR_CHECK;	
	output.ptr = new int[output.col_num+1];
	output.idx = new int[output.nnz];
	output.val = new float[output.nnz];
	
	cudaMemcpy(output.ptr, output.ptr_dev, sizeof(int)*(output.col_num+1),cudaMemcpyDeviceToHost);
	cudaMemcpy(output.idx, output.idx_dev, sizeof(int)*output.nnz,cudaMemcpyDeviceToHost);
	cudaMemcpy(output.val, output.val_dev, sizeof(float)*output.nnz,cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	Mat temp;
	temp.row_num = output.row_num;
	temp.col_num = output.col_num;
	temp.coo = new COO[output.nnz];
	int temp_idx = 0;
	for(int i=0; i<output.col_num; i++)
	{
		output.ptr[i+1] += output.ptr[i];
		int nnz_per_col = output.ptr[i+1] - output.ptr[i] ;
		int offset = output.ptr[i];
		for(int j=0;j<nnz_per_col;j++)
		{
			temp.coo[temp_idx].row = output.idx[offset+j];
			temp.coo[temp_idx].col = i;
			temp.coo[temp_idx].val = output.val[offset+j];
			temp_idx++;
		}
	}
	temp.nnz = output.ptr[output.col_num];

	qsort(temp.coo, temp.nnz, sizeof(COO),compare2);
	print_coo(temp);
	
}

void sparse_conv_cuda(Mat& input, Mat& filter, Param& p, Mat& output)
{
	cube_to_coo_cuda(input, filter, p);
	int* non_zero_vectors = new int[input.col_num+1];
	int number_of_non_zero_vectors = 0;
	coo_to_csc_cuda(input, number_of_non_zero_vectors, non_zero_vectors);
	printf("number of non zero vectors : %d\n",number_of_non_zero_vectors);
	dense_sparse_mm_cuda(input, filter, output, number_of_non_zero_vectors, non_zero_vectors);

	
}
