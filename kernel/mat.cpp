#include "mat.h"
#include <stdio.h>
#include <stdlib.h>
int get_mat_size(const Mat& mat)
{
	return mat.N*mat.C*mat.D*mat.H*mat.W;
}

void print_mat(const Mat& mat)
{
	int nnz = 0;
	for(int n=0;n<mat.N;n++)
	{
		for(int c = 0; c<mat.C;c++)
		{
			for(int d = 0;d<mat.D;d++)
			{
				for(int h = 0; h<mat.H;h++)
				{
					for(int w=0; w<mat.W;w++)
					{
						int idx = n*mat.C*mat.D*mat.H*mat.W
							+c*mat.D*mat.H*mat.W
							+d*mat.H*mat.W
							+h*mat.W
							+w;
							//printf("%d %d %d %d %f\n",c,d,h,w,mat.data[idx]);
							printf("%f\n",mat.data[idx]);
							nnz++;
					}
				}
			}
		}
	}
	printf("nnz : %d\n",nnz);
}

void print_coo(const Mat& mat)
{
	for(int i=0;i<mat.nnz;i++)
	{
		printf("%f\n",mat.coo[i].val);

	//	printf("%d %d %f\n",mat.coo[i].row, mat.coo[i].col, mat.coo[i].val);
	}
}

void set_rand(Mat& mat)
{
	for(int i=0;i<get_mat_size(mat);i++)
	{
		mat.data[i] = ((float)(rand()%100))/100;
	}
}

void set_transpose(Mat& mat)
{
	mat.data_trans = new float[get_mat_size(mat)];
	int row = mat.N;
	int col = mat.C*mat.D*mat.H*mat.W;
	
	for(int j=0; j < col; j++)
	{
		for(int i = 0; i < row ;i++)
		{
			mat.data_trans[j*row + i] = mat.data[i*col + j];
		}
	}
}

float* transpose(const Mat& mat)
{
	return mat.data_trans;
}


