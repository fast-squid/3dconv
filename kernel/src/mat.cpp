#include "mat.h"
#include <stdio.h>
#include <stdlib.h>

void print_mat(const Mat& mat)
{
	int nnz = 0;
	for(int n=0;n<mat.n;n++)
	{
		for(int c = 0; c<mat.c;c++)
		{
			for(int d = 0;d<mat.d;d++)
			{
				for(int h = 0; h<mat.h;h++)
				{
					for(int w=0; w<mat.w;w++)
					{
						int idx = n*mat.c*mat.d*mat.h*mat.w
							+c*mat.d*mat.h*mat.w
							+d*mat.h*mat.w
							+h*mat.w
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
	for(int i=0;i<mat.get_mat_size();i++)
	{
		mat.data[i] = ((float)(rand()%100))/100;
	}
}

void Mat::set_transpose()
{
	data_trans = new float[get_mat_size()];
	int row = n;
	int col = c*d*h*w;
	
	for(int j = 0; j < col; j++)
	{
		for(int i = 0; i < row ;i++)
		{
			data_trans[j*row + i] = data[i*col + j];
		}
	}
}

float* transpose(const Mat& mat)
{
	return mat.data_trans;
}


