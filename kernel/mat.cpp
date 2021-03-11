#include "mat.h"
#include <stdio.h>
int get_mat_size(const Mat& mat)
{
	return mat.N*mat.C*mat.D*mat.H*mat.W;
}

void print_mat(Mat& mat)
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
						if(mat.data[idx]>0.00001){
							//printf("%d %d %d %d %f\n",c,d,h,w,mat.data[idx]);
							printf("%f\n",mat.data[idx]);
							nnz++;
						}
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
		printf(" %f\n",mat.coo[i].val);

	//	printf("%d %d %f\n",mat.coo[i].row, mat.coo[i].col, mat.coo[i].val);
	}
}

