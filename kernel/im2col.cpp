#include <iostream>

// 3-Dimension representation
typedef struct _Mat{
    int N;
    int C;
    int D;
    int H;
    int W;
    float *data;
}Mat;

// convolution parameter
typedef struct _Param{
    int stride;
    int padding;
    int groups;
    int dilation;
}Param;

void im2col(const Mat& input, const Mat& filter, const Param& p)
{
    Mat output;
    output.N = 1;
    output.C = filter.N;
    output.D = 1+(input.D - filter.D + 2*p.padding)/p.stride;
    output.H = 1+(input.H - filter.H + 2*p.padding)/p.stride;
    output.W = 1+(input.W - filter.W + 2*p.padding)/p.stride;

    Mat input_2d;
    input_2d.N = 1;
    input_2d.C = 1;
    input_2d.D = 1;
    input_2d.H = (filter.D*filter.H*filter.W)*filter.C;
    input_2d.W = (output.D*output.H*output.W);
    input_2d.data = new float[input_2d.H*input_2d.W];

    Mat filter_2d;
    filter_2d.N = 1;
    filter_2d.C = 1;
    filter_2d.D = 1;
    filter_2d.H = filter.N;
    filter_2d.W = filter.D*filter.H*filter.W*filter.C;

    Mat output_2d;
    output_2d.N = 1;
    output_2d.C = 1;
    output_2d.D = 1;
    output_2d.H = filter_2d.H;
    output_2d.W = input_2d.W;

    printf("input_2d (%d,%d)\n",input_2d.H,input_2d.W);
    printf("filter_2d (%d,%d)\n",filter_2d.H,filter_2d.W);
    printf("output_2d (%d,%d)\n",output_2d.H,output_2d.W);

    // Filter im2col conversion(same access pattern) 
    filter_2d.data = filter.data;

    // Input im2col conversion
    for(int fc = 0; fc < filter.C; fc++)
    {
        int input_2d_channel_offset = fc*(filter.D*filter.H*filter.W)*input_2d.W;
        int input_channel_offset = fc*(input.D*input.H*input.W);

        for(int od = 0; od < output.D; od++)
        {
            for(int oh = 0; oh < output.H; oh++)
            {
                for(int ow = 0; ow < output.W; ow++)
                {
                    int window_offset = od*(input.H*input.W)
                        +oh*(input.W)
                        +ow;
                    int window_idx = od*(output.H*output.W)
                        +oh*(output.W)
                        +ow;
                    for(int fd = 0; fd < filter.D; fd++)
                    {
                        for(int fh = 0; fh < filter.H; fh++)
                        {
                            for(int fw = 0; fw < filter.W; fw++)
                            {
                                int f2i_offset = fd*(input.H*input.W)
                                    +fh*(input.W)
                                    +fw;
                                int input_idx = f2i_offset + window_offset + input_channel_offset;

                                int filter_idx = fd*(filter.H*filter.W)
                                    +fh*(filter.W)
                                    +fw;
                                input_2d.data[input_2d_channel_offset + window_idx + filter_idx*input_2d.W] = input_idx;
                            }
                        }
                    }
                }
            }
        }
    }
    for(int h = 0; h < input_2d.H; h++)
    {
        for(int w = 0; w < input_2d.W; w++)
        {
            printf("%.0f ",input_2d.data[(h*input_2d.W) + w]);
        }
        printf("\n");
    }

    return;   
}

void set_matrix(Mat& mat, int N, int C, int D, int H, int W)
{
    mat.N = N;
    mat.C = C;
    mat.D = D;
    mat.H = H;
    mat.W = W;
    mat.data = new float[mat.N*mat.C*mat.D*mat.H*mat.W];
}

void set_parameter(Param& p, int stride, int padding, int groups, int dilation)
{
    p.stride = stride;
    p.padding = padding;
    p.groups = groups;
    p.dilation = dilation;
}

int main()
{
    Param p;
    // Default convolution parameter
    set_parameter(p, 1,0,1,0);
    Mat input;
    // Default input size
    set_matrix(input,1,3,3,3,3);
    Mat filter;
    // Default kernel siz
    set_matrix(filter,4,3,2,2,2);    

    printf("input (%d,%d,%d,%d,%d)\n",input.N,input.C,input.D,input.H,input.W);
    printf("filter (%d,%d,%d,%d,%d)\n",filter.N,filter.C,filter.D,filter.H,filter.W);
    printf("param (stride = %d, padding = %d, groups=%d, dilation=%d)\n",p.stride,p.padding,p.groups,p.dilation);
    
    im2col(input, filter, p);
    return 0;
}