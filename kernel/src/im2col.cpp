#include <iostream>
#include <fstream>
#include "mat.h"
#include "sparse.h"
#include "model.h"
#include <vector>


void load_mat_from_bin(Mat& mat, std::string path)
{
    std::ifstream read_file(path, std::ios::binary);
	printf("%d\n",mat.get_mat_size());
    if(!read_file.is_open())
    {
        printf("file open error\n");
        exit(-1);
    }
    float val;
    int idx = 0;
    while(read_file.read(reinterpret_cast<char*>(&val), sizeof(float)))
    {
        mat.data[idx++] = val;
    }
}


int main(int argc, char* argv[])
{
    Mat filter(32,1,5,5,5);
    Mat input(1,1,32,32,32);
	Mat output;
	Mat temp;
    Param p;
	Model voxnet("voxnet");
	voxnet.push_inner_layers(new Conv({1,1,32,32,32},{32,1,5,5,5},{2,0,1,0}));
	voxnet.push_inner_layers(new ReLU);
	voxnet.push_inner_layers(new Conv({1,32,14,14,14},{32,32,3,3,3},{2,0,1,0}));
	voxnet.push_inner_layers(new ReLU);
	voxnet.push_inner_layers(new Pooling);

	// print network model
	for(int i=0;i<voxnet.inner_layers.size();i++)
	{
		voxnet.inner_layers[i]->forward(temp);
	}
	std::cout << voxnet.name << std::endl;
	
	load_mat_from_bin(input, "../data/conv1_input.bin");
	load_mat_from_bin(((Conv*)voxnet.inner_layers[0])->filter, "../data/conv1_weight.bin");
	load_mat_from_bin(((Conv*)voxnet.inner_layers[2])->filter, "../data/conv2_weight.bin");

	voxnet.set_cuda();		
	return 0;
	// conv layer 1
	sparse_conv_cuda(input, filter, p, output);
	
	
	
    return 0;
}
