#pragma once
#include "mat.h"
#include <vector>
#include <string>
using namespace std;

const int CONV=0;
const int LKRELU=1;
const int POOL=2;
const int FC=3;

class Layer
{
public:
	int id;
	int type;
	vector<Layer*> inner_layers;
	Layer()
	{
	}
	Layer(int _id)
		: id(_id){}
	Layer(const int (&in_shape)[5], const int (&filter_shape)[5]){}
	virtual void push_inner_layers(Layer* new_layer)
	{
		inner_layers.push_back(new_layer);
	}
	virtual void forward(){};
	virtual void forward(Mat&){};
	virtual void forward(const Mat&){};
};

class Conv : public Layer
{
public:
	int input_shape[5];
	Mat filter;
	Mat bias;
	Param param;
	Conv(){}
	Conv(const int (&in_shape)[5], const int (&filter_shape)[5], const int (&param_shape)[4], bool bias = true)
	{
		input_shape[0] = in_shape[0];	// N
		input_shape[1] = in_shape[1];	// C
		input_shape[2] = in_shape[2];	// D
		input_shape[3] = in_shape[3];	// H
		input_shape[4] = in_shape[4];	// W
		filter.set_matrix(filter_shape);	
		param.set_parameter(param_shape);
	}
	void forward(Mat&)
	{
		printf("conv forward\n");
	}
};

class ReLU : public Layer
{
public:
	int inplace;
	ReLU()
	{
	}
	void forward(Mat&)
	{
		printf("relu forward\n");
	}
};

class Pooling : public Layer
{
public:
	Param param;
	Pooling()
	{
	}
	void forward(Mat&)
	{
		printf("pooling forward\n");
	}
};

class Model : public Layer
{
public:
	std::string name;
	Model(std::string _name="") : name(_name)
	{
	}
	void forward(Mat&)
	{
		printf("model forward\n");
	}
	void forward(Mat&, int start, int end)
	{
		printf("model forward %d %d\n",start,end);
	}
	void set_cuda();
	void print_model()
	{
	}
};
