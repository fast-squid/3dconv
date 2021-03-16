#pragma once
#include "mat.h"
#include <vector>

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
	Mat filter;
	Mat bias;
	Param param;
	void forward(const Mat&)
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
	string name;
	Model(string _name="") : name(_name)
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
	void print_model()
	{

	}
};
