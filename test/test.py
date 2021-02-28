import torch
import torch.nn as nn
import os
import sys
import importlib

import numpy as np
import scipy.io

import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.insert(0,'../')

from model.voxnet import VoxNet


def main():
    # load VoxNet
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    module = importlib.import_module("model."+"voxnet")
    model = VoxNet(num_classes=10,input_shape=(32,32,32))  #10,30,40
    print(model)

    # load Dataset
    x = scipy.io.loadmat("../data/volumetric_data/toilet/30/test/toilet_000000135_9.mat")
    m = nn.ConstantPad3d(1,0)
    x_tensor = torch.tensor(x['instance'],dtype=torch.float32) 
    x_pad = m(x_tensor)
    x_pad = np.reshape(x_pad,(1,1,32,32,32))
   
    #print(x_pad)


    # load Weights
    data_dict = np.load("../shapenet10_weights.npz")
    model.state_dict()['body.conv1.weight'].data.copy_(torch.nn.Parameter(torch.Tensor(data_dict['conv1.W'])))
    model.state_dict()['body.conv1.bias'].data.copy_(torch.nn.Parameter(torch.Tensor(data_dict['conv1.b'])))
    model.state_dict()['body.conv2.weight'].data.copy_(torch.nn.Parameter(torch.Tensor(data_dict['conv2.W'])))
    model.state_dict()['body.conv2.bias'].data.copy_(torch.nn.Parameter(torch.Tensor(data_dict['conv2.b'])))

    model.state_dict()['head.fc1.weight'].data.copy_(torch.nn.Parameter(torch.Tensor(data_dict['fc1.W'])).t())
    model.state_dict()['head.fc1.bias'].data.copy_(torch.nn.Parameter(torch.Tensor(data_dict['fc1.b'])))
    model.state_dict()['head.fc2.weight'].data.copy_(torch.nn.Parameter(torch.Tensor(data_dict['fc2.W'])).t())
    model.state_dict()['head.fc2.bias'].data.copy_(torch.nn.Parameter(torch.Tensor(data_dict['fc2.b'])))
    
    # inference
    model.eval()
    with torch.no_grad():
        y=model.forward(x_pad)
        print(y)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(x_tensor, facecolors=None, edgecolor='k')
    plt.show()

if __name__ == "__main__":
    main()
