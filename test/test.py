import torch
import torch.nn as nn
import os
import sys
import importlib

import numpy as np
import scipy.io

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
    x = scipy.io.loadmat("../3DShapeNets/volumetric_data/car/30/test/car_000000353_9.mat")
    m = nn.ConstantPad3d(1,0)
    x_tensor = torch.tensor(x['instance']) 
    x_pad = m(x_tensor)
    #print(x_pad)


    # load Weights
    model_params = model.state_dict()
    data_dict = np.load("../shapenet10_weights.npz")
    model_params['body.conv1.weight'] = data_dict['conv1.W']
    model_params['body.conv1.bias'] = data_dict['conv1.b']
    model_params['body.conv2.weight'] = data_dict['conv2.W']
    model_params['body.conv2.bias'] = data_dict['conv2.b']
    model_params['head.fc1.weight'] = data_dict['fc1.W']
    model_params['head.fc1.bias'] = data_dict['fc1.b']
    model_params['head.fc2.weight'] = data_dict['fc2.W']
    model_params['head.fc2.bias'] = data_dict['fc2.b']
    
    # inference
    model.eval()
    y=model.forward(x_pad)
    


if __name__ == "__main__":
    main()