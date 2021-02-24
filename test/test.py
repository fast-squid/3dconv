import torch
import torch.nn as nn
import os
import sys
import importlib

import numpy as np

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


    # load Weights
    
    model_params = model.state_dict()
    data_dict = np.load("../shapenet10_weights.npz")
    model_params['body.conv1.weight'] = data_dict['conv1.W']
    model_params['body.conv1.bias'] = data_dict['conv1.b']
    model_params['body.conv1.weight'] = data_dict['conv1.W']
    model_params['body.conv1.bias'] = data_dict['conv1.b']
    for key, value in data_dict.items():
        print(key)
    for key, value in model_params.items():
        print(key)

    #model_params["body"] = torch.from_numpy(data_dict["conv1"]).type(float)
    
    #print(model_params["conv1.weight"])
    


if __name__ == "__main__":
    main()