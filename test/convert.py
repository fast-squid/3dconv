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
    # load Dataset
    x = scipy.io.loadmat("../data/volumetric_data/toilet/30/test/toilet_000000135_9.mat")
    m = nn.ConstantPad3d(1,0)
    x_tensor = torch.tensor(x['instance'],dtype=torch.float32) 
    x_pad = m(x_tensor)
    x_pad = np.reshape(x_pad,(1,1,32,32,32))
    x_pad.numpy().astype("float32").tofile("./conv1_input.bin")
    print(x_pad.to_sparse())
    #print(x_pad)


    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(x_tensor, facecolors=None, edgecolor='k')
    plt.show()

if __name__ == "__main__":
    main()
