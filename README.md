## Voxnet Architecture - Netron
![fig](./fig/Figure2.png)
## Tutorial Codes for 3D convolution
This is a tutorial code for studying the __3D CNN__
It aims to make beginners to acheive basic understanding on 3D Convolution.
I'm working on the code to make the application fully functional.
My goal is to profile 3D convolution in GPU and replace the kernel to Sparse Matrix Multiplication based Convolution.
_Since the 3D data is very sparse_, there can be some issues when the CNN task is offloaded to GPU.
The sparsity and irregular distribution of the 3D data can incur _poor utilization_ in GPU.
Also representing 3D world in dense matrix format cause memory pressure.
I expect implementing Sparse 3D CNN will greatly improve both performance and the scalabilty.

The base idea is referencing the [VoxNet](http://dimatura.net/publications/voxnet_maturana_scherer_iros15.pdf).

## Converting Dense Matrix to Sparse Matrix
To convert Dense matrix to Sparse matrix, sparse.cu provides gpu accelerated function.
2d Dense array -> coo(coordinate) -> csc/csr format


### Dependencies

### Examples
### Usage

