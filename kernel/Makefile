CXX=g++
NVCC=nvcc

CPPFLAGS =-g -c  
NVCCFLAGS=-g -c -arch=sm_61


./test: im2col.cpp.o sparse.cu.o
	nvcc -arch=sm_61 -o $@ $^

./im2col.cpp.o: ./im2col.cpp
	$(CXX) $(CPPFLAGS) -o $@ $< 

./sparse.cu.o : ./sparse.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

