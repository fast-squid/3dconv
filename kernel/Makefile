CXX=g++
NVCC=nvcc

CPPFLAGS =-g -c  
NVCCFLAGS=-g -c -arch=sm_61
INCLUDE=-I./include
 
./bin/test: ./obj/im2col.cpp.o ./obj/sparse.cu.o ./obj/mat.cpp.o
	nvcc -arch=sm_61 -o $@ $^

./obj/im2col.cpp.o: ./src/im2col.cpp
	$(CXX) $(CPPFLAGS) $(INCLUDE) -o $@ $< 

./obj/sparse.cu.o : ./src/sparse.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -o $@ $<

./obj/mat.cpp.o: ./src/mat.cpp
	$(CXX) $(CPPFLAGS) $(INCLUDE) -o $@ $< 
clean:
	rm ./obj/*
	rm ./bin/*

