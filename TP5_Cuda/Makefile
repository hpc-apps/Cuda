CXX=g++
CXXFLAGS=-O3 -march=native
LDLIBS1=-lm -lIL
LDLIBS2=`pkg-config --libs opencv`


all: sobel-cu

sobel: sobel.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS1)

sobel-cu: sobel.cu
	nvcc -o $@ $< $(LDLIBS2)

.PHONY: clean

clean:
	rm sobel sobel-cu
