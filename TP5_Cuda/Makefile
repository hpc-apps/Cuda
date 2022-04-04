CXX=g++
CXXFLAGS=-O3 -march=native
LDLIBS=`pkg-config --libs opencv`


all: sobel-cu

sobel: sobel.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

sobel-cu: sobel.cu
	nvcc -o $@ $< $(LDLIBS)

.PHONY: clean

clean:
	rm sobel sobel-cu
