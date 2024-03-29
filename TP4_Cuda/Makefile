CXX=g++
CXXFLAGS=-O3 -march=native
LDLIBS1=`pkg-config --libs opencv`
LDLIBS2=-lm -lIL

all: sobel sobel-cu

sobel: sobel.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS2)

sobel-cu: sobel.cu
	nvcc -o $@ $<  $(LDLIBS1)

.PHONY: clean

clean:
	rm sobel sobel-cu
