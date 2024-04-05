CXX=g++
CXXFLAGS=-O3 -march=native -I/usr/include/opencv4/
LDLIBS1=`pkg-config --libs opencv4`
NVFLAGS=-O3 -I/usr/include/opencv4/ -ccbin g++-10
LDLIBS2=-lm -lIL

all: sobel sobel-cu

sobel: sobel.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS2)

sobel-cu: sobel.cu
	nvcc $(NVFLAGS) -o $@ $<  $(LDLIBS1)

.PHONY: clean

clean:
	rm sobel sobel-cu
