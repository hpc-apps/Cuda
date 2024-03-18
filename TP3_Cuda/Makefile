CXX=g++
CXXFLAGS=-O3 -march=native -I/usr/include/opencv4/
LDLIBS=`pkg-config --libs opencv4`


grayscale: grayscale.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

grayscale-cu: grayscale.cu
	nvcc -o $@ $< $(LDLIBS)

.PHONY: clean

clean:
	rm grayscale grayscale-cu
