CXX=g++
CXXFLAGS=-O3 -march=native -I/usr/include/opencv4/
NVFLAGS=-O3 -I/usr/include/opencv4/ -ccbin g++-10
LDLIBS=`pkg-config --libs opencv4`

grayscale: grayscale.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

grayscale-cu: grayscale.cu
	nvcc $(NVFLAGS) -o $@ $< $(LDLIBS)

.PHONY: clean

clean:
	rm grayscale grayscale-cu
