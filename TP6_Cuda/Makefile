CXX=g++
CXXFLAGS=-O3 -march=native
LDLIBS=`pkg-config --libs opencv`


.PHONY: clean

julia: julia.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

julia-cu: julia.cu
	nvcc -o $@ $< $(LDLIBS)

clean:
	rm -f julia julia-cu

