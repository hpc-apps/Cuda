#include <opencv2/opencv.hpp>

#include <iostream>
#include <cstring>
#include <chrono>

using namespace std;

using ui32 = unsigned int;

ui32 const dim = 2048;


struct complex {
  float r; float i;
  __host__ __device__ complex(float r, float i) : r(r), i(i) {}
  __host__ __device__ float magnitude() {return r*r + i*i;}
  __host__ __device__ complex operator*(const complex& c) {
    return complex(r * c.r - i * c.i, i * c.r + r * c.i);
  }
  __host__ __device__ complex operator+(const complex& c) {
    return complex(r + c.r, i + c.i);
  }
};


__device__ unsigned char julia( int x, int y )
{
  const float scale = 1.5;

  float jx = scale * (float)(dim/2.0f - x)/(dim/2.0f);
  float jy = scale * (float)(dim/2.0f - y)/(dim/2.0f);

  ::complex c(-0.8, 0.156);
  ::complex a(jx, jy);

  for(unsigned int i = 0 ; i < 200 ; ++i) {

    a = a * a + c;

    if(a.magnitude() > 1000) {
      return 0;
    }

  }

  return 255;
}

__global__ void julia_kernel( unsigned char * out, ui32 dim )
{
  auto tidx = blockIdx.x * blockDim.x + threadIdx.x;
  auto tidy = blockIdx.y * blockDim.y + threadIdx.y;

  if( tidx < dim && tidy < dim )
  {
    out[ tidy*dim + tidx ] = julia( tidx, tidy );
  } 
}



int main()
{
  unsigned char * out = reinterpret_cast< unsigned char * >( aligned_alloc( 16, dim * dim ) );

  //

  unsigned char * out_d = nullptr;

  cudaMalloc( &out_d, dim*dim );

  dim3 block( 128, 4 );
  dim3 grid( (dim-1)/block.x+1, (dim-1)/block.y+1 );

cudaEvent_t start, stop;
cudaEventCreate( &start );
cudaEventCreate( &stop );

cudaEventRecord( start );

  julia_kernel<<< grid, block >>>( out_d, dim );



  cudaMemcpy( out, out_d, dim*dim, cudaMemcpyDeviceToHost );

  cudaEventRecord( stop );

  cudaEventSynchronize( stop );

  float duration;
  cudaEventElapsedTime( &duration, start, stop );

  std::cout << duration << "ms\n";


  cv::Mat m_out( dim, dim, CV_8UC1, out );
  
  imwrite( "julia.png", m_out );

  return 0;
}

