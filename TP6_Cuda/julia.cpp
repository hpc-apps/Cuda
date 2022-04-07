#include <opencv2/opencv.hpp>

#include <iostream>
#include <cstring>
#include <chrono>

using namespace std;

using ui32 = unsigned int;

ui32 const dim = 2048;


struct complex {
  float r; float i;
  complex(float r, float i) : r(r), i(i) {}
  float magnitude() {return r*r + i*i;}
  complex operator*(const complex& c) {
    return complex(r * c.r - i * c.i, i * c.r + r * c.i);
  }
  complex operator+(const complex& c) {
    return complex(r + c.r, i + c.i);
  }
};


unsigned char julia( int x, int y )
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


int main()
{
  unsigned char * out = reinterpret_cast< unsigned char * >( aligned_alloc( 16, dim * dim ) );

  //
  unsigned char * p = out;

  auto start = std::chrono::system_clock::now();

  for( ui32 j = 0 ; j < dim ; ++j ) {
    for( ui32 i = 0 ; i < dim ; ++i ) {

      out[ i + j * dim ] = julia( i, j );

    }
  }
  
  auto stop = std::chrono::system_clock::now();

  std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(stop -start).count() << "ms\n";

  //
  
  cv::Mat m_out( dim, dim, CV_8UC1, out );
  
  imwrite( "julia.png", m_out );

  return 0;
}

