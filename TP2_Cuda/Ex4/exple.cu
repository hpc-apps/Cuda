#include <iostream>
#include <vector>


__global__ void addMat( float * mA_d, float * mB_d, std::size_t w, std::size_t h )
{

  // coord glob/local		


  // sum
}


int main()
{
  size_t const w = 100;
  size_t const h = 100;
  size_t const size = w * h;
  
  std::vector< float > mA ( size );
  std::vector< float > mB ( size );
  std::vector< float > mC ( size );

  float * mA_d = nullptr;
  float * mB_d = nullptr;
  
  std::fill( begin( mA), end( mA ), 1.0f );
  std::fill( begin( mB), end( mB ), 1.0f );

// cudamalloc 

//cudamemcpy


// 2D grid definition

 
// Kernel  
  addMat<<< grid, block >>>( mA_d, mB_d, w, h );


// memcpy  

  for( std::size_t j = 0 ; j < h ; ++j )
  {
    for( std::size_t i = 0 ; i < w ; ++i )
    {
      std::cout << mC[ j * w + i ] << ' ';
    }
    std::cout << std::endl;
  }
  
  return 0;
}
