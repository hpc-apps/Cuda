#include <iostream>
#include <vector>


__global__ void vecswap( int * v1, int * v2  )
{
  auto tid = threadIdx.x;
  int  swap;
         swap = v1[tid];
	 v1[tid] = v2 [tid];
	 v2[tid] = swap ;
}


int main()
{
  std::vector< int > v1( 10 );
  std::vector< int > v2( 10 );
  
  int * v1_d = nullptr;
  int * v2_d = nullptr;

  for( std::size_t i = 0 ; i < v1.size() ; ++i )
  {
    v1[ i ] =  i;
    v2[ i ] =  v1.size()-i;
  }
  
  cudaMalloc( &v1_d, v1.size() * sizeof( int ) );
  cudaMalloc( &v2_d, v2.size() * sizeof( int ) );

  cudaMemcpy(v1_d, v1.data(), v1.size() * sizeof( int ), cudaMemcpyHostToDevice );
  cudaMemcpy(v2_d, v2.data(), v2.size() * sizeof( int ), cudaMemcpyHostToDevice );
   
  for (size_t idex = 0; idex < v1.size() ; idex++)
	std::cout << v1[idex] <<"/"<< v2[idex] << " ";
  	std::cout << std::endl;


  vecswap<<< 1, 10 >>>( v1_d, v2_d );

  cudaMemcpy(v1.data(),v1_d, v1.size() * sizeof( int ), cudaMemcpyDeviceToHost );
  cudaMemcpy(v2.data(),v2_d, v2.size() * sizeof( int ), cudaMemcpyDeviceToHost );

 
 
  for (size_t idex = 0; idex < v1.size(); idex++)
    std::cout << v1[idex] <<"//"<<  v2[idex] << " ";
    std::cout << std::endl;


  cudaFree( v1_d );
  cudaFree( v2_d );

  return 0;
}
