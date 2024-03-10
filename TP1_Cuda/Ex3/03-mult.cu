#include <iostream>
#include <vector>


__global__ void vecmult( int * v, std::size_t size )
{
  // Get the id of the thread ( 0 -> 10 ).
  auto tid = threadIdx.x;
  if (!(v[ tid ]%2)) v[tid] = v[tid]*2;
}


int main()
{
  std::vector< int > v( 10 );
  int * v_d = nullptr;


  for ( std::size_t i = 0 ; i < v.size() ; ++i)
	v[i] = i;


  // Allocate an array an the device.
  cudaMalloc( &v_d, v.size() * sizeof( int ) );

  cudaMemcpy( v_d, v.data() , v.size() * sizeof(int),cudaMemcpyHostToDevice);

  // In this block, threads are numbered from 0 to 10.
  vecmult<<< 1, 10 >>>( v_d, v.size() );

  // Copy data from the device memory to the host memory.
  cudaMemcpy( v.data(), v_d, v.size() * sizeof( int ), cudaMemcpyDeviceToHost );


  for (size_t idex =0 ; idex < v.size(); ++idex)
    std::cout << v[idex]  << " "; 
    std::cout << std::endl;

  cudaFree( v_d );

  return 0;
}
