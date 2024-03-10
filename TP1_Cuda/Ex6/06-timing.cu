#include <iostream>
#include <vector>


__global__ void vecadd( int * v1, int * v2 )
{
  auto tid = threadIdx.x;

         v2[tid] += v1[tid];
	 
}


int main()
{
  int N=1024;
  std::vector< int > v1( N );
  std::vector< int > v2( N );
  cudaError_t cudaStatus;
  cudaError_t kernelStatus;
  float elapsedTime;
  cudaEvent_t start, stop;

  int * v1_d = nullptr;
  int * v2_d = nullptr;



  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  for( std::size_t i = 0 ; i < v1.size() ; ++i )
  {
    v1[ i ] =  i;
    v2[ i ] =  i;
  }
  
  
  
  cudaStatus = cudaMalloc( &v1_d, v1.size() * sizeof( int ) );
  if (cudaStatus != cudaSuccess)
  {
	std::cout << "Error CudaMalloc v1_d"  << " ";
  }


  cudaStatus = cudaMalloc( &v2_d, v2.size() * sizeof( int ) );
  if (cudaStatus != cudaSuccess)
  {
	std::cout << "Error CudaMalloc v2_d" << " ";
  }


  cudaStatus= cudaMemcpy(v1_d, v1.data(), v1.size() * sizeof( int ), cudaMemcpyHostToDevice );
  if (cudaStatus  != cudaSuccess)
  {
	  std::cout << "Error cudaMemcpy v1_d - HostToDevice" << " ";
  }



  cudaStatus = cudaMemcpy(v2_d, v2.data(), v2.size() * sizeof( int ), cudaMemcpyHostToDevice );
  if (cudaStatus != cudaSuccess)
   {
	   std::cout << "Error cudaMemcpy v2_d - HotToDevice" << " ";
  }

  cudaEventRecord(start, 0);  
  vecadd<<< 1, N >>>( v1_d, v2_d );
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  std::cout << "Timing (ms) = " << elapsedTime << " ";
  std::cout << std::endl;



  kernelStatus = cudaGetLastError();
   if ( kernelStatus != cudaSuccess )
   {
	   std::cout << "CUDA Error"<< cudaGetErrorString(kernelStatus) << " ";
  }


  cudaStatus = cudaMemcpy(v2.data(),v2_d, v2.size() * sizeof( int ), cudaMemcpyDeviceToHost );
  if (cudaStatus != cudaSuccess) 
  {
	std::cout << "Error Cuda Memcpy v2_d DeviceToHost"  << " " ;
  }
 
 
//  for (size_t idex = 0; idex < v1.size(); idex++)
//    std::cout <<   v2[idex] << " ";
//    std::cout << std::endl;



  cudaFree( v1_d );
  cudaFree( v2_d );

  return 0;
}
