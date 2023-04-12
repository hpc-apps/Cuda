#include <opencv2/opencv.hpp>

#include <vector>
#include <chrono>

int main()
{
  cv::Mat m_in = cv::imread("in.jpg", cv::IMREAD_UNCHANGED );
  auto rgb = m_in.data;
  
  std::vector< unsigned char > g( m_in.rows * m_in.cols );
  cv::Mat m_out( m_in.rows, m_in.cols, CV_8UC1, g.data() );

  auto start = std::chrono::system_clock::now();

  #pragma omp parallel for
  for( std::size_t j = 0 ; j < m_in.rows ; ++j )
    {
      for( std::size_t i = 0 ; i < m_in.cols ; ++i )
	{
	  g[ j * m_in.cols + i ] = (
			 307 * rgb[ 3 * ( j * m_in.cols + i ) ]
		       + 604 * rgb[ 3 * ( j * m_in.cols + i ) + 1 ]
		       + 113 * rgb[  3 * ( j * m_in.cols + i ) + 2 ]
		       ) / 1024;
	}
    }

  auto stop = std::chrono::system_clock::now();

  auto duration = stop - start;
  auto ms = std::chrono::duration_cast< std::chrono::milliseconds >( duration ).count();

  std::cout << ms << " ms" << std::endl;
  
  cv::imwrite( "out.jpg", m_out );
  
  return 0;
}
