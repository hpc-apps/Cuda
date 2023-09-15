#include <opencv2/opencv.hpp>

#include <vector>
#include <chrono>

int main()
{
  cv::Mat m_in = cv::imread("in.jpg", cv::IMREAD_UNCHANGED );
  auto rgb = m_in.data;
  
  std::vector< unsigned char > g( m_in.rows * m_in.cols );
  cv::Mat m_out( m_in.rows, m_in.cols, CV_8UC1, g.data() );


//for( std::size_t j = 0 ; j < m_in.rows ; ++j )
//	for( std::size_t i = 0 ; i < m_in.cols ; ++i )
//		g[ j * m_in.cols + i ] = 0;

//rgb[ 3 * ( j * m_in.cols + i ) ]
//rgb[ 3 * ( j * m_in.cols + i ) + 1 ] 
//rgb[  3 * ( j * m_in.cols + i ) + 2 ]

  
  cv::imwrite( "out.jpg", m_out );
  
  return 0;
}
