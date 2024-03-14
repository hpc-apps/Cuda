#include <opencv2/opencv.hpp>

#include <vector>
#include <chrono>

int main()
{
  cv::Mat m_in = cv::imread("in.jpg", cv::IMREAD_UNCHANGED );
  auto rgb = m_in.data;
  
  std::vector< unsigned char > g( m_in.rows * m_in.cols );
  cv::Mat m_out( m_in.rows, m_in.cols, CV_8UC1, g.data() );




  
  cv::imwrite( "out.jpg", m_out );
  
  return 0;
}
