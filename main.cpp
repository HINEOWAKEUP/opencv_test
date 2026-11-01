#include <iostream>
#include <opencv2/opencv.hpp>

int main()
{
    std::cout << "OpenCV version: "
              << CV_VERSION << std::endl;

    cv::Mat image(400, 600, CV_8UC3,
                  cv::Scalar(0, 0, 0));

    cv::putText(image,
                "OpenCV C++ is working!",
                cv::Point(40, 200),
                cv::FONT_HERSHEY_SIMPLEX,
                1.0,
                cv::Scalar(0, 255, 0),
                2);

    cv::imwrite("test.png", image);

    return 0;
}
