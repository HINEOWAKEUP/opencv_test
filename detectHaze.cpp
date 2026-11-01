#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    cv::Mat Image = cv::imread("road.jpg");

    if (Image.empty())
        return -1;

    cv::Mat Gray;
    cv::cvtColor(Image, Gray, cv::COLOR_BGR2GRAY);

    cv::Scalar Mean, StdDev;
    cv::meanStdDev(Gray, Mean, StdDev);

    std::cout << "Image contrast: "
              << StdDev[0] << std::endl;

    if (StdDev[0] < 35.0)
        std::cout << "Possible low contrast / haze"
                  << std::endl;

    return 0;
}
