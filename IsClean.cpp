#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>

struct CameraHealthStatus
{
    bool RainDetected = false;
    bool DropletsDetected = false;
    bool HazeDetected = false;

    double RainConfidence = 0.0;
    double DropletConfidence = 0.0;
    double HazeConfidence = 0.0;

    bool IsClean() const
    {
        return !RainDetected &&
               !DropletsDetected &&
               !HazeDetected;
    }
};

// --------------------------------------------------
// Haze detection (preliminary image-quality heuristic)
// --------------------------------------------------
double DetectHaze(const cv::Mat& image)
{
    cv::Mat gray;
    cv::Mat hsv;

    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    const int rows = gray.rows;
    const int cols = gray.cols;

    const int regions = 4;
    double contrastSum = 0.0;
    double edgeSum = 0.0;
    double brightnessSum = 0.0;

    for (int i = 0; i < regions; ++i)
    {
        int y1 = i * rows / regions;
        int y2 = (i + 1) * rows / regions;

        cv::Mat region = gray(
            cv::Range(y1, y2),
            cv::Range(0, cols));

        cv::Scalar mean, stddev;
        cv::meanStdDev(region, mean, stddev);

        contrastSum += stddev[0];

        cv::Mat edges;
        cv::Canny(region, edges, 50, 150);

        double edgeDensity =
            static_cast<double>(cv::countNonZero(edges)) /
            edges.total();

        edgeSum += edgeDensity;

        brightnessSum += mean[0];
    }

    double averageContrast = contrastSum / regions;
    double averageEdgeDensity = edgeSum / regions;
    double averageBrightness = brightnessSum / regions;

    // Preliminary fog indicators
    double contrastScore =
        std::clamp((45.0 - averageContrast) / 45.0, 0.0, 1.0);

    double edgeScore =
        std::clamp((0.06 - averageEdgeDensity) / 0.06, 0.0, 1.0);

    double brightnessScore =
        std::clamp((averageBrightness - 100.0) / 100.0, 0.0, 1.0);

    double confidence =
        0.45 * contrastScore +
        0.35 * edgeScore +
        0.20 * brightnessScore;

    return confidence;
}

// --------------------------------------------------
// Rain streak detection (directional edge heuristic)
// --------------------------------------------------
double DetectRain(const cv::Mat& image)
{
    cv::Mat gray;
    cv::Mat edges;

    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
         -1,  0,  1,
         -1,  0,  1,
         -1,  0,  1);

    cv::Mat filtered;
    cv::filter2D(gray, filtered, CV_32F, kernel);

    cv::convertScaleAbs(filtered, edges);

    cv::threshold(edges, edges, 80, 255, cv::THRESH_BINARY);

    int edgePixels = cv::countNonZero(edges);
    double totalPixels = static_cast<double>(edges.total());

    return std::min(
        static_cast<double>(edgePixels) / (totalPixels * 0.15),
        1.0
    );
}

// --------------------------------------------------
// Water droplet detection (contour heuristic)
// --------------------------------------------------
double DetectDroplets(const cv::Mat& image)
{
    cv::Mat gray;
    cv::Mat blurred;
    cv::Mat edges;

    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    cv::GaussianBlur(
        gray, blurred,
        cv::Size(5, 5), 1.5
    );

    cv::Canny(blurred, edges, 50, 150);

    std::vector<std::vector<cv::Point>> contours;

    cv::findContours(
        edges,
        contours,
        cv::RETR_EXTERNAL,
        cv::CHAIN_APPROX_SIMPLE
    );

    int dropletCandidates = 0;

    for (const auto& contour : contours)
    {
        double area = cv::contourArea(contour);

        if (area < 30 || area > 3000)
            continue;

        double perimeter = cv::arcLength(contour, true);

        if (perimeter <= 0)
            continue;

        double circularity =
            4.0 * CV_PI * area /
            (perimeter * perimeter);

        if (circularity > 0.65)
            dropletCandidates++;
    }

    double confidence =
        std::min(dropletCandidates / 20.0, 1.0);

    return confidence;
}

// --------------------------------------------------
// Analyze image
// --------------------------------------------------
CameraHealthStatus AnalyzeImage(const cv::Mat& image)
{
    CameraHealthStatus status;

    status.RainConfidence = DetectRain(image);
    status.DropletConfidence = DetectDroplets(image);
    status.HazeConfidence = DetectHaze(image);

    // Initial thresholds for experimentation.
    status.RainDetected = status.RainConfidence > 0.30;
    status.DropletsDetected = status.DropletConfidence > 0.30;
    status.HazeDetected = status.HazeConfidence > 0.50;

    return status;
}

// --------------------------------------------------
// Main
// --------------------------------------------------
int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cout << "Usage: IsClean <image_filename>\n";
        return 1;
    }

    std::string filename = argv[1];

    cv::Mat image = cv::imread(filename);

    if (image.empty())
    {
        std::cerr << "Error: Cannot open image: "
                  << filename << std::endl;
        return 1;
    }

    CameraHealthStatus status = AnalyzeImage(image);

    std::cout << "\n========== Camera Image Analysis ==========\n";

    std::cout << "Image: " << filename << "\n";

    std::cout << "Rain: "
              << (status.RainDetected ? "DETECTED" : "NOT DETECTED")
              << " (confidence: " << status.RainConfidence << ")\n";

    std::cout << "Water droplets: "
              << (status.DropletsDetected ? "DETECTED" : "NOT DETECTED")
              << " (confidence: " << status.DropletConfidence << ")\n";

    std::cout << "Haze: "
              << (status.HazeDetected ? "DETECTED" : "NOT DETECTED")
              << " (confidence: " << status.HazeConfidence << ")\n";

    std::cout << "-------------------------------------------\n";

    std::cout << "Camera status: "
              << (status.IsClean() ? "CLEAN" : "DEGRADED")
              << "\n";

    std::cout << "===========================================\n";

    return 0;
}
