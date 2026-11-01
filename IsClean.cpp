#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
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
// Dark Channel Prior
// --------------------------------------------------
cv::Mat DarkChannel(const cv::Mat& image, int patchSize = 15)
{
    std::vector<cv::Mat> channels;
    cv::split(image, channels);

    cv::Mat minimum;
    cv::min(channels[0], channels[1], minimum);
    cv::min(minimum, channels[2], minimum);

    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT,
        cv::Size(patchSize, patchSize));

    cv::Mat dark;
    cv::erode(minimum, dark, kernel);

    return dark;
}

// --------------------------------------------------
// Estimate atmospheric light
// --------------------------------------------------
cv::Vec3f EstimateAtmosphericLight(
    const cv::Mat& image,
    const cv::Mat& dark)
{
    struct Candidate
    {
        float Dark;
        int Index;
    };

    std::vector<Candidate> candidates;

    for (int y = 0; y < dark.rows; ++y)
    {
        const float* row = dark.ptr<float>(y);

        for (int x = 0; x < dark.cols; ++x)
        {
            candidates.push_back({
                row[x],
                y * dark.cols + x
            });
        }
    }

    int count = std::max(
        1,
        static_cast<int>(candidates.size() * 0.001));

    std::partial_sort(
        candidates.begin(),
        candidates.begin() + count,
        candidates.end(),
        [](const Candidate& a, const Candidate& b)
        {
            return a.Dark > b.Dark;
        });

    cv::Vec3f atmosphericLight(0, 0, 0);
    float brightest = -1.0f;

    for (int i = 0; i < count; ++i)
    {
        int index = candidates[i].Index;

        int y = index / image.cols;
        int x = index % image.cols;

        cv::Vec3f pixel = image.at<cv::Vec3f>(y, x);

        float brightness = pixel[0] + pixel[1] + pixel[2];

        if (brightness > brightest)
        {
            brightest = brightness;
            atmosphericLight = pixel;
        }
    }

    return atmosphericLight;
}

// --------------------------------------------------
// Estimate transmission
// --------------------------------------------------
cv::Mat EstimateTransmission(
    const cv::Mat& image,
    const cv::Vec3f& atmosphericLight,
    float omega = 0.95f)
{
    cv::Mat normalized = image.clone();

    for (int y = 0; y < image.rows; ++y)
    {
        for (int x = 0; x < image.cols; ++x)
        {
            cv::Vec3f& pixel = normalized.at<cv::Vec3f>(y, x);

            for (int c = 0; c < 3; ++c)
            {
                pixel[c] /= std::max(
                    atmosphericLight[c], 0.1f);
            }
        }
    }

    cv::Mat dark = DarkChannel(normalized);

    cv::Mat transmission =
        1.0f - omega * dark;

    // Smooth transmission to reduce patch artifacts.
    cv::Mat refined;

    cv::bilateralFilter(
        transmission,
        refined,
        9,
        0.1,
        7.0);

    return refined;
}

// --------------------------------------------------
// Remove fog
// --------------------------------------------------
cv::Mat Dehaze(const cv::Mat& input)
{
    cv::Mat image;

    input.convertTo(image, CV_32FC3, 1.0 / 255.0);

    cv::Mat dark = DarkChannel(image);

    cv::Vec3f atmosphericLight =
        EstimateAtmosphericLight(image, dark);

    cv::Mat transmission =
        EstimateTransmission(image, atmosphericLight);

    cv::Mat result(image.size(), CV_32FC3);

    const float minimumTransmission = 0.10f;

    for (int y = 0; y < image.rows; ++y)
    {
        for (int x = 0; x < image.cols; ++x)
        {
            cv::Vec3f original =
                image.at<cv::Vec3f>(y, x);

            float t = std::max(
                transmission.at<float>(y, x),
                minimumTransmission);

            cv::Vec3f restored;

            for (int c = 0; c < 3; ++c)
            {
                restored[c] =
                    (original[c] - atmosphericLight[c]) / t
                    + atmosphericLight[c];

                restored[c] = std::clamp(
                    restored[c], 0.0f, 1.0f);
            }

            result.at<cv::Vec3f>(y, x) = restored;
        }
    }

    cv::Mat output;
    result.convertTo(output, CV_8UC3, 255.0);

    return output;
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

    // Analyze original image.
    CameraHealthStatus status = AnalyzeImage(image);

    std::cout << "\n========== CAMERA HEALTH ==========\n";

    std::cout << "Image: " << filename << "\n";

    std::cout << "Rain: "
              << (status.RainDetected ? "DETECTED" : "NOT DETECTED")
              << "\n";

    std::cout << "Water droplets: "
              << (status.DropletsDetected ? "DETECTED" : "NOT DETECTED")
              << "\n";

    std::cout << "Fog: "
              << (status.HazeDetected ? "DETECTED" : "NOT DETECTED")
              << "\n";

    std::cout << "Camera status: "
              << (status.IsClean() ? "CLEAN" : "DEGRADED")
              << "\n";

    // Only create a comparison image if degraded.
    if (!status.IsClean())
    {
        std::cout << "\nDehazing image...\n";

        cv::Mat cleared = Dehaze(image);

        // Place original on left and enhanced on right.
        cv::Mat comparison;

        cv::hconcat(image, cleared, comparison);

        // Add labels.
        cv::putText(
            comparison,
            "ORIGINAL",
            cv::Point(20, 40),
            cv::FONT_HERSHEY_SIMPLEX,
            1.0,
            cv::Scalar(0, 255, 255),
            2);

        cv::putText(
            comparison,
            "DEHAZED",
            cv::Point(image.cols + 20, 40),
            cv::FONT_HERSHEY_SIMPLEX,
            1.0,
            cv::Scalar(0, 255, 0),
            2);

        std::filesystem::path inputPath(filename);

        std::string outputFilename =
            inputPath.stem().string() + "_comparison.jpg";

        if (!cv::imwrite(outputFilename, comparison))
        {
            std::cerr << "Error: Cannot save comparison image\n";
            return 1;
        }

        std::cout << "Output: " << outputFilename << "\n";
    }
    else
    {
        std::cout << "\nImage is clean. No processing required.\n";
    }

    std::cout << "===================================\n";

    return 0;
}