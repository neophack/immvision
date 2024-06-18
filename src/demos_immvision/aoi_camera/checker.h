//
// Created by nn on 4/3/24.
//

#ifndef IMMVISION_CHECKER_H
#define IMMVISION_CHECKER_H
#include <thread>
#include <atomic>
#include <opencv2/opencv.hpp>
#include "appstate.h"


// Function to detect circles in an image
void detectCircles(cv::Mat image, std::vector<std::vector<float>>& circle_info) {
    // Convert image to HSV color space
    cv::Mat hsv;
    if(image.empty()){
        return;
    }
    if (image.rows <= 0 || image.cols <= 0) {
        std::cout << "Could not open or find the image!\n";
        return;
    }
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    // Define the HSV color range for red
    cv::Mat mask1, mask2;
    cv::inRange(hsv, cv::Scalar(0, 70, 50), cv::Scalar(10, 255, 255), mask1);
    cv::inRange(hsv, cv::Scalar(170, 70, 50), cv::Scalar(180, 255, 255), mask2);

    // Merge the two color range masks
    cv::Mat mask = mask1 | mask2;

    // Perform opening operation to remove noise
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

    // Find contours in the mask
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Clear previous circle information
    circle_info.clear();
    std::vector<cv::Point> approx;
    for (int i = 0; i < contours.size(); i++) {
        // Approximate each contour as a polygon
        cv::approxPolyDP(cv::Mat(contours[i]), approx, cv::arcLength(cv::Mat(contours[i]), true) * 0.02, true);

        // If the number of approximated points is greater than 6, assume a circle is found
        if (approx.size() > 6) {
            // Calculate the area of the contour
            float area = cv::contourArea(contours[i]);

            // Calculate the bounding rectangle of the contour
            cv::Rect r = cv::boundingRect(contours[i]);

            cv::Moments moments = cv::moments(contours[i]);
            cv::Point2f centroid(moments.m10 / moments.m00, moments.m01 / moments.m00);

            // Calculate the area of the rectangle
            int rect_area = r.height * r.width;

            // Check the area ratio to determine if it's a circle
            float ratio = float(area) / rect_area;
            if (ratio > 0.5) {
                // cv::circle(image, cv::Point(r.x + r.width / 2, r.y + r.height / 2), (r.width + r.height) / 4,
                //            cv::Scalar(0, 255, 0), 2);
                // cv::circle(image, cv::Point(r.x + r.width / 2, r.y + r.height / 2), 5, cv::Scalar(255, 0, 0), 5);

                // Store circle information
                circle_info.emplace_back(std::vector<float>({centroid.x, centroid.y, r.width, r.height}));
            }
        }
    }
}


int getLargerCircleIfRadiusDouble(std::vector<std::vector<float>> circle_info) {
    // 确保至少有两个圆
    if (circle_info.size() != 2) {
        return -1;
    }

    // 获取两个圆的信息
    std::vector<float> circle1 = circle_info[0];
    std::vector<float> circle2 = circle_info[1];

    // 计算两个圆的半径
    int radius1 = circle1[2] / 2;
    int radius2 = circle2[2] / 2;

    // 检查半径是否接近两倍
    if (std::abs(static_cast<float>(radius1) / radius2 - 2.0) <= 0.5 ||
        std::abs(static_cast<float>(radius2) / radius1 - 2.0) <= 0.5) {
        // 如果半径接近两倍，返回半径较大的圆的索引
        if (radius1 > radius2) {
            return 0;
        } else {
            return 1;
        }
    } else {
        // 如果半径不接近两倍，返回-1
        return -1;
    }
}

float calculateDistance(const std::vector<float>& circle1, const std::vector<float>& circle2) {
    float x1 = circle1[0];
    float y1 = circle1[1];
    float x2 = circle2[0];
    float y2 = circle2[1];

    float distance = std::sqrt(std::pow(x2 - x1, 2) + std::pow(y2 - y1, 2));
    return distance;
}

// Function to draw the result with a larger circle and coordinate system
void drawResult(cv::Mat& image, const std::vector<std::vector<float>>& circle_info, int largerCircleIndex) {
    // Draw coordinate system
    std::vector<float> largerCircle = circle_info[largerCircleIndex];
    std::vector<float> circle = circle_info[1 - largerCircleIndex];
    cv::line(image, cv::Point(largerCircle[0], largerCircle[1]), cv::Point(circle[0], circle[1]),
             cv::Scalar(255, 255, 255), 5);
    cv::Point v = cv::Point(circle[0] - largerCircle[0], circle[1] - largerCircle[1]);
    cv::Point v_perpendicular = cv::Point(-v.y, v.x);

    // Choose a scale length (e.g., image width or height)
    float scale = image.size().width;

    // Calculate a point on the perpendicular line
    cv::Point point_on_perpendicular_line = cv::Point(largerCircle[0] + scale * v_perpendicular.x,
                                                      largerCircle[1] + scale * v_perpendicular.y);
    // Draw the perpendicular line
    cv::line(image, cv::Point(largerCircle[0], largerCircle[1]), point_on_perpendicular_line,
             cv::Scalar(255, 255, 255), 5);


}

std::vector<cv::Point> rotatedRectangle( const cv::Rect &rect, const std::vector<float> &largerCircle, const cv::Point &v) {
    float rotationAngle = -std::atan2(v.y, v.x) * 180 / CV_PI;
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(cv::Point(largerCircle[0], largerCircle[1]),
                                                     rotationAngle,
                                                     1.0);
    std::vector<cv::Point2f> srcPoints = {cv::Point2f(rect.x, rect.y),
                                          cv::Point2f(rect.x + rect.width, rect.y),
                                          cv::Point2f(rect.x + rect.width, rect.y + rect.height),
                                          cv::Point2f(rect.x, rect.y + rect.height)};
    std::vector<cv::Point2f> dstPoints(4);
    cv::transform(srcPoints, dstPoints, rotationMatrix);
    std::vector<cv::Point> rotatedRectPoints;
    for (const auto &point: dstPoints) {
        rotatedRectPoints.push_back(cv::Point(static_cast<int>(point.x), static_cast<int>(point.y)));
        // std::cout<<"point.x point.y :"<<point.x<<" "<< point.y <<std::endl;
    }

    return rotatedRectPoints;
}

cv::Mat cropAndRotateImage(cv::Mat& image, const cv::Rect &boundingRect, float rotationAngle) {
    cv::Mat croppedImage = image(boundingRect);
    cv::Mat rotatedImage;
    cv::Mat rotationMatrix2 = cv::getRotationMatrix2D(
            cv::Point(croppedImage.size().width / 2, croppedImage.size().height / 2), -rotationAngle,
            1.0);
    cv::warpAffine(croppedImage, rotatedImage, rotationMatrix2, croppedImage.size(), cv::INTER_LINEAR);
    return rotatedImage;
}

// cv::Mat binarizeImage(AppState &appState, cv::Mat &image,int ind) {
//     cv::Mat binaryImage;
//     cv::Mat grayImage;
//     if(!image.empty()){
//         cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
//     }
//     cv::threshold(grayImage, binaryImage, appState.measure_boxes[ind].bin_threshold, 255, cv::THRESH_BINARY);


//     return binaryImage;
// }
cv::Mat binarizeImage(AppState &appState, cv::Mat &image, int ind) {
    cv::Mat binaryImage;
    cv::Mat grayImage;

    if (!image.empty()) {
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    }

    // cv::Mat edges1, edges2, edges3;
    // cv::Canny(grayImage, edges1, 20, 90); // 第一个边缘检测算法
    // cv::Sobel(grayImage, edges2, CV_8U, 1, 0, 3); // 第二个边缘检测算法
    // cv::Laplacian(grayImage, edges3, CV_8U); // 第三个边缘检测算法

    // // 将多个边缘图像叠加
    // cv::Mat edges = edges1 | edges2 | edges3;

    // // 统计边缘像素值的平均值
    // cv::Scalar meanPixelValue = cv::mean(grayImage, edges);

    // 使用平均像素值作为二值化的阈值
    cv::threshold(grayImage, binaryImage, appState.measure_boxes[ind].bin_threshold, 255, cv::THRESH_BINARY);

    return binaryImage;
}


// void findPins(AppState &appState, cv::Mat &binaryImage, int ind) {
//     std::vector<std::vector<cv::Point> > contours;
//     cv::Mat bimage = binaryImage.clone();
//         // Perform opening operation to remove noise
//     cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
//     cv::morphologyEx(bimage, bimage, cv::MORPH_OPEN, kernel);

//     cv::findContours(bimage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
//     appState.pin_info[ind].clear();
//     std::vector<cv::Point> approx;
//     for (int i = 0; i < contours.size(); i++) {
//         cv::approxPolyDP(cv::Mat(contours[i]), approx, cv::arcLength(cv::Mat(contours[i]), true) * 0.02,
//                          true);
//         if (approx.size() > 3) {
//             float area = cv::contourArea(contours[i]);
//             cv::Rect r = cv::boundingRect(contours[i]);
//             int rect_area = r.height * r.width;
//             float ratio = float(area) / rect_area;
//             // std::cout<<"float(area) / rect_area:"<<float(area) <<" "<< rect_area<<std::endl;
//             if (ratio > 0.2 && rect_area<200) {
//                 appState.pin_info[ind].emplace_back(
//                         std::vector<int>({r.x + r.width / 2, r.y + r.height / 2, r.width, r.height}));
//             }
//         }
//     }
// }

void findPins(AppState &appState, cv::Mat &binaryImage, int ind) {
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(binaryImage.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    appState.pin_info[ind].clear();
    std::vector<cv::Point> approx;

    std::vector<int> rect_areas;
    for (int i = 0; i < contours.size(); i++) {
        cv::approxPolyDP(cv::Mat(contours[i]), approx, cv::arcLength(cv::Mat(contours[i]), true) * 0.02,
                         true);
        if (approx.size() > 3) {
            cv::Rect r = cv::boundingRect(contours[i]);
            int rect_area = r.height * r.width;
            rect_areas.push_back(rect_area);
        }
    }

    // Calculate mean and standard deviation of rect_areas
    double sum = std::accumulate(rect_areas.begin(), rect_areas.end(), 0.0);
    double mean = sum / rect_areas.size();

    double sq_sum = std::inner_product(rect_areas.begin(), rect_areas.end(), rect_areas.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / rect_areas.size() - mean * mean);

    for (int i = 0; i < contours.size(); i++) {
        cv::approxPolyDP(cv::Mat(contours[i]), approx, cv::arcLength(cv::Mat(contours[i]), true) * 0.02,
                         true);
        if (approx.size() > 3) {
            float area = cv::contourArea(contours[i]);
            cv::Moments moments = cv::moments(contours[i]);
            cv::Point2f centroid(moments.m10 / moments.m00, moments.m01 / moments.m00);

            float perimeter = cv::arcLength(contours[i], true);
            float diameter = 2 * std::sqrt(area / CV_PI); // Calculate diameter
            cv::Rect r = cv::boundingRect(contours[i]);
            int rect_area = r.height * r.width;
            float ratio = area / rect_area;

            // Check if rect_area is within 3 standard deviations of the mean
            if (ratio > 0.2 && rect_area >20 && rect_area < 400 && std::abs(rect_area - mean) <= 3 * stdev) {
                appState.pin_info[ind].emplace_back(
                        std::vector<float>({centroid.x, centroid.y, static_cast<float>(moments.m00), static_cast<float>(moments.m00)}));
            }
        }
    }
}


class Checker {
private:
    std::thread checkerThread;

public:
    std::atomic<bool> isRunning;

    Checker(): isRunning(false)  {

    }
    ~Checker() {
        Stop(); // Call Stop() in the destructor to ensure thread termination
    }

    void Start(AppState &appState) {
        if (!isRunning) {
            isRunning = true;
            checkerThread = std::thread(&Checker::CheckerThreadFunc, this, std::ref(appState));
            checkerThread.detach();
        }
    }

    void Stop() {
        if (isRunning) {
            isRunning = false;
            // checkerThread.join();

        }
    }

    struct Circle {
        int x;
        int y;
        int radius;
    };

    bool _isCircleMoved(const Circle& circle1, const Circle& circle2, int threshold) {
        int deltaX = std::abs(circle1.x - circle2.x);
        int deltaY = std::abs(circle1.y - circle2.y);
        int deltaRadius = std::abs(circle1.radius - circle2.radius);
        // std::cout<<"_isCircleMoved:"<<deltaX<<" "<<deltaY<<" "<<deltaRadius<<" "<<std::endl;
        return (deltaX > threshold) || (deltaY > threshold) || (deltaRadius > threshold);
    }

    int _circleMoved(const Circle& circle1, const Circle& circle2) {
        int deltaX = std::abs(circle1.x - circle2.x);
        int deltaY = std::abs(circle1.y - circle2.y);
        int deltaRadius = std::abs(circle1.radius - circle2.radius);
        // std::cout<<"_isCircleMoved:"<<deltaX<<" "<<deltaY<<" "<<deltaRadius<<" "<<std::endl;
        return deltaX + deltaY + deltaRadius ;
    }

    void CheckerThreadFunc(AppState& appState) {
        std::queue<Circle> circle_history;
        std::vector<std::vector<int>> circle_history2;
        static int lastMove=0;
        while (isRunning) {
            if (!appState.image.empty()) {
                std::vector<std::vector<float>> circle_info;
                detectCircles(appState.image, circle_info);
                appState.image.copyTo(appState.imageDraw);

                if (circle_info.size() == 2) {
                    int largerCircleIndex = getLargerCircleIfRadiusDouble(circle_info);
                    if (largerCircleIndex < 0 || largerCircleIndex >= circle_info.size()) {
                        continue;  // Skip this iteration if largerCircleIndex is invalid
                    }

                    Circle largerCircle;
                    largerCircle.x = circle_info[largerCircleIndex][0];
                    largerCircle.y = circle_info[largerCircleIndex][1];
                    largerCircle.radius = (circle_info[largerCircleIndex][2]+circle_info[largerCircleIndex][2])/2;

                    std::vector<float> circle = circle_info[1 - largerCircleIndex];

                    circle_history.push(largerCircle);
                    if (circle_history.size() > appState.qsize) {
                        circle_history.pop();
                    }

                    int largerCircleMoved = 0;
                    if (circle_history.size() == appState.qsize) {
                        std::queue<Circle> prevCircles = circle_history;

                        // Check if the queue is not empty before popping
                        if (!prevCircles.empty()) {
                            prevCircles.pop();
                        }

                        while (!prevCircles.empty()) {
                            Circle prevCircle = prevCircles.front();
                            prevCircles.pop();
                            int move = _circleMoved(largerCircle, prevCircle);

                            if (_isCircleMoved(largerCircle, prevCircle, appState.noiseThreshold) && lastMove < move) {
                                largerCircleMoved = 1;
                                break;
                            }
                            lastMove = move;

                        }
                    }

                    appState.largerCircleMoved = largerCircleMoved;
                }
                if (circle_info.size() < 2){
                    appState.largerCircleMoved = -1;
                }

            }
            usleep(100000);
        }
    }


};


#endif //IMMVISION_CHECKER_H
