

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <thread>
#include <iostream>
#include <filesystem>
#include <unordered_set>
#include <mutex>
#include <string>
#include <fstream>
#include <vector>
#include <cmath>
#include <limits>
#include <utility>

#include <nlohmann/json.hpp>

#include "immvision/immvision.h"
#include "immvision/internal/misc/portable_file_dialogs.h"
#include "immvision/internal/imgui/image_widgets.h"

#include "imgui_freetype.h"
#include "hello_imgui/hello_imgui.h"

#include "checker.h"
#include "database.h"
#include "scaner.h"
#include "upload.h"

CameraController cameraController; // Moved cameraController declaration here
Checker checker;
DatabaseHandler dbHandler("database.db");

bool scannerIsRunning = false;

std::string ResourcesDir()
{
    std::filesystem::path this_file(__FILE__);
    return (this_file.parent_path().parent_path() / "resources").string();
}

enum class Orientation
{
    Horizontal,
    Vertical
};

float calculateDistance(const cv::Point2f &p1, const cv::Point2f &p2)
{
    float dx = p2.x - p1.x;
    float dy = p2.y - p1.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Function to calculate the angle between three points
float calculateAngle(const cv::Point2f &p1, const cv::Point2f &p2, const cv::Point2f &p3, bool clockwise = false)
{
    cv::Point2f v1 = p2 - p1;
    cv::Point2f v2 = p3 - p1;
    float angle = std::atan2(v2.y, v2.x) - std::atan2(v1.y, v1.x);
    angle = angle * 180.0 / CV_PI; // Convert to degree
    if (angle < 0)
        angle += 360.0; // Make it positive

    if (clockwise)
    {
        angle = 360.0 - angle; // Reverse the angle if clockwise
    }

    return angle;
}
float calculateAngle(const cv::Point2f &p0, const cv::Point2f &p1, const cv::Point2f &p2, const cv::Point2f &p3, bool clockwise=false)
{
    cv::Point2f v1 = p1 - p0;
    cv::Point2f v2 = p3 - p2;

    float angleRad1 = std::atan2(v1.y, v1.x);
    float angleRad2 = std::atan2(v2.y, v2.x);

    // Convert the angle to [0, 2π)
    if (angleRad1 < 0)
    {
        angleRad1 += 2 * CV_PI;
    }
    if (angleRad2 < 0)
    {
        angleRad2 += 2 * CV_PI;
    }

    float angleRad;
    if (clockwise)
    {
        angleRad = angleRad1 - angleRad2;
        if (angleRad < 0)
        {
            angleRad += 2 * CV_PI;
        }
    }
    else
    {
        angleRad = angleRad2 - angleRad1;
        if (angleRad < 0)
        {
            angleRad += 2 * CV_PI;
        }
    }

    // Convert the angle to degrees
    float angleDeg = angleRad * 180.0 / CV_PI;

    return angleDeg;
}


struct PointInd
{
    float x, y;
    int index; // 点的编号
};

float distance(const PointInd &p1, const PointInd &p2)
{
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

std::vector<std::pair<int, int>> prim(const std::vector<PointInd> &points)
{
    int n = points.size();
    std::vector<bool> visited(n, false);
    std::vector<int> parent(n, -1);
    std::vector<float> minDistance(n, std::numeric_limits<float>::max());
    if (n > 0)
    {
        minDistance[0] = 0;
    }
    for (int i = 0; i < n; ++i)
    {
        int u = -1;
        for (int j = 0; j < n; ++j)
        {
            if (!visited[j] && (u == -1 || minDistance[j] < minDistance[u]))
            {
                u = j;
            }
        }

        if (u == -1)
            break;

        visited[u] = true;

        for (int v = 0; v < n; ++v)
        {
            float dist = distance(points[u], points[v]);
            if (!visited[v] && dist < minDistance[v])
            {
                minDistance[v] = dist;
                parent[v] = u;
            }
        }
    }

    std::vector<std::pair<int, int>> matching;
    for (int i = 1; i < n; ++i)
    {
        if (parent[i] != -1)
        {
            matching.push_back(std::make_pair(points[parent[i]].index, points[i].index));
        }
    }

    return matching;
}

std::vector<int> createChain(const std::vector<std::pair<int, int>> &matching)
{
    std::vector<int> chain;
    if (matching.empty())
    {
        return chain;
    }

    int start = matching[0].first;
    int end = matching[0].second;

    chain.push_back(start);
    chain.push_back(end);

    while (true)
    {
        bool found = false;
        for (const auto &pair : matching)
        {
            if (pair.first == end)
            {
                end = pair.second;
                chain.push_back(end);
                found = true;
                break;
            }
        }
        if (!found)
        {
            break;
        }
    }
    chain.clear();
    std::unordered_set<int> connectedIndicesSet;
    while (true)
    {
        bool found = false;
        for (const auto &pair : matching)
        {
            if (pair.second == end && connectedIndicesSet.find(pair.second) == connectedIndicesSet.end())
            {
                connectedIndicesSet.insert(pair.second);
                chain.push_back(end);
                end = pair.first;

                found = true;
                break;
            }
            else if (pair.first == end && connectedIndicesSet.find(pair.second) == connectedIndicesSet.end())
            {
                connectedIndicesSet.insert(pair.second);
                chain.push_back(end);
                end = pair.second;

                found = true;
                break;
            }
        }
        if (!found)
        {
            chain.push_back(end);
            break;
        }
    }

    return chain;
}

std::vector<int> findMatchingRelationships2(const std::vector<std::vector<float>> &pin_info, bool reverse_point)
{
    std::vector<PointInd> points;
    // Find the matching relationships
    for (size_t i = 0; i < pin_info.size(); ++i)
    {
        points.push_back({pin_info[i][0], pin_info[i][1], int(i)});
    }

    std::vector<std::pair<int, int>> matching = prim(points);

    std::vector<int> chain = createChain(matching);

    // std::cout << "Matching:" << std::endl;
    // for (const auto &pair: matching) {
    //     std::cout << "(" << pair.first << ", " << pair.second << ")" << std::endl;
    // }
    // 现在，connectedIndices向量中包含了所有串联起来的pin端点的索引
    if (chain.size() > 0)
    {
        // 比较首尾两个点的坐标值
        cv::Point2f startPoint(pin_info[chain.front()][0], pin_info[chain.front()][1]);
        cv::Point2f endPoint(pin_info[chain.back()][0], pin_info[chain.back()][1]);

        // 如果终点的坐标值小于起始点的坐标值，则反转 connectedIndices 向量
        if ((endPoint.x < startPoint.x) || (endPoint.x == startPoint.x && endPoint.y < startPoint.y))
        {
            std::reverse(chain.begin(), chain.end());
        }
        if (reverse_point)
        {
            std::reverse(chain.begin(), chain.end());
        }
    }
    // for (int i = 0; i < chain.size(); ++i) {
    //     std::cout << "PointInd " << chain[i];
    //     if (i < chain.size() - 1) {
    //         std::cout << " -> ";
    //     }
    // }
    std::cout << std::endl;

    return chain;
}

// Function to draw nearest distances between pin points
void drawNearestDistances(cv::Mat &image, const std::vector<std::vector<float>> &pin_info,
                          const std::vector<int> connectedIndices)
{

    cv::Scalar color(255, 0, 0);
    int thickness = 1;
    int font = cv::FONT_HERSHEY_SIMPLEX;
    float fontScale = 0.3;

    std::vector<cv::Scalar> colors = {cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255)};
    int colorIndex = 0;
    if (connectedIndices.size() > 1)
        for (size_t i = 0; i < connectedIndices.size() - 1; ++i)
        {
            int ind = connectedIndices[i];
            int ind_next = connectedIndices[i + 1];
            cv::Point2f p1(pin_info[ind][0], pin_info[ind][1]);
            cv::Point2f p2(pin_info[ind_next][0], pin_info[ind_next][1]);
            float distance = calculateDistance(p1, p2);

            cv::line(image, p1, p2, colors[colorIndex], thickness);
            std::stringstream stream;
            stream << std::fixed << std::setprecision(1) << distance;
            std::string distanceText = stream.str();
            cv::putText(image, distanceText, (p1 + p2) / 2, font, fontScale, colors[colorIndex], thickness / 2);
        }

    for (size_t i = 0; i < connectedIndices.size(); ++i)
    {
        int ind = connectedIndices[i];
        cv::Point p1(pin_info[ind][0] + 5, pin_info[ind][1] - 5);
        cv::putText(image, std::to_string(i + 1), p1, font, fontScale, colors[colorIndex + 1], thickness / 2);
        cv::circle(image, cv::Point(pin_info[ind][0], pin_info[ind][1]),
                   (pin_info[ind][2] + pin_info[ind][3]) / 4,
                   cv::Scalar(0, 255, 0), 1);

        // printf("圆半径：%f %f，中心坐标：%d %d\n", float(pin_info[ind][0]), float(pin_info[ind][1]), pin_info[ind][2],
        //        pin_info[ind][3]);
    }
}

void drawMeasureDistances(cv::Mat &image, const std::vector<std::vector<float>> &pin_info,
                          const std::vector<int> connectedIndices,
                          std::vector<AppState::MEASURE_SETTING> &measure_settings, int ind, double scale)
{

    cv::Scalar color(255, 0, 0);
    int thickness = 1;
    int font = cv::FONT_HERSHEY_SIMPLEX;
    float fontScale = 0.3;

    std::vector<cv::Scalar> colors = {cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255)};
    int colorIndex = 0;

    for (size_t i = 0; i < measure_settings.size(); ++i)
    {
        auto measure_setting = measure_settings[i];

        if (measure_setting.box != ind)
        {
            continue;
        }
        if (measure_setting.pin0 >= connectedIndices.size() || measure_setting.pin1 >= connectedIndices.size())
        {
            measure_settings[i].distance = -1;
            continue;
        }

        int ind = connectedIndices[measure_setting.pin0];
        int ind_next = connectedIndices[measure_setting.pin1];
        cv::Point2f p1(pin_info[ind][0], pin_info[ind][1]);
        cv::Point2f p2(pin_info[ind_next][0], pin_info[ind_next][1]);
        float distance = calculateDistance(p1, p2) * scale;
        // std::cout << "distance:" << distance << std::endl;
        measure_settings[i].distance = distance;

        if (distance < measure_setting.minDistance || distance > measure_setting.maxDistance)
        {
            cv::line(image, p1, p2, cv::Scalar(0, 0, 255), thickness); // Red color
            std::stringstream stream;
            stream << std::fixed << std::setprecision(2) << distance;
            std::string distanceText = stream.str();
            cv::putText(image, distanceText, (p1 + p2) / 2, font, fontScale, cv::Scalar(0, 0, 255),
                        thickness / 2); // Red color
        }
        else
        {
            cv::line(image, p1, p2, colors[colorIndex], thickness);
            std::stringstream stream;
            stream << std::fixed << std::setprecision(2) << distance;
            std::string distanceText = stream.str();
            cv::putText(image, distanceText, (p1 + p2) / 2, font, fontScale, colors[colorIndex], thickness / 2);
        }
    }

    for (size_t i = 0; i < connectedIndices.size(); ++i)
    {
        int ind = connectedIndices[i];
        cv::Point p1(pin_info[ind][0] + 5, pin_info[ind][1] - 5);
        cv::putText(image, std::to_string(i + 1), p1, font, fontScale, colors[colorIndex + 1], thickness / 2);
        cv::circle(image, cv::Point(pin_info[ind][0], pin_info[ind][1]),
                   (pin_info[ind][2] + pin_info[ind][3]) / 4,
                   cv::Scalar(0, 255, 0), 1);

        printf("圆面积：%f %f，中心坐标：%f %f\n", float(pin_info[ind][2]), float(pin_info[ind][3]), pin_info[ind][0],
               pin_info[ind][1]);
    }
}

void drawMeasureAngles(cv::Mat &image, const std::vector<std::vector<float>> &pin_info,
                       const std::vector<int> connectedIndices,
                       std::vector<AppState::ANGLE_MEASURE_SETTING> &angle_measure_settings, int ind)
{

    cv::Scalar color(255, 0, 0);
    int thickness = 1;
    int font = cv::FONT_HERSHEY_SIMPLEX;
    float fontScale = 0.3;

    std::vector<cv::Scalar> colors = {cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255)};
    int colorIndex = 0;

    for (size_t i = 0; i < angle_measure_settings.size(); ++i)
    {
        auto angle_measure_setting = angle_measure_settings[i];

        if (angle_measure_setting.box != ind)
        {
            continue;
        }
        if (angle_measure_setting.pin0 >= connectedIndices.size() ||
            angle_measure_setting.pin1 >= connectedIndices.size() ||
            angle_measure_setting.pin2 >= connectedIndices.size() ||
            angle_measure_setting.pin3 >= connectedIndices.size())
        {
            angle_measure_settings[i].angle = -1;
            continue;
        }

        int ind0 = connectedIndices[angle_measure_setting.pin0];
        int ind1 = connectedIndices[angle_measure_setting.pin1];
        int ind2 = connectedIndices[angle_measure_setting.pin2];
        int ind3 = connectedIndices[angle_measure_setting.pin3];

        cv::Point2f p0(pin_info[ind0][0], pin_info[ind0][1]);
        cv::Point2f p1(pin_info[ind1][0], pin_info[ind1][1]);
        cv::Point2f p2(pin_info[ind2][0], pin_info[ind2][1]);
        cv::Point2f p3(pin_info[ind3][0], pin_info[ind3][1]);

        float angle = calculateAngle(p0, p1, p2, p3);
        // std::cout << "angle: " << angle << std::endl;
        angle_measure_settings[i].angle = angle;

        if (angle < angle_measure_setting.minAngle || angle > angle_measure_setting.maxAngle)
        {
            cv::line(image, p0, p1, cv::Scalar(0, 0, 255), thickness); // Red color
            cv::line(image, p2, p3, cv::Scalar(0, 0, 255), thickness); // Red color
            std::stringstream stream;
            stream << std::fixed << std::setprecision(2) << angle;
            std::string angleText = stream.str();
            cv::putText(image, angleText, (p0 + p1 + p2 + p3) / 4, font, fontScale, cv::Scalar(0, 0, 255),
                        thickness / 2); // Red color
        }
        else
        {
            cv::line(image, p0, p1, colors[colorIndex], thickness);
            cv::line(image, p2, p3, colors[colorIndex], thickness);
            std::stringstream stream;
            stream << std::fixed << std::setprecision(2) << angle;
            std::string angleText = stream.str();
            cv::putText(image, angleText, (p0 + p1 + p2 + p3) / 4, font, fontScale, colors[colorIndex], thickness / 2);
        }
    }

    for (size_t i = 0; i < connectedIndices.size(); ++i)
    {
        int ind = connectedIndices[i];
        cv::Point p1(pin_info[ind][0] + 5, pin_info[ind][1] - 5);
        cv::putText(image, std::to_string(i + 1), p1, font, fontScale, colors[colorIndex + 1], thickness / 2);
        cv::circle(image, cv::Point(pin_info[ind][0], pin_info[ind][1]),
                   (pin_info[ind][2] + pin_info[ind][3]) / 4,
                   cv::Scalar(0, 255, 0), 1);

        // printf("圆半径：%f %f，中心坐标：%d %d\n", float(pin_info[ind][2]), float(pin_info[ind][3]), pin_info[ind][0],
        //        pin_info[ind][1]);
    }
}

std::mutex globalVariableMutex;

void algorithmFunction(AppState &appState)
{
    try
    {
        // 访问或修改全局变量之前加锁
        std::lock_guard<std::mutex> lock(globalVariableMutex);
        appState.algorithmRunning = true;
        cv::Mat imageCap = appState.imageCap.clone();

        // cv::Mat imageCap32;
        //  cv::Mat imageCap;
        // if(cameraController.isRunning) {
        //     int numFrames = 10; // Number of frames to average

        //     // Accumulate frames
        //     for (int i = 0; i < numFrames; i++) {
        //         while(appState.largerCircleMoved != 0){
        //            usleep(10000);
        //         }
        //         cv::Mat frame = cameraController.GetFrame();
        //         if (frame.empty()) {
        //             // Handle empty frame
        //             appState.algorithmFinished = true;
        //             appState.algorithmRunning = false;
        //             return;
        //         }
        //         if (imageCap.empty())
        //         {
        //             imageCap32 = cv::Mat(frame.size(), CV_32FC3);
        //         }
        //         cv::add(imageCap32, frame, imageCap32, cv::noArray(), CV_32F);
        //     }

        //     // Average frames
        //     imageCap32 /= numFrames;
        //     imageCap32.convertTo(imageCap, CV_8UC3);
        // }else{
        //     imageCap = appState.imageCap.clone();
        // }
        if (!imageCap.empty())
        {
            // 进行畸变校正
            cv::Mat undistortedImage;
            cv::undistort(imageCap, undistortedImage, appState.intrinsics_in, appState.dist_coeffs);

            // Detect circles
            detectCircles(undistortedImage, appState.circle_info);
            if (appState.circle_info.size() != 2)
            {
                appState.algorithmFinished = true;
                appState.algorithmRunning = false;
                return;
            }

            int largerCircleIndex = getLargerCircleIfRadiusDouble(appState.circle_info);
            if (largerCircleIndex == -1)
            {
                appState.algorithmFinished = true;
                appState.algorithmRunning = false;
                std::cerr << "The radii of the first two circles are not approximately float" << std::endl;
                return;
            }
            appState.large_circle_id = largerCircleIndex;

            undistortedImage.copyTo(appState.imageProcess);
            drawResult(appState.imageProcess, appState.circle_info, appState.large_circle_id);

            appState.large_circle_id = largerCircleIndex;
            appState.imageProcess2.resize(appState.measure_boxes.size());
            appState.imageProcess3.resize(appState.measure_boxes.size());
            appState.imageProcess4.resize(appState.measure_boxes.size());
            appState.pin_info.resize(appState.measure_boxes.size());
            appState.immvisionParamsStaticSub.resize(appState.measure_boxes.size());
            //        appState.scanResVec.resize(appState.measure_boxes.size());

            for (int i = 0; i < appState.measure_boxes.size(); i++)
            {
                appState.immvisionParamsStaticSub[i] = ImmVision::ImageParams();
                appState.immvisionParamsStaticSub[i].ImageDisplaySize = cv::Size(300, 300);
                appState.immvisionParamsStaticSub[i].ZoomKey = std::to_string(i + 1).c_str();
                //        immvisionParamsSub.ShowOptionsPanel = true;
                appState.immvisionParamsStaticSub[i].RefreshImage = true;
                appState.immvisionParamsStaticSub[i].forceFullView = true;
                appState.immvisionParamsStaticSub[i].ShowZoomButtons = false;
                appState.immvisionParamsStaticSub[i].ShowOptionsButton = false;
                appState.immvisionParamsStaticSub[i].ShowImageInfo = false;
                appState.immvisionParamsStaticSub[i].ShowPixelInfo = false;

                //        immvisionParamsSub.ShowOptionsInTooltip = false;
                // std::cout<<"appState.rect_w  appState.rect_h :"<<appState.rect_w<<" "<< appState.rect_h <<std::endl;
                if (appState.measure_boxes[i].rect_w > 1e-10 && appState.measure_boxes[i].rect_h > 1e-10)
                {
                    std::vector<float> largerCircle = appState.circle_info[appState.large_circle_id];
                    std::vector<float> circle = appState.circle_info[1 - appState.large_circle_id];
                    appState.circleDistance = calculateDistance(largerCircle, circle);
                    // std::cout << "Circle distance:" << appState.circleDistance << std::endl;

                    cv::Point v = cv::Point(circle[0] - largerCircle[0], circle[1] - largerCircle[1]);
                    float vnorm = cv::norm(v);
                    appState.relativeRectX = appState.measure_boxes[i].rect_x * vnorm;
                    appState.relativeRectY = appState.measure_boxes[i].rect_y * vnorm;
                    appState.relativeRectW = appState.measure_boxes[i].rect_w * vnorm;
                    appState.relativeRectH = appState.measure_boxes[i].rect_h * vnorm;
                    cv::Rect rect(appState.relativeRectX + largerCircle[0], appState.relativeRectY + largerCircle[1],
                                  appState.relativeRectW, appState.relativeRectH);
                    std::vector<cv::Point> rotatedRectPoints = rotatedRectangle(rect,
                                                                                largerCircle, v);
                    cv::polylines(appState.imageProcess, rotatedRectPoints, true, cv::Scalar(0, 255, 0), 10);

                    // 添加文字
                    cv::putText(appState.imageProcess, std::to_string(i + 1).c_str(), rotatedRectPoints[0],
                                cv::FONT_HERSHEY_SIMPLEX, 6.0, cv::Scalar(255, 0, 0), 5);

                    float rotationAngle = -std::atan2(v.y, v.x) * 180 / CV_PI;
                    if (appState.measure_boxes[i].flip)
                    {
                        rotationAngle += 180;
                    }
                    cv::Rect boundingRect = cv::boundingRect(rotatedRectPoints);
                    cv::Mat rotatedImage = cropAndRotateImage(undistortedImage, boundingRect, rotationAngle);
                    cv::Mat enhanced_image;
                    cv::LUT(rotatedImage, appState.lookup_table, enhanced_image);
                    cv::Mat binaryImage = binarizeImage(appState, enhanced_image, i);
                    if (!enhanced_image.empty())
                    {

                        enhanced_image.copyTo(appState.imageProcess4[i]);
                    }
                    if (binaryImage.channels() == 1)
                    {
                        if (!binaryImage.empty())
                        {
                            cv::cvtColor(binaryImage, appState.imageProcess2[i], cv::COLOR_GRAY2RGB);
                        }
                    }
                    else
                    {
                        appState.imageProcess2[i] = binaryImage;
                    }
                    findPins(appState, binaryImage, i);
                }

                appState.immvisionParamsStatic.RefreshImage = true;
                appState.needRefresh = true;
                appState.algorithmFinished = true;
                appState.algorithmRunning = false;
            }
        }
    }
    catch (const std::exception &e)
    {
        // 在此处处理异常
        std::cout << "algorithmFunction时发生异常: " << e.what() << std::endl;
    }
}

void measureFunction(AppState &appState)
{
    try
    {
        appState.measureRunning = true;
        int forCnt = 0;
        int forNum = 10;
        cv::Mat imageCap;
        std::vector<int> lastPinNum(appState.measure_boxes.size());
        for (size_t j = 0; j < appState.pin_measure_settings.size(); ++j)
        {
            appState.pin_measure_settings[j].distanceVec.clear();
        }
        for (size_t j = 0; j < appState.angle_measure_settings.size(); ++j)
        {
            appState.angle_measure_settings[j].angleVec.clear();
        }

        while (appState.measureRunning)
        {
            if (appState.largerCircleMoved == 0)
            {
                imageCap = cameraController.GetFrame();
                if (imageCap.empty() && !appState.imageCap.empty())
                {
                    imageCap = appState.imageCap;
                }
                if (!imageCap.empty())
                {

                    std::vector<std::vector<float>> circle_info;
                    // Detect circles
                    detectCircles(imageCap, circle_info);
                    if (circle_info.size() != 2)
                    {
                        continue;
                    }

                    int largerCircleIndex = getLargerCircleIfRadiusDouble(circle_info);
                    if (largerCircleIndex == -1)
                    {
                        std::cerr << "The radii of the first two circles are not approximately float" << std::endl;
                        continue;
                    }

                    appState.circle_info = circle_info;
                    appState.large_circle_id = largerCircleIndex;

                    imageCap.copyTo(appState.imageProcess);
                    drawResult(appState.imageProcess, appState.circle_info, appState.large_circle_id);

                    appState.imageProcess2.resize(appState.measure_boxes.size());
                    appState.imageProcess3.resize(appState.measure_boxes.size());
                    appState.pin_info.resize(appState.measure_boxes.size());
                    appState.immvisionParamsStaticSub.resize(appState.measure_boxes.size());
                    //        appState.scanResVec.resize(appState.measure_boxes.size());
                    bool ignoreflag = false;
                    for (int i = 0; i < appState.measure_boxes.size(); i++)
                    {
                        appState.immvisionParamsStaticSub[i] = ImmVision::ImageParams();
                        //                        appState.immvisionParamsStaticSub[i].ImageDisplaySize = cv::Size(300, 300);
                        appState.immvisionParamsStaticSub[i].ZoomKey = std::to_string(i + 1).c_str();
                        //        immvisionParamsSub.ShowOptionsPanel = true;
                        appState.immvisionParamsStaticSub[i].RefreshImage = true;
                        appState.immvisionParamsStaticSub[i].forceFullView = true;
                        appState.immvisionParamsStaticSub[i].ShowZoomButtons = false;
                        appState.immvisionParamsStaticSub[i].ShowOptionsButton = false;
                        appState.immvisionParamsStaticSub[i].ShowImageInfo = false;
                        appState.immvisionParamsStaticSub[i].ShowPixelInfo = false;

                        // std::cout<<"appState.rect_w  appState.rect_h :"<<appState.rect_w<<" "<< appState.rect_h <<std::endl;
                        if (appState.measure_boxes[i].rect_w > 1e-10 && appState.measure_boxes[i].rect_h > 1e-10)
                        {
                            std::vector<float> largerCircle = circle_info[largerCircleIndex];
                            std::vector<float> circle = circle_info[1 - largerCircleIndex];
                            //                    appState.circleDistance = calculateDistance(largerCircle, circle);
                            // std::cout << "Circle distance:" << appState.circleDistance << std::endl;

                            cv::Point v = cv::Point(circle[0] - largerCircle[0], circle[1] - largerCircle[1]);
                            float vnorm = cv::norm(v);
                            appState.relativeRectX = appState.measure_boxes[i].rect_x * vnorm;
                            appState.relativeRectY = appState.measure_boxes[i].rect_y * vnorm;
                            appState.relativeRectW = appState.measure_boxes[i].rect_w * vnorm;
                            appState.relativeRectH = appState.measure_boxes[i].rect_h * vnorm;
                            cv::Rect rect(appState.relativeRectX + largerCircle[0],
                                          appState.relativeRectY + largerCircle[1],
                                          appState.relativeRectW, appState.relativeRectH);
                            std::vector<cv::Point> rotatedRectPoints = rotatedRectangle(rect,
                                                                                        largerCircle, v);
                            cv::polylines(appState.imageProcess, rotatedRectPoints, true, cv::Scalar(0, 255, 0), 10);

                            // 添加文字
                            cv::putText(appState.imageProcess, std::to_string(i + 1).c_str(), rotatedRectPoints[0],
                                        cv::FONT_HERSHEY_SIMPLEX, 6.0, cv::Scalar(255, 0, 0), 5);
                            float rotationAngle = -std::atan2(v.y, v.x) * 180 / CV_PI;
                            if (appState.measure_boxes[i].flip)
                            {
                                rotationAngle += 180;
                            }
                            cv::Rect boundingRect = cv::boundingRect(rotatedRectPoints);
                            cv::Mat rotatedImage = cropAndRotateImage(imageCap, boundingRect, rotationAngle);
                            cv::Mat binaryImage = binarizeImage(appState, rotatedImage, i);

                            findPins(appState, binaryImage, i);

                            if (binaryImage.channels() == 1)
                            {
                                if (!binaryImage.empty())
                                {
                                    cv::cvtColor(binaryImage, appState.imageProcess2[i], cv::COLOR_GRAY2RGB);
                                }
                            }
                            else
                            {
                                appState.imageProcess2[i] = binaryImage;
                            }

                            if (lastPinNum[i] != 0 && lastPinNum[i] != appState.pin_info[i].size())
                            {
                                ignoreflag = true;
                                continue;
                            }
                            lastPinNum[i] = appState.pin_info[i].size();
                            auto connectedIndices = findMatchingRelationships2(appState.pin_info[i],
                                                                               appState.measure_boxes[i].flip);

                            for (size_t j = 0; j < appState.pin_measure_settings.size(); ++j)
                            {
                                auto measure_setting = appState.pin_measure_settings[j];

                                if (measure_setting.box != i)
                                {
                                    continue;
                                }
                                if (measure_setting.pin0 >= connectedIndices.size() ||
                                    measure_setting.pin1 >= connectedIndices.size())
                                {
                                    appState.pin_measure_settings[j].distance = -1;
                                    continue;
                                }

                                int ind = connectedIndices[measure_setting.pin0];
                                int ind_next = connectedIndices[measure_setting.pin1];
                                cv::Point2f p1(appState.pin_info[i][ind][0], appState.pin_info[i][ind][1]);
                                cv::Point2f p2(appState.pin_info[i][ind_next][0], appState.pin_info[i][ind_next][1]);
                                float distance = calculateDistance(p1, p2) * appState.mmPrePixel;
                                appState.pin_measure_settings[j].distanceVec.push_back(distance);
                                // std::cout << "distance:" << j << " " << distance << std::endl;
                            }

                            for (size_t j = 0; j < appState.angle_measure_settings.size(); ++j)
                            {
                                auto angle_measure_setting = appState.angle_measure_settings[j];

                                if (angle_measure_setting.box != i)
                                {
                                    continue;
                                }
                                if (angle_measure_setting.pin0 >= connectedIndices.size() ||
                                    angle_measure_setting.pin1 >= connectedIndices.size() ||
                                    angle_measure_setting.pin2 >= connectedIndices.size() ||
                                    angle_measure_setting.pin3 >= connectedIndices.size())
                                {
                                    appState.angle_measure_settings[j].angle = -1;
                                    continue;
                                }

                                int ind0 = connectedIndices[angle_measure_setting.pin0];
                                int ind1 = connectedIndices[angle_measure_setting.pin1];
                                int ind2 = connectedIndices[angle_measure_setting.pin2];
                                int ind3 = connectedIndices[angle_measure_setting.pin3];

                                cv::Point2f p0(appState.pin_info[i][ind0][0], appState.pin_info[i][ind0][1]);
                                cv::Point2f p1(appState.pin_info[i][ind1][0], appState.pin_info[i][ind1][1]);
                                cv::Point2f p2(appState.pin_info[i][ind2][0], appState.pin_info[i][ind2][1]);
                                cv::Point2f p3(appState.pin_info[i][ind3][0], appState.pin_info[i][ind3][1]);

                                float angle = calculateAngle(p0, p1, p2, p3);
                                appState.angle_measure_settings[j].angleVec.push_back(angle);
                                // std::cout << "angle:" << j << " " << angle << std::endl;
                            }
                        }
                    }
                    if (ignoreflag)
                    {
                        continue;
                    }
                    forCnt++;
                    std::cout << "forCnt:" << forCnt << std::endl;
                    if (forCnt >= forNum)
                    {
                        for (size_t j = 0; j < appState.pin_measure_settings.size(); ++j)
                        {
                            std::cout << appState.pin_measure_settings[j].distanceVec.size() << std::endl;
                            if (appState.pin_measure_settings[j].distanceVec.size() == forNum)
                            {
                                float sum = 0.0;
                                std::vector<float> &distanceVec = appState.pin_measure_settings[j].distanceVec;
                                std::sort(distanceVec.begin(), distanceVec.end()); // Sort the vector in ascending order
                                for (size_t k = 2; k < distanceVec.size() -
                                                       2;
                                     ++k)
                                { // Exclude the two minimum and two maximum values
                                    sum += distanceVec[k];
                                    //                                    std::cout << "mdistance:" << j << " " << distanceVec[k] << std::endl;
                                }
                                float average =
                                        sum / (distanceVec.size() - 4); // Calculate the average of the remaining values
                                appState.pin_measure_settings[j].distance = average;
                                std::cout << "pin_measure_settings:" << j << " "
                                          << appState.pin_measure_settings[j].distance << std::endl;
                            }
                        }
                        for (size_t j = 0; j < appState.angle_measure_settings.size(); ++j)
                        {
                            if (appState.angle_measure_settings[j].angleVec.size() == forNum)
                            {
                                float sum = 0.0;
                                std::vector<float> &angleVec = appState.angle_measure_settings[j].angleVec;
                                std::sort(angleVec.begin(), angleVec.end()); // Sort the vector in ascending order
                                for (size_t k = 2;
                                     k < angleVec.size() - 2; ++k)
                                { // Exclude the two minimum and two maximum values
                                    sum += angleVec[k];
                                    //                                    std::cout << "mangle:" << j << " " << angleVec[k] << std::endl;
                                }
                                float average =
                                        sum / (angleVec.size() - 4); // Calculate the average of the remaining values
                                appState.angle_measure_settings[j].angle = average;
                                std::cout << "pin_measure_settings:" << j << " "
                                          << appState.angle_measure_settings[j].angle << std::endl;
                            }
                        }

                        appState.immvisionParamsStatic.RefreshImage = true;
                        appState.needMeasure = 1;
                        appState.needRefresh = true;
                        appState.measureFinished = true;
                        appState.measureRunning = false;
                        return;
                    }

                    usleep(100);
                }
            }
            usleep(10000);
        }
    }
    catch (const std::exception &e)
    {
        // 在此处处理异常
        std::cout << "measureFunction时发生异常: " << e.what() << std::endl;
    }
}

bool AOIParams(AppState &appState)
{
    //    SobelParams params = appState.sobelParams;
    bool changed = false;
    static bool viewarea = false;

    if (ImGui::Button("打开关闭"))
    {
        cameraController.SetResolution(4056, 3040);
        cameraController.ToggleCamera(0);
        appState.image_points_seq.clear();
        appState.object_points.clear();
        appState.paramsSummary.clear();
    }
    // appState.needMeasure = -1;
    static int notdet = 0;
    if (cameraController.isRunning)
    {
        ImGui::SameLine();
        ImGui::Text("摄像头已打开");
        checker.Start(appState);
        if (appState.largerCircleMoved == -1)
        {
            ImGui::SameLine();
            ImGui::Text("未检测到");

            if (notdet++ > 10)
            {
                appState.imageProcess = cv::Mat();
                appState.needMeasure = 0;
                notdet = 0;
                appState.imageCap = cv::Mat();
            }
        }
        else if (appState.largerCircleMoved == 0)
        {
            ImGui::SameLine();
            ImGui::Text("静止");
            notdet = 0;
            if (appState.needMeasure == 0)
            {
                appState.imageCap = cameraController.GetFrame();
                if (!appState.imageCap.empty())
                {
                    // appState.imageCap.copyTo(appState.imageProcess);

                    // Draw the result
                    // drawResult(appState.imageProcess, appState.circle_info, largerCircleIndex);

                    appState.needRefresh = true;
                }
                appState.needMeasure = 1;
                viewarea = true;
            }
        }
        else
        { // 运动
            appState.imageCap = cv::Mat();
            appState.imageProcess = cv::Mat();
        }
    }

    static std::string saveFolderPath;

    if (!ImGui::CollapsingHeader("采集图像"))
    {
        if (ImGui::Button("抓取图像"))
        {
            appState.imageCap = cameraController.GetFrame();

            // 生成时间戳作为文件名
            std::time_t currentTime = std::time(nullptr);
            std::stringstream ss;
            ss << std::put_time(std::localtime(&currentTime), "%Y%m%d%H%M%S");
            std::string timestamp = ss.str();
            // if (saveFolderPath.size() <2) {
            // 创建文件夹选择对话框
            pfd::select_folder dialog("选择保存文件夹");
            // 如果用户选择了文件夹
            if (!dialog.result().empty())
            {
                // 设置保存文件夹路径
                saveFolderPath = dialog.result() + "/";
            }
            // }
            // 构建保存文件路径
            std::string saveFilePath = saveFolderPath + timestamp + ".png";
            if (!appState.imageCap.empty())
                // 保存图像到文件
                cv::imwrite(saveFilePath, appState.imageCap);
        }

        ImGui::SameLine();
        if (ImGui::Button("实时图像"))
        {
            appState.imageCap = cv::Mat();
        }

        if (ImGui::Button("打开文件"))
        {
            std::string title = "打开图片";
            std::string default_path;
            std::vector<std::string> filters = {"Image Files", "*.png *.jpg *.jpeg *.bmp"};
            pfd::open_file dialog(title, default_path, filters,
                                  pfd::opt::none);
            if (dialog.result().size() > 0)
            {
                std::cout << "open:" << dialog.result()[0] << std::endl;
                appState.imageCap = cv::imread(dialog.result()[0]);
                appState.imageProcess = cv::Mat();
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("原始图像"))
        {
            appState.imageProcess = cv::Mat();
        }
    }

    // Usage example:
    if (!ImGui::CollapsingHeader("图像处理"))
    {
        if (ImGui::Button("检测红圆"))
        {
            if (!appState.imageCap.empty())
            {
                appState.imageCap.copyTo(appState.imageProcess);

                // Detect circles
                detectCircles(appState.imageCap, appState.circle_info);

                int largerCircleIndex = getLargerCircleIfRadiusDouble(appState.circle_info);
                if (largerCircleIndex == -1)
                {
                    std::cerr << "The radii of the first two circles are not approximately float" << std::endl;
                    return -1;
                }
                appState.large_circle_id = largerCircleIndex;

                // Draw the result
                drawResult(appState.imageProcess, appState.circle_info, largerCircleIndex);
                viewarea = true;
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("添加检测框"))
        {
            appState.measure_boxes.push_back({});
            appState.last_measure_boxes.push_back({});
        }

        ImGui::SameLine();
        if (ImGui::Button("加载参数"))
        {
            //            appState.imageProcess = cv::Mat();
            //             cv::FileStorage fs(appState.savePath.data(), cv::FileStorage::READ);

            //             if (fs.isOpened()) {
            //                 fs["rect_x"] >> appState.measure_boxes[sel_box].rect_x;
            //                 fs["rect_y"] >> appState.measure_boxes[sel_box].rect_y;
            //                 fs["rect_w"] >> appState.measure_boxes[sel_box].rect_w;
            //                 fs["rect_h"] >> appState.measure_boxes[sel_box].rect_h;
            //                 fs["bin_threshold"] >> appState.measure_boxes[sel_box].bin_threshold;
            //                 fs["flip"] >> appState.measure_boxes[sel_box].flip;

            // //                fs["circle_info"] >> appState.circle_info;
            // //                fs["large_circle_id"] >> appState.large_circle_id;

            //                 fs.release();
            //                 viewarea = true;
            //             }
            appState.loadBoxSettings();
            appState.last_measure_boxes.resize(appState.measure_boxes.size());
            viewarea = true;
        }
        ImGui::Text("圆距离：%f", appState.circleDistance);

        if (!ImGui::CollapsingHeader("目标框"))
        {
            for (int i = 0; i < appState.measure_boxes.size(); i++)
            {
                if (ImGui::Selectable(("框" + std::to_string(i + 1)).c_str(), i == appState.sel_box))
                {
                    appState.sel_box = i;
                    std::cout << "sel_box:" << i << std::endl;
                }
            }

            int sel_box = appState.sel_box;
            if (sel_box >= 0 && sel_box < appState.measure_boxes.size() && appState.measure_boxes.size() > 0)
            {

                // Keep track of the previous values
                appState.last_measure_boxes[sel_box] = appState.measure_boxes[sel_box];
                static int prev_sel_box = appState.sel_box;

                ImGui::SliderFloat("X", &appState.measure_boxes[sel_box].rect_x, -2.0f, 2.0f);
                ImGui::SameLine();
                ImGui::InputFloat("##XInput", &appState.measure_boxes[sel_box].rect_x);

                ImGui::SliderFloat("Y", &appState.measure_boxes[sel_box].rect_y, -2.0f, 2.0f);
                ImGui::SameLine();
                ImGui::InputFloat("##YInput", &appState.measure_boxes[sel_box].rect_y);

                ImGui::SliderFloat("宽", &appState.measure_boxes[sel_box].rect_w, 0.0f, 1.0f);
                ImGui::SameLine();
                ImGui::InputFloat("##WidthInput", &appState.measure_boxes[sel_box].rect_w);

                ImGui::SliderFloat("高", &appState.measure_boxes[sel_box].rect_h, 0.0f, 1.0f);
                ImGui::SameLine();
                ImGui::InputFloat("##HeightInput", &appState.measure_boxes[sel_box].rect_h);

                ImGui::SliderFloat("二值化", &appState.measure_boxes[sel_box].bin_threshold, -0.0f, 255.0f);
                ImGui::SameLine();
                ImGui::InputFloat("##ThresholdInput", &appState.measure_boxes[sel_box].bin_threshold);

                ImGui::Checkbox("反向安装", &appState.measure_boxes[sel_box].flip);

                // Check if any slider value has changed
                if (appState.last_measure_boxes[sel_box].rect_x != appState.measure_boxes[sel_box].rect_x ||
                    appState.last_measure_boxes[sel_box].rect_y != appState.measure_boxes[sel_box].rect_y ||
                    appState.last_measure_boxes[sel_box].rect_w != appState.measure_boxes[sel_box].rect_w ||
                    appState.last_measure_boxes[sel_box].rect_h != appState.measure_boxes[sel_box].rect_h ||
                    appState.last_measure_boxes[sel_box].bin_threshold !=
                    appState.measure_boxes[sel_box].bin_threshold ||
                    appState.last_measure_boxes[sel_box].flip != appState.measure_boxes[sel_box].flip ||
                    prev_sel_box != appState.sel_box)
                {
                    // Update the previous values
                    appState.last_measure_boxes[sel_box] = appState.measure_boxes[sel_box];
                    prev_sel_box = appState.sel_box;
                    viewarea = true;
                }

                if (viewarea)
                {
                    appState.immvisionParamsStatic.RefreshImage = true;
                    appState.needRefresh = true;
                }

                // if (appState.circle_info.size() == 2) {
                if (viewarea)
                {
                    viewarea = false;
                    if (!appState.algorithmRunning)
                    {
                        std::thread algorithmThread(algorithmFunction, std::ref(appState));
                        algorithmThread.detach(); // 分离线程，使其在后台执行
                        // 标记算法正在执行中
                        appState.algorithmFinished = false;
                    }
                    //                    if ( !appState.measureRunning) {
                    //                        std::thread measureThread(measureFunction, std::ref(appState));
                    //                        measureThread.detach(); // 分离线程，使其在后台执行
                    //                        // 标记算法正在执行中
                    //                        appState.measureFinished = false;
                    //
                    //                    }
                }
                //                if (ImGui::Button("test")) {
                //                    appState.largerCircleMoved = 0;
                //                    if ( !appState.measureRunning) {
                //                        std::thread measureThread(measureFunction, std::ref(appState));
                //                        measureThread.detach(); // 分离线程，使其在后台执行
                //                        // 标记算法正在执行中
                //                        appState.measureFinished = false;
                //
                //                    }
                //                }
                // }

                if (ImGui::Button("保存参数"))
                {
                    //            appState.imageProcess = cv::Mat();
                    //             cv::FileStorage fs(appState.savePath.data(), cv::FileStorage::WRITE);
                    //             if (fs.isOpened()) {
                    //                 fs << "rect_x" << appState.rect_x;
                    //                 fs << "rect_y" << appState.rect_y;
                    //                 fs << "rect_w" << appState.rect_w;
                    //                 fs << "rect_h" << appState.rect_h;
                    //                 fs << "bin_threshold" << appState.bin_threshold;
                    //                 fs << "flip" << appState.flip;
                    // //                fs << "large_circle_id" << appState.large_circle_id;

                    //                 fs.release();
                    //             }
                    appState.saveBoxSettings();
                }
                ImGui::SameLine();
                if (ImGui::Button("删除该项"))
                {
                    if (appState.sel_box < appState.measure_boxes.size())
                    {
                        appState.last_measure_boxes.erase(appState.last_measure_boxes.begin() + appState.sel_box);
                        appState.measure_boxes.erase(appState.measure_boxes.begin() + appState.sel_box);
                        appState.sel_box = 0;
                    }
                }
            }
        }
    }
    if (!ImGui::CollapsingHeader("标定信息"))
    {
        {
            std::stringstream ss;
            ss << "Intrinsics: \n"
               << appState.intrinsics_in << std::endl;
            ss << "Distortion coefficients: \n"
               << appState.dist_coeffs << std::endl;
            ss << "Reprojection error: \n"
               << appState.reproj_err << std::endl;
            ss << "mm/Pixel: \n"
               << appState.mmPrePixel << std::endl;

            char buffer[1000];
            strncpy(buffer, ss.str().c_str(), sizeof(buffer));
            ImGui::InputTextMultiline("内参", buffer, sizeof(buffer));
        }
        ImGui::InputText("路径", appState.savePath.data(), 1000);
        if (ImGui::Button("加载标定"))
        {
            std::string title = "打开标定文件";
            std::string default_path;
            std::vector<std::string> filters = {"Calib Files", "*.yaml *.yml"};
            pfd::open_file dialog(title, default_path, filters,
                                  pfd::opt::none);
            if (dialog.result().size() > 0)
            {
                std::cout << "open calib:" << dialog.result()[0] << std::endl;
                appState.savePath = dialog.result()[0];
                appState.loadCalibrationParameters();
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("保存标定"))
        {
            // 定义相机内参矩阵
            appState.intrinsics_in = (cv::Mat_<double>(3, 3) << 1.76709614e+04, 0.00000000e+00, 2.02631760e+03,
                    0.00000000e+00, 1.76555437e+04, 1.51881402e+03,
                    0.00000000e+00, 0.00000000e+00, 1.00000000e+00);

            // 定义畸变系数
            appState.dist_coeffs = (cv::Mat_<double>(1, 5)
                    << -1.22490106e+00,
                    2.39645334e+01, 3.31088241e-03, -7.69564899e-03, 1.47150370e-01);

            appState.reproj_err = 0.05438925777095047;
            appState.mmPrePixel = 0.03734429785177405;
            // Save calibration parameters to YAML file
            appState.saveCalibrationParameters();
        }
    }
    if (!ImGui::CollapsingHeader("登陆信息"))
    {
        std::string jsonString = appState.scanResJson;
        if (jsonString.starts_with("{") && jsonString.ends_with("}"))
        {
            // 解析JSON字符串
            try
            {
                nlohmann::json data = nlohmann::json::parse(jsonString);
                // 在此处处理解析后的 JSON 数据

                // 读取参数并存储到结构体实例中
                //                appState.loginInfo.code = data.count("Code") > 0 ? data["Code"] : "N/A";
                if (data.count("MO") > 0)
                {
                    appState.loginInfo.mo = data["MO"];
                }

                if (data.count("Line") > 0)
                {
                    appState.loginInfo.line = data["Line"];
                }

                //                if (data.count("Station") > 0) {
                //                    appState.loginInfo.station = data["Station"];
                //                }
                //
                //                if (data.count("Workshop") > 0) {
                //                    appState.loginInfo.workshop = data["Workshop"];
                //                }
                //
                //                if (data.count("Model") > 0) {
                //                    appState.loginInfo.model = data["Model"];
                //                }

                //                if (data.count("Result") > 0) {
                //                    appState.loginInfo.result = data["Result"];
                //                }

                if (data.count("User") > 0)
                {
                    appState.loginInfo.user = data["User"];
                }

                if (data.count("url") > 0)
                {
                    appState.loginInfo.url = data["url"];
                }

                appState.saveLoginInfo();
            }
            catch (const std::exception &e)
            {
                // 在此处处理异常
                std::cout << "解析 JSON 时发生异常: " << e.what() << std::endl;
            }
        }

        ImGui::BeginTable("appState.loginInfoTable", 2, ImGuiTableFlags_Borders);
        ImGui::TableSetupColumn("字段", ImGuiTableColumnFlags_WidthFixed, 40.0f);
        ImGui::TableSetupColumn("值", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
        ImGui::TableNextColumn();
        ImGui::Text("字段");
        ImGui::TableNextColumn();
        ImGui::Text("值");
        ImGui::TableNextColumn();

        // 显示参数值
        //        ImGui::TableNextRow();
        //        ImGui::TableSetColumnIndex(0);
        //        ImGui::Text("Code:");
        //        ImGui::TableSetColumnIndex(1);
        //        ImGui::Text(appState.loginInfo.code.c_str());

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("MO:");
        ImGui::TableSetColumnIndex(1);
        ImGui::Text(appState.loginInfo.mo.c_str());

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Line:");
        ImGui::TableSetColumnIndex(1);
        ImGui::Text(appState.loginInfo.line.c_str());

        //        ImGui::TableNextRow();
        //        ImGui::TableSetColumnIndex(0);
        //        ImGui::Text("Station:");
        //        ImGui::TableSetColumnIndex(1);
        //        ImGui::Text(appState.loginInfo.station.c_str());
        //
        //        ImGui::TableNextRow();
        //        ImGui::TableSetColumnIndex(0);
        //        ImGui::Text("Workshop:");
        //        ImGui::TableSetColumnIndex(1);
        //        ImGui::Text(appState.loginInfo.workshop.c_str());
        //
        //        ImGui::TableNextRow();
        //        ImGui::TableSetColumnIndex(0);
        //        ImGui::Text("Model:");
        //        ImGui::TableSetColumnIndex(1);
        //        ImGui::Text(appState.loginInfo.model.c_str());

        //        ImGui::TableNextRow();
        //        ImGui::TableSetColumnIndex(0);
        //        ImGui::Text("Result:");
        //        ImGui::TableSetColumnIndex(1);
        //        ImGui::Text(appState.loginInfo.result.c_str());

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("User:");
        ImGui::TableSetColumnIndex(1);
        ImGui::Text(appState.loginInfo.user.c_str());

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("url:");
        ImGui::TableSetColumnIndex(1);
        ImGui::TextWrapped(appState.loginInfo.url.c_str());
        ImGui::EndTable();
    }

    return changed;
}

// Function to calculate and store the distances and angles for each point
void calculateMeasurements(int boxid, AppState &appState)
{
    // Clear the previous distance and angle measurement settings
    //    appState.pin_measure_settings.clear();
    //    appState.angle_measure_settings.clear();
    // Remove elements from appState.angle_measure_settings where boxid is the same as specified
    appState.angle_measure_settings.erase(
            std::remove_if(
                    appState.angle_measure_settings.begin(),
                    appState.angle_measure_settings.end(),
                    [boxid](const AppState::ANGLE_MEASURE_SETTING &setting)
                    {
                        return setting.box == boxid;
                    }),
            appState.angle_measure_settings.end());
    appState.pin_measure_settings.erase(
            std::remove_if(
                    appState.pin_measure_settings.begin(),
                    appState.pin_measure_settings.end(),
                    [boxid](const AppState::MEASURE_SETTING &setting)
                    {
                        return setting.box == boxid;
                    }),
            appState.pin_measure_settings.end());

    // Iterate over the points vector
    for (int i = 0; i < appState.cad_points.size() - 1; ++i)
    {
        // Calculate the distance between the current pair of appState.cad_points
        float distance = calculateDistance(appState.cad_points[i], appState.cad_points[i + 1]);

        // Create a new MEASURE_SETTING struct and assign the box and pin indices
        AppState::MEASURE_SETTING measure_setting;
        measure_setting.box = boxid;
        measure_setting.pin0 = i;
        measure_setting.pin1 = i + 1;
        measure_setting.minDistance = distance - appState.cad_tolerance;
        measure_setting.maxDistance = distance + appState.cad_tolerance;

        // Add the MEASURE_SETTING struct to the appState.pin_measure_settings vector
        appState.pin_measure_settings.push_back(measure_setting);
        int ii = i + 2;
        if (ii < appState.cad_points.size())
        {
            // Calculate the angle between the current triplet of appState.cad_points

            cv::Point2f tolerancex{appState.cad_tolerance, 0};
            cv::Point2f tolerancey{0, appState.cad_tolerance};
            float angle_minus0 = calculateAngle(appState.cad_points[i + 1], appState.cad_points[i] - tolerancex,
                                                appState.cad_points[i + 1], appState.cad_points[ii], true);
            float angle_minus1 = calculateAngle(appState.cad_points[i + 1] - tolerancex, appState.cad_points[i],
                                                appState.cad_points[i + 1] - tolerancex, appState.cad_points[ii], true);
            float angle_minus2 = calculateAngle(appState.cad_points[i + 1], appState.cad_points[i],
                                                appState.cad_points[i + 1], appState.cad_points[ii] - tolerancex, true);
            float angle_minus3 = calculateAngle(appState.cad_points[i + 1], appState.cad_points[i] - tolerancey,
                                                appState.cad_points[i + 1], appState.cad_points[ii], true);
            float angle_minus4 = calculateAngle(appState.cad_points[i + 1] - tolerancey, appState.cad_points[i],
                                                appState.cad_points[i + 1] - tolerancey, appState.cad_points[ii], true);
            float angle_minus5 = calculateAngle(appState.cad_points[i + 1], appState.cad_points[i],
                                                appState.cad_points[i + 1], appState.cad_points[ii] - tolerancey, true);
            float angle_plus0 = calculateAngle(appState.cad_points[i + 1], appState.cad_points[i] + tolerancex,
                                               appState.cad_points[i + 1], appState.cad_points[ii], true);
            float angle_plus1 = calculateAngle(appState.cad_points[i + 1] + tolerancex, appState.cad_points[i],
                                               appState.cad_points[i + 1] + tolerancex, appState.cad_points[ii], true);
            float angle_plus2 = calculateAngle(appState.cad_points[i + 1], appState.cad_points[i],
                                               appState.cad_points[i + 1], appState.cad_points[ii] + tolerancex, true);
            float angle_plus3 = calculateAngle(appState.cad_points[i + 1], appState.cad_points[i] + tolerancey,
                                               appState.cad_points[i + 1], appState.cad_points[ii], true);
            float angle_plus4 = calculateAngle(appState.cad_points[i + 1] + tolerancey, appState.cad_points[i],
                                               appState.cad_points[i + 1] + tolerancey, appState.cad_points[ii], true);
            float angle_plus5 = calculateAngle(appState.cad_points[i + 1], appState.cad_points[i],
                                               appState.cad_points[i + 1], appState.cad_points[ii] + tolerancey, true);
            // Print values
            std::cout << "angle_minus0: " << angle_minus0 << std::endl;
            std::cout << "angle_minus1: " << angle_minus1 << std::endl;
            std::cout << "angle_minus2: " << angle_minus2 << std::endl;
            std::cout << "angle_minus3: " << angle_minus3 << std::endl;
            std::cout << "angle_minus4: " << angle_minus4 << std::endl;
            std::cout << "angle_minus5: " << angle_minus5 << std::endl;
            std::cout << "angle_plus0: " << angle_plus0 << std::endl;
            std::cout << "angle_plus1: " << angle_plus1 << std::endl;
            std::cout << "angle_plus2: " << angle_plus2 << std::endl;
            std::cout << "angle_plus3: " << angle_plus3 << std::endl;
            std::cout << "angle_plus4: " << angle_plus4 << std::endl;
            std::cout << "angle_plus5: " << angle_plus5 << std::endl;
            // Assuming the angles are calculated as shown in your code

            float max_angle = std::max(
                    {angle_minus0, angle_minus1, angle_minus2, angle_minus3, angle_minus4, angle_minus5, angle_plus0,
                     angle_plus1, angle_plus2, angle_plus3, angle_plus4, angle_plus5});
            float min_angle = std::min(
                    {angle_minus0, angle_minus1, angle_minus2, angle_minus3, angle_minus4, angle_minus5, angle_plus0,
                     angle_plus1, angle_plus2, angle_plus3, angle_plus4, angle_plus5});

            // Create a new ANGLE_MEASURE_SETTING struct and assign the box and pin indices
            AppState::ANGLE_MEASURE_SETTING angle_measure_setting;
            angle_measure_setting.box = boxid;
            angle_measure_setting.pin0 = i + 1;
            angle_measure_setting.pin1 = i;
            angle_measure_setting.pin2 = i + 1;
            angle_measure_setting.pin3 = ii;
            angle_measure_setting.minAngle = min_angle;
            angle_measure_setting.maxAngle = max_angle;

            // Add the ANGLE_MEASURE_SETTING struct to the appState.angle_measure_settings vector
            appState.angle_measure_settings.push_back(angle_measure_setting);
        }
    }
}

// 用于绘制ImGui界面的函数
void DrawUI(AppState &appState)
{
    if (ImGui::BeginPopupModal("CAD", NULL, ImGuiWindowFlags_MenuBar))
    {

        // 检查并设置窗口的最小大小
        ImVec2 minWindowSize = ImVec2(800, 400);
        ImVec2 currentWindowSize = ImGui::GetWindowSize();
        if (currentWindowSize.x < minWindowSize.x || currentWindowSize.y < minWindowSize.y)
        {
            ImGui::SetWindowSize(minWindowSize); // 设置窗口大小
        }

        ImGui::Text("添加pin针编号，设置公差");

        ImVec2 windowSize = ImGui::GetContentRegionAvail();

        ImGui::BeginChild("Canvas", ImVec2(windowSize.x / 3, windowSize.y - 30));

        // 左侧画布
        ImGui::Text("绘图");
        ImGui::Separator();

        ImVec2 canvasPos = ImGui::GetCursorScreenPos();
        ImVec2 canvasSize = ImGui::GetContentRegionAvail();

        // 绘制画布边框
        ImGui::GetWindowDrawList()->AddRect(canvasPos, ImVec2(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y),
                                            IM_COL32(255, 255, 255, 255));

        // 绘制圆点
        if (!appState.cad_points.empty())
        {
            // 计算点的外接矩形
            float minX = appState.cad_points[0].x, minY = appState.cad_points[0].y;
            float maxX = appState.cad_points[0].x, maxY = appState.cad_points[0].y;
            for (const auto &point : appState.cad_points)
            {
                minX = std::min(minX, point.x);
                minY = std::min(minY, point.y);
                maxX = std::max(maxX, point.x);
                maxY = std::max(maxY, point.y);
            }

            // 计算外接矩形的宽度和高度
            float rectWidth = maxX - minX;
            float rectHeight = maxY - minY;

            // 计算画布的目标宽度和高度（70%的画布大小）
            float targetWidth = canvasSize.x * 0.7;
            float targetHeight = canvasSize.y * 0.7;

            // 计算缩放因子，以长边为准
            float scaleFactor = std::min(targetWidth / rectWidth, targetHeight / rectHeight);

            // 计算新的中心点
            ImVec2 newCenter = ImVec2(canvasPos.x + canvasSize.x / 2.0f, canvasPos.y + canvasSize.y / 2.0f);

            int index = 1;
            for (const auto &point : appState.cad_points)
            {
                // 缩放并转移到新的中心
                ImVec2 newPoint((point.x - minX) * scaleFactor + newCenter.x - rectWidth * scaleFactor / 2.0f,
                                (maxY - point.y ) * scaleFactor + newCenter.y - rectHeight * scaleFactor / 2.0f);

                // 绘制点
                ImGui::GetWindowDrawList()->AddCircleFilled(newPoint, 3.0f, IM_COL32(255, 0, 0, 255));

                // 绘制序号
                ImVec2 textPos = newPoint;
                textPos.x += 5; // 将文本稍微向右移动，避免覆盖点
                ImGui::GetWindowDrawList()->AddText(textPos, IM_COL32(0, 255, 0, 255), std::to_string(index).c_str());

                index++;
            }
        }

        ImGui::EndChild();

        ImGui::SameLine();

        ImGui::BeginChild("Table", ImVec2(windowSize.x / 3, windowSize.y - 30));

        // 中间表格
        ImGui::Text("坐标");
        ImGui::Separator();

        // 设置每个圆点的坐标
        float itemWidth = (windowSize.x / 3 - 140) / 2;
        int delid = -1;
        for (int i = 0; i < appState.cad_points.size(); ++i)
        {
            ImGui::Text("点%d:", i + 1);
            ImGui::SameLine();

            ImGui::SetNextItemWidth(itemWidth);
            ImGui::InputFloat(("X" + std::to_string(i + 1)).c_str(), &appState.cad_points[i].x);
            ImGui::SameLine();

            ImGui::SetNextItemWidth(itemWidth);
            ImGui::InputFloat(("Y" + std::to_string(i + 1)).c_str(), &appState.cad_points[i].y);
            ImGui::SameLine();

            if (ImGui::Button("删除"))
            {
                delid = i;
            }
        }
        if (delid >= 0)
        {
            appState.cad_points.erase(appState.cad_points.begin() + delid);
        }

        // 添加按钮
        if (ImGui::Button("添加点"))
        {
            appState.cad_points.push_back({0.0f, 0.0f});
        }

        ImGui::EndChild();

        ImGui::SameLine();

        ImGui::BeginChild("Settings", ImVec2(windowSize.x / 3, windowSize.y - 30));

        // 右侧设置界面
        ImGui::Text("设置");
        ImGui::Separator();

        // 设置公差范围
        //        ImGui::Text("公差:");
        //        ImGui::SliderFloat("##公差", &appState.cad_tolerance, 0.0f, 50.0f);
        ImGui::InputFloat("公差", &appState.cad_tolerance);
        ImGui::Separator();
        static int boxid = 1;
        ImGui::InputInt("框", &boxid);
        if (ImGui::Button("更新到框"))
        {
            if (boxid - 1 >= 0 && boxid - 1 < appState.measure_boxes.size())
                calculateMeasurements(boxid - 1, appState);
        }

        ImGui::EndChild();

        if (ImGui::Button("关闭"))
            ImGui::CloseCurrentPopup();

        ImGui::End();
    }
}

void DistMeasureTable(AppState &appState)
{
    bool update_edit = false;
    //    size_t box_num = appState.pin_info.size();
    if (ImGui::Button("添加测量"))
        ImGui::OpenPopup("添加测量");
    if (ImGui::BeginPopupModal("添加测量", NULL, ImGuiWindowFlags_MenuBar))
    {

        ImGui::Text("选择两个pin针编号，设置容许范围");
        // Testing behavior of widgets stacking their own regular popups over the modal.
        static int box = 1;
        static int pin0 = 1;
        static int pin1 = 2;
        static float minDistance = 1.5;
        static float maxDistance = 1.9;

        ImGui::InputInt("框编号", &box);
        ImGui::InputInt("针编号1", &pin0);
        ImGui::InputInt("针编号2", &pin1);
        ImGui::InputFloat("最小距离", &minDistance);
        ImGui::InputFloat("最大距离", &maxDistance);

        //        if(pin0>=pin_num){
        //            pin0 = pin_num-1;
        //        }
        //        if(pin1>=pin_num){
        //            pin1 = pin_num-1;
        //        }
        if (pin0 < 1)
        {
            pin0 = 1;
        }
        if (pin1 < 1)
        {
            pin1 = 1;
        }

        ImGui::Separator();
        if (ImGui::Button("关闭"))
            ImGui::CloseCurrentPopup();
        ImGui::SameLine();
        if (ImGui::Button("添加"))
        {
            appState.pin_measure_settings.emplace_back(
                    AppState::MEASURE_SETTING{box - 1, pin0 - 1, pin1 - 1, minDistance, maxDistance});
            ImGui::CloseCurrentPopup();
            appState.needRefresh = true;
        }
        ImGui::EndPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("加载设置"))
    {
        appState.loadPairSettings();
        appState.needRefresh = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("保存设置"))
    {
        appState.savePairSettings();
    }
    ImGui::SameLine();
    if (ImGui::Button("CAD"))
    {
        ImGui::OpenPopup("CAD");
    }
    DrawUI(appState);
    ImGui::BeginTable("WatchedPixels", 6, ImGuiTableFlags_Borders);
    ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
    //    ImGui::TableNextColumn();
    //    ImGui::Text("IND");
    ImGui::TableNextColumn();
    ImGui::Text("框");
    ImGui::TableNextColumn();

    ImGui::Text("两针");
    ImGui::TableNextColumn();
    ImGui::Text("最小");
    ImGui::TableNextColumn();
    ImGui::Text("最大");
    ImGui::TableNextColumn();
    ImGui::Text("距离");
    ImGui::TableNextColumn();
    ImGui::Text("编辑");

    static size_t indices_to_remove = -1;
    static size_t indices_to_edit = -1;
    appState.pass_status.resize(appState.measure_boxes.size());
    for (int i = 0; i < appState.pass_status.size(); ++i)
    {
        if (appState.imageProcess3.size() == appState.pass_status.size() && !appState.imageProcess3[i].empty())
        {
            appState.pass_status[i] = 1; // pass
        }
        else
        {
            appState.pass_status[i] = 0; // check
        }
    }

    for (size_t i = 0; i < appState.pin_measure_settings.size(); ++i)
    {
        auto pin_measure_setting = appState.pin_measure_settings[i];
        ImGui::TableNextRow();
        if (appState.imageProcess3.size() > 0 && !appState.imageProcess3[0].empty() && !appState.imageProcess.empty())
        {
            if (pin_measure_setting.distance < 0)
            {
                ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0,
                                       ImGui::GetColorU32(ImVec4(0.5f, 0.5f, 0.0f, 1.0f)));
                appState.pass_status[pin_measure_setting.box] = 3; // loss
            }
            else if (pin_measure_setting.distance < pin_measure_setting.minDistance ||
                     pin_measure_setting.distance > pin_measure_setting.maxDistance)
            {
                ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0,
                                       ImGui::GetColorU32(ImVec4(1.0f, 0.0f, 0.0f, 1.0f))); // 设置一行的背景颜色为红色
                appState.pass_status[pin_measure_setting.box] = 2;                          // ng
            }
        }
        else
        {
            appState.pass_status[pin_measure_setting.box] = 0; // wait
            pin_measure_setting.distance = -1;
        }

        // index
        //        ImGui::TableNextColumn();
        //        ImGui::Text("#%i: ", (int) i);

        ImGui::TableNextColumn();
        ImGui::Text("%i", (int)pin_measure_setting.box + 1);

        // (x,y)
        ImGui::TableNextColumn();
        std::string posStr =
                std::to_string(pin_measure_setting.pin0 + 1) + "-" + std::to_string(pin_measure_setting.pin1 + 1);
        ImGui::Text("%s", posStr.c_str());

        ImGui::TableNextColumn();
        ImGui::Text("%0.3f", pin_measure_setting.minDistance);
        ImGui::TableNextColumn();
        ImGui::Text("%0.3f", pin_measure_setting.maxDistance);

        // Show Color Cell
        ImGui::TableNextColumn();
        ImGui::Text("%0.3f", pin_measure_setting.distance);

        // Actions
        ImGui::TableNextColumn();
        std::string lblRemove = "x##" + std::to_string(i + 1);
        if (ImGui::SmallButton(lblRemove.c_str()))
        {
            indices_to_remove = i;
        }
        ImGui::SameLine();
        std::string lblEdit = "e##" + std::to_string(i + 1);
        if (ImGui::SmallButton(lblEdit.c_str()))
        {
            indices_to_edit = i;
            update_edit = true;
        }
        ImGui::SameLine();
    }

    ImGui::EndTable();
    // Remove elements in reverse order
    if (indices_to_remove != -1)
    {
        appState.pin_measure_settings.erase(appState.pin_measure_settings.begin() + indices_to_remove);
        indices_to_remove = -1;
        appState.needRefresh = true;
    }
    if (indices_to_edit != -1)
    {
        ImGui::OpenPopup("编辑测量");
        if (ImGui::BeginPopupModal("编辑测量"))
        {
            ImGui::Text("选择两个pin针编号，设置容许范围");
            auto setting = (appState.pin_measure_settings.begin() + indices_to_edit);
            // Testing behavior of widgets stacking their own regular popups over the modal.
            static int box = setting->box + 1;
            static int pin0 = setting->pin0 + 1;
            static int pin1 = setting->pin1 + 1;
            static float minDistance = setting->minDistance;
            static float maxDistance = setting->maxDistance;

            ImGui::InputInt("框编号", &box);
            ImGui::InputInt("针编号1", &pin0);
            ImGui::InputInt("针编号2", &pin1);
            ImGui::InputFloat("最小距离", &minDistance);
            ImGui::InputFloat("最大距离", &maxDistance);

            //            if(pin0>=pin_num){
            //                pin0 = pin_num-1;
            //            }
            //            if(pin1>=pin_num){
            //                pin1 = pin_num-1;
            //            }
            if (pin0 < 1)
            {
                pin0 = 1;
            }
            if (pin1 < 1)
            {
                pin1 = 1;
            }

            ImGui::Separator();
            if (update_edit)
            {
                box = setting->box + 1;
                pin0 = setting->pin0 + 1;
                pin1 = setting->pin1 + 1;
                minDistance = setting->minDistance;
                maxDistance = setting->maxDistance;
                update_edit = false;
            }
            ImGui::SameLine();
            if (ImGui::Button("关闭"))
            {
                ImGui::CloseCurrentPopup();
                indices_to_edit = -1;
            }
            ImGui::SameLine();
            if (ImGui::Button("更新"))
            {
                setting->box = box - 1;
                setting->pin0 = pin0 - 1;
                setting->pin1 = pin1 - 1;
                setting->minDistance = minDistance;
                setting->maxDistance = maxDistance;
                setting->distance = -1;
                ImGui::CloseCurrentPopup();
                indices_to_edit = -1;
                appState.needRefresh = true;
            }
            ImGui::EndPopup();
        }
    }
}

void AngleMeasureTable(AppState &appState)
{
    bool update_edit = false;
    if (ImGui::Button("添加测量"))
        ImGui::OpenPopup("添加测量");
    if (ImGui::BeginPopupModal("添加测量", NULL, ImGuiWindowFlags_MenuBar))
    {

        ImGui::Text("选择三个pin针编号，设置容许范围");

        static int box = 1;
        static int pin0 = 1;
        static int pin1 = 2;
        static int pin2 = 2;
        static int pin3 = 3;
        static float minAngle = 170;
        static float maxAngle = 190;

        ImGui::InputInt("框编号", &box);
        ImGui::InputInt("针编号1", &pin0);
        ImGui::InputInt("针编号2", &pin1);
        ImGui::InputInt("针编号3", &pin2);
        ImGui::InputInt("针编号4", &pin3);
        ImGui::InputFloat("最小角度", &minAngle);
        ImGui::InputFloat("最大角度", &maxAngle);
        if (ImGui::Button("180"))
        {
            minAngle = 170;
            maxAngle = 190;
        }
        ImGui::SameLine();
        if (ImGui::Button("90"))
        {
            minAngle = 80;
            maxAngle = 100;
        }

        if (pin0 < 1)
        {
            pin0 = 1;
        }
        if (pin1 < 1)
        {
            pin1 = 1;
        }
        if (pin2 < 1)
        {
            pin2 = 1;
        }
        if (pin3 < 1)
        {
            pin3 = 1;
        }

        ImGui::Separator();
        if (ImGui::Button("关闭"))
            ImGui::CloseCurrentPopup();
        ImGui::SameLine();
        if (ImGui::Button("添加"))
        {
            appState.angle_measure_settings.emplace_back(
                    AppState::ANGLE_MEASURE_SETTING{box - 1, pin0 - 1, pin1 - 1, pin2 - 1, pin3 - 1, minAngle,
                                                    maxAngle});
            ImGui::CloseCurrentPopup();
            appState.needRefresh = true;
        }
        ImGui::EndPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("加载设置"))
    {
        appState.loadAngleSettings();
        appState.needRefresh = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("保存设置"))
    {
        appState.saveAngleSettings();
    }

    ImGui::BeginTable("WatchedPixels", 6, ImGuiTableFlags_Borders);
    ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
    ImGui::TableNextColumn();
    ImGui::Text("框");
    ImGui::TableNextColumn();
    ImGui::Text("两线");
    ImGui::TableNextColumn();
    ImGui::Text("最小");
    ImGui::TableNextColumn();
    ImGui::Text("最大");
    ImGui::TableNextColumn();
    ImGui::Text("角度");
    ImGui::TableNextColumn();
    ImGui::Text("编辑");

    static size_t indices_to_remove = -1;
    static size_t indices_to_edit = -1;
    appState.angle_pass_status.resize(appState.measure_boxes.size());
    for (int i = 0; i < appState.angle_pass_status.size(); ++i)
    {
        if (appState.imageProcess3.size() == appState.angle_pass_status.size() && !appState.imageProcess3[i].empty())
        {
            appState.angle_pass_status[i] = 1; // pass
        }
        else
        {
            appState.angle_pass_status[i] = 0; // check
        }
    }

    for (size_t i = 0; i < appState.angle_measure_settings.size(); ++i)
    {
        auto angle_measure_setting = appState.angle_measure_settings[i];
        ImGui::TableNextRow();
        if (appState.imageProcess3.size() > 0 && !appState.imageProcess3[0].empty() && !appState.imageProcess.empty())
        {
            if (angle_measure_setting.angle < 0)
            {
                ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0,
                                       ImGui::GetColorU32(ImVec4(0.5f, 0.5f, 0.0f, 1.0f)));
                appState.angle_pass_status[angle_measure_setting.box] = 3; // loss
            }
            else if (angle_measure_setting.angle < angle_measure_setting.minAngle ||
                     angle_measure_setting.angle > angle_measure_setting.maxAngle)
            {
                ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0,
                                       ImGui::GetColorU32(ImVec4(1.0f, 0.0f, 0.0f, 1.0f))); // 设置一行的背景颜色为红色
                appState.angle_pass_status[angle_measure_setting.box] = 2;                  // ng
            }
        }
        else
        {
            appState.angle_pass_status[angle_measure_setting.box] = 0; // wait
            angle_measure_setting.angle = -1;
        }

        ImGui::TableNextColumn();
        ImGui::Text("%i", (int)angle_measure_setting.box + 1);

        ImGui::TableNextColumn();
        std::string posStr =
                std::to_string(angle_measure_setting.pin0 + 1) + std::to_string(angle_measure_setting.pin1 + 1) +
                "-" + std::to_string(angle_measure_setting.pin2 + 1) + std::to_string(angle_measure_setting.pin3 + 1);
        ImGui::Text("%s", posStr.c_str());

        ImGui::TableNextColumn();
        ImGui::Text("%0.1f", angle_measure_setting.minAngle);
        ImGui::TableNextColumn();
        ImGui::Text("%0.1f", angle_measure_setting.maxAngle);

        ImGui::TableNextColumn();
        ImGui::Text("%0.1f", angle_measure_setting.angle);

        ImGui::TableNextColumn();
        std::string lblRemove = "x##" + std::to_string(i + 1);
        if (ImGui::SmallButton(lblRemove.c_str()))
        {
            indices_to_remove = i;
        }
        ImGui::SameLine();
        std::string lblEdit = "e##" + std::to_string(i + 1);
        if (ImGui::SmallButton(lblEdit.c_str()))
        {
            indices_to_edit = i;
            update_edit = true;
        }
        ImGui::SameLine();
    }

    ImGui::EndTable();
    // Remove elements in reverse order
    if (indices_to_remove != -1)
    {
        appState.angle_measure_settings.erase(appState.angle_measure_settings.begin() + indices_to_remove);
        indices_to_remove = -1;
        appState.needRefresh = true;
    }
    if (indices_to_edit != -1)
    {
        ImGui::OpenPopup("编辑测量");
        if (ImGui::BeginPopupModal("编辑测量"))
        {
            ImGui::Text("选择三个pin针编号，设置容许范围");
            auto setting = (appState.angle_measure_settings.begin() + indices_to_edit);
            static int box = setting->box + 1;
            static int pin0 = setting->pin0 + 1;
            static int pin1 = setting->pin1 + 1;
            static int pin2 = setting->pin2 + 1;
            static int pin3 = setting->pin3 + 1;
            static float minAngle = setting->minAngle;
            static float maxAngle = setting->maxAngle;

            ImGui::InputInt("框编号", &box);
            ImGui::InputInt("针编号1", &pin0);
            ImGui::InputInt("针编号2", &pin1);
            ImGui::InputInt("针编号3", &pin2);
            ImGui::InputInt("针编号4", &pin3);
            ImGui::InputFloat("最小角度", &minAngle);
            ImGui::InputFloat("最大角度", &maxAngle);

            if (pin0 < 1)
            {
                pin0 = 1;
            }
            if (pin1 < 1)
            {
                pin1 = 1;
            }
            if (pin2 < 1)
            {
                pin2 = 1;
            }
            if (pin3 < 1)
            {
                pin3 = 1;
            }
            if (update_edit)
            {
                box = setting->box + 1;
                pin0 = setting->pin0 + 1;
                pin1 = setting->pin1 + 1;
                pin2 = setting->pin2 + 1;
                pin3 = setting->pin3 + 1;
                minAngle = setting->minAngle;
                maxAngle = setting->maxAngle;
                update_edit = false;
            }
            ImGui::Separator();
            if (ImGui::Button("关闭"))
            {
                ImGui::CloseCurrentPopup();
                indices_to_edit = -1;
            }
            ImGui::SameLine();
            if (ImGui::Button("更新"))
            {
                setting->box = box - 1;
                setting->pin0 = pin0 - 1;
                setting->pin1 = pin1 - 1;
                setting->pin2 = pin2 - 1;
                setting->pin3 = pin3 - 1;
                setting->minAngle = minAngle;
                setting->maxAngle = maxAngle;
                setting->angle = -1;
                ImGui::CloseCurrentPopup();
                indices_to_edit = -1;
                appState.needRefresh = true;
            }
            ImGui::EndPopup();
        }
    }
}

ImVec2 set_ImageSize(float listWidth, bool showOptionsColumn)
{
    ImVec2 imageSize;

    float emSize = ImGui::GetFontSize();
    float x_margin = emSize * 2.f;
    float y_margin = emSize / 3.f;
    float image_info_height = ImGui::GetFontSize() * 6.f;
    //    image_info_height -= emSize * 1.5f;

    float image_options_width = showOptionsColumn ? ImGui::GetFontSize() * 19.f : 0.f;
    ImVec2 winSize = ImGui::GetWindowSize();
    imageSize = ImVec2(
            winSize.x - listWidth - x_margin - image_options_width,
            winSize.y - y_margin - image_info_height);
    if (imageSize.x < 1.f)
        imageSize.x = 1.f;
    if (imageSize.y < 1.f)
        imageSize.y = 1.f;

    return imageSize;
};

void DrawRectangles(std::vector<int> pass_status, std::vector<int> angle_pass_status)
{
    ImGuiIO &io = ImGui::GetIO();
    ImGuiStyle &style = ImGui::GetStyle();
    ImVec2 windowPos = ImGui::GetCursorScreenPos();
    ImVec2 winSize = ImGui::GetWindowSize();

    int numCols = ceil(sqrt(pass_status.size()));            // 列数
    int numRows = ceil(float(pass_status.size()) / numCols); // 行数
    float squareSize = winSize.x / numCols;                  // 矩形的大小

    for (int row = 0; row < numRows; row++)
    {
        for (int col = 0; col < numCols; col++)
        {
            int pos = col + row * numCols;
            if (pos >= pass_status.size())
                continue;
            // 计算矩形的位置
            ImVec2 rectPos = ImVec2(windowPos.x + col * squareSize, windowPos.y + row * squareSize / 2);

            // 根据行列计算颜色
            std::vector<ImU32> colors = {IM_COL32(127, 127, 127, 255), IM_COL32(0, 255, 0, 255),
                                         IM_COL32(255, 0, 0, 255), IM_COL32(255, 255, 0, 255)};
            std::vector<std::string> words = {"CHECK", "PASS", "NG", "MISS"};
            int status = std::max(pass_status[pos], angle_pass_status[pos]);

            std::string word = std::to_string(pos + 1) + ":" + words[status];

            // 定义放大镜的缩放倍数
            const float zoomFactor = squareSize * 0.012;
            // 计算文字的位置
            ImVec2 textSize = {ImGui::CalcTextSize(word.c_str()).x * zoomFactor,
                               ImGui::CalcTextSize(word.c_str()).y * zoomFactor};
            ImVec2 textPos = ImVec2(rectPos.x + (squareSize - textSize.x) * 0.5f,
                                    rectPos.y + (squareSize / 2 - textSize.y) * 0.5f);

            // 写入文字

            // 设置放大镜窗口的缩放
            ImGui::SetWindowFontScale(zoomFactor);
            // 绘制矩形
            ImGui::GetWindowDrawList()->AddRectFilled(rectPos,
                                                      ImVec2(rectPos.x + squareSize - 1,
                                                             rectPos.y + squareSize / 2 - 1),
                                                      colors[status]);
            // 写入放大后的文字
            ImGui::GetWindowDrawList()->AddText(textPos, 0xff000000, word.c_str());

            ImGui::SetWindowFontScale(1);
        }
    }

    // 更新光标位置，以便在同一窗口中绘制其他内容
    ImGui::SetCursorScreenPos(ImVec2(windowPos.x, windowPos.y + numRows * squareSize / 2 + style.ItemSpacing.y));
}

void ImageView(AppState &appState)
{
    //    static AppState appState;
    ImmVision::ImageWidgets::s_CollapsingHeader_CacheState_Sync = true;

    bool showOptionsColumn = true;
    if ((appState.immvisionParams.ShowOptionsInTooltip) || (!appState.immvisionParams.ShowOptionsPanel))
        showOptionsColumn = false;
    ImVec2 imageSize = set_ImageSize(0, showOptionsColumn);
    if (imageSize.x < 0)
    {
        imageSize.x = 100;
    }
    //    std::cout<<showOptionsColumn<<" "<<imageSize.x<<" "<<imageSize.y<<std::endl;

    try
    {
        {
            if (cameraController.isRunning)
            {
                cv::Mat tim = cameraController.GetThumbnail();
                if (!tim.empty())
                {
                    appState.image = tim;
                }
            }
            // if(!appState.needRefresh)
            {
                appState.immvisionParamsStatic.ImageDisplaySize = cv::Size((int)imageSize.x, 0);
                appState.immvisionParams.ImageDisplaySize = cv::Size((int)imageSize.x, 0);
                if (!appState.imageProcess.empty() && appState.algorithmFinished)
                {
                    ImmVision::Image("Process", appState.imageProcess, &appState.immvisionParamsStatic);
                }
                else if (!appState.imageCap.empty())
                {
                    ImmVision::Image("File", appState.imageCap, &appState.immvisionParamsStatic);
                }
                else if (checker.isRunning && !appState.imageDraw.empty())
                {
                    ImmVision::Image("Checker", appState.imageDraw, &appState.immvisionParams);
                    // usleep(1e3);
                    //                appState.imageDraw = cv::Mat();
                }
                else if (!appState.image.empty())
                {
                    ImmVision::Image("Stream", appState.image, &appState.immvisionParams);
                }
            }
            if (appState.pass_status.size() > 0 && appState.pass_status.size() == appState.angle_pass_status.size())
                DrawRectangles(appState.pass_status, appState.angle_pass_status);
        }
        if (!ImGui::CollapsingHeader("扫码枪"))
        {
            if (!scannerIsRunning)
            {
                ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "扫码枪连接失败");
            }
            ImGui::BeginTable("WatchedPixels", 3, ImGuiTableFlags_Borders);

            // 设置列的宽度
            ImGui::TableSetupColumn("序列", ImGuiTableColumnFlags_WidthFixed, 30.0f);
            ImGui::TableSetupColumn("码", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("编辑", ImGuiTableColumnFlags_WidthFixed, 30.0f);

            ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
            ImGui::TableNextColumn();
            ImGui::Text("序列");
            ImGui::TableNextColumn();
            ImGui::Text("码");
            ImGui::TableNextColumn();
            ImGui::Text("编辑");

            static size_t indices_to_remove = -1;
            for (size_t i = 0; i < appState.scanResVec.size(); ++i)
            {
                auto scan_res = appState.scanResVec[i];
                ImGui::TableNextRow();

                // index
                ImGui::TableNextColumn();
                ImGui::Text("%i", (int)i + 1);

                // (x,y)
                ImGui::TableNextColumn();
                ImGui::TextWrapped("%s", scan_res.c_str());

                // Actions
                ImGui::TableNextColumn();
                std::string lblRemove = "x##" + std::to_string(i + 1);
                if (ImGui::SmallButton(lblRemove.c_str()))
                {
                    indices_to_remove = i;
                }
                ImGui::SameLine();
            }
            if (indices_to_remove != -1)
            {
                appState.scanResVec.erase(appState.scanResVec.begin() + indices_to_remove);
                indices_to_remove = -1;
            }

            ImGui::EndTable();
        }

        if (!ImGui::CollapsingHeader("上传结果"))
        {
            static std::vector<UploadRecord> uploadRecords;

            //        if(appState.needMeasure == 2){
            //         for (size_t i = 0; i < appState.scanResVec.size(); ++i) {
            //             int pass_status = std::max(appState.pass_status[i], appState.angle_pass_status[i]);
            // //                uploadRecords.emplace_back();
            //             ImGui::InputInt("pass_status",&pass_status);
            //         }
            // ImGui::InputInt("needMeasure",&appState.needMeasure);
            if (ImGui::Button("上传") || appState.needMeasure == 2)
            {
                uploadRecords.resize(appState.scanResVec.size());
                bool upload_flag = false;
                for (size_t i = 0; i < appState.scanResVec.size(); ++i)
                {
                    int pass_status = std::max(appState.pass_status[i], appState.angle_pass_status[i]);
                    //                uploadRecords.emplace_back();
                    if (pass_status > 0)
                    {
                        std::thread uploadThread(upload, std::ref(appState), i, std::ref(uploadRecords[i]));
                        uploadThread.detach();
                        upload_flag = true;
                    }
                }
                if (upload_flag)
                    appState.needMeasure = 3;
            }
            if (appState.needMeasure == 1)
            {
                uploadRecords.clear();
            }

            if (ImGui::BeginTable("upload_table", 3))
            {
                ImGui::TableSetupColumn("序列");
                ImGui::TableSetupColumn("状态");
                ImGui::TableSetupColumn("响应");
                ImGui::TableHeadersRow();

                for (const auto &record : uploadRecords)
                {
                    ImGui::TableNextRow();
                    // index
                    ImGui::TableSetColumnIndex(0);
                    ImGui::TextWrapped("%s", record.code.c_str());
                    ImGui::TableSetColumnIndex(1);
                    std::vector<std::string> status_words = {"失败", "成功", "上传中"};
                    std::vector<ImVec4> colors = {ImVec4(1.0f, 0.0f, 0.0f, 1.0f), ImVec4(0.0f, 1.0f, 0.0f, 1.0f),
                                                  ImVec4(1.0f, 1.0f, 1.0f, 1.0f)};
                    if (record.success >= 0)
                        ImGui::TextColored(colors[record.success],
                                           status_words[record.success].c_str());
                    ImGui::TableSetColumnIndex(2);
                    ImGui::TextWrapped(record.response.c_str());
                }

                ImGui::EndTable();
            }
        }
    }
    catch (const std::exception &e)
    {
        // 在此处处理异常
        std::cout << "ImageView时发生异常: " << e.what() << std::endl;
    }
    ImmVision::ImageWidgets::s_CollapsingHeader_CacheState_Sync = false;
}

namespace nlohmann
{
    void to_json(json &j, const AppState::MEASURE_SETTING &m)
    {
        j = json{
                {"box", m.box},
                {"pin0", m.pin0},
                {"pin1", m.pin1},
                {"minDistance", m.minDistance},
                {"maxDistance", m.maxDistance},
                {"distance", m.distance}};
    }

    void to_json(json &j, const AppState::ANGLE_MEASURE_SETTING &a)
    {
        j = json{
                {"box", a.box},
                {"pin0", a.pin0},
                {"pin1", a.pin1},
                {"pin2", a.pin2},
                {"pin3", a.pin3},
                {"minAngle", a.minAngle},
                {"maxAngle", a.maxAngle},
                {"angle", a.angle}};
    }
}

// Convert a struct to JSON string
std::string
MeasureToJson(std::vector<std::string> scan_acodes, std::vector<AppState::MEASURE_SETTING> pin_measure_settings,
              std::vector<AppState::ANGLE_MEASURE_SETTING> angle_measure_settings)
{
    nlohmann::json root;
    root["acodes"] = scan_acodes;
    root["pin_measure_settings"] = pin_measure_settings;
    root["angle_measure_settings"] = angle_measure_settings;
    return root.dump();
}

void ResView(AppState &appState)
{
    //    static AppState appState;
    ImmVision::ImageWidgets::s_CollapsingHeader_CacheState_Sync = true;

    //    ImVec2 imageSize = set_ImageSize(0, false);
    ImVec2 winSize = ImGui::GetWindowSize();

    int numCols = ceil(sqrt(appState.measure_boxes.size()));            // 列数
    int numRows = ceil(float(appState.measure_boxes.size()) / numCols); // 行数
    float squareSizex = winSize.x / numCols - 30;
    float squareSizey = winSize.y / (numRows * 2) - 40;
    bool xgty = squareSizex > squareSizey; // 矩形的大小
    if (squareSizex < 100)
    {
        squareSizex = 100;
    }
    if (squareSizey < 100)
    {
        squareSizey = 100;
    }

    //    std::cout<<"winSize"<<" "<<winSize.x<<" "<<winSize.y<<std::endl;

    try
    {
        {
            // ImGui::BeginGroup();
            //            int numCols = ceil(sqrt(appState.measure_boxes.size())); // 列数
            appState.imageProcess3.resize(appState.measure_boxes.size());
            for (int i = 0; i < appState.measure_boxes.size(); i++)
            {
                if (i % numCols > 0)
                    ImGui::SameLine();
                if (appState.needRefresh)
                {
                    if (appState.pin_info.size() > 0 &&
                        appState.imageProcess2.size() == appState.measure_boxes.size() &&
                        i < appState.imageProcess2.size() && !appState.imageProcess2[i].empty())
                    {
                        auto connectedIndices = findMatchingRelationships2(appState.pin_info[i],
                                                                           false);
                        appState.connectedIndices = connectedIndices;

                        appState.imageProcess2[i].copyTo(appState.imageProcess3[i]);

                        drawMeasureDistances(appState.imageProcess3[i], appState.pin_info[i], appState.connectedIndices,
                                             appState.pin_measure_settings, i, appState.mmPrePixel);
                        drawMeasureAngles(appState.imageProcess3[i], appState.pin_info[i], appState.connectedIndices,
                                          appState.angle_measure_settings, i);

                        if (appState.needMeasure == 1)
                        {
                            std::string jsonString = MeasureToJson(appState.scanResVec, appState.pin_measure_settings,
                                                                   appState.angle_measure_settings);
                            std::string date = getCurrentDate(); // Get the current date and time

                            dbHandler.insertJsonData(date, jsonString);
                            appState.needMeasure = 2;
                        }
                    }
                }

                if (appState.immvisionParamsStaticSub.size() == appState.measure_boxes.size() &&
                    !appState.imageProcess3[i].empty())
                {
                    if (!xgty)
                    {
                        appState.immvisionParamsStaticSub[i].ImageDisplaySize = cv::Size((int)squareSizex, 0);
                    }
                    else
                    {
                        appState.immvisionParamsStaticSub[i].ImageDisplaySize = cv::Size(0, (int)squareSizey);
                    }

                    //                    appState.immvisionParamsStaticSub[i].ImageDisplaySize.width = (int) imageSize.y / 2 ;
                    //                ImmVision::Image("Process2", appState.imageProcess2, &appState.immvisionParamsSub);
                    ImmVision::Image("框:" + std::to_string(i + 1), appState.imageProcess3[i],
                                     &appState.immvisionParamsStaticSub[i]);
                    appState.immvisionParamsStaticSub[i].RefreshImage = true;
                }
            }

            for (int i = 0; i < appState.measure_boxes.size(); i++)
            {
                if (i % numCols > 0)
                    ImGui::SameLine();

                if (appState.immvisionParamsStaticSub.size() == appState.measure_boxes.size() &&
                    !appState.imageProcess4[i].empty())
                {

                    // 应用查找表进行非线性提亮处理
                    cv::Mat enhanced_image;
                    //                    appState.immvisionParamsStaticSub[i].ImageDisplaySize.width = (int) imageSize.y / 2 ;
                    //                ImmVision::Image("Process2", appState.imageProcess2, &appState.immvisionParamsSub);
                    ImmVision::Image("接口:" + std::to_string(i + 1), appState.imageProcess4[i],
                                     &appState.immvisionParamsStaticSub[i]);
                    appState.immvisionParamsStaticSub[i].RefreshImage = true;
                }
            }
            // ImGui::EndGroup();
            if (!appState.needRefresh)
            {
                if (appState.immvisionParamsStaticSub.size() == appState.measure_boxes.size())
                    for (int i = 0; i < appState.measure_boxes.size(); i++)
                    {
                        appState.immvisionParamsStaticSub[i].RefreshImage = false;
                    }
                appState.immvisionParamsStatic.RefreshImage = false;
            }
            if (appState.needRefresh && appState.algorithmFinished)
            {
                appState.needRefresh = false;
            }
        }
    }
    catch (const std::exception &e)
    {
        // 在此处处理异常
        std::cout << "ResView时发生异常: " << e.what() << std::endl;
    }

    ImmVision::ImageWidgets::s_CollapsingHeader_CacheState_Sync = false;
}

void MyLoadFonts()
{

    //    HelloImGui::ImGuiDefaultSettings::LoadDefaultFont_WithFontAwesomeIcons(); // The font that is loaded first is the default font
    ImGuiIO &io = ImGui::GetIO();
    //    HelloImGui::SetAssetsFolder("/home/nn/Documents/immvision/src/demos_immvision/calib_camera/assets");
    HelloImGui::LoadFontTTF_WithFontAwesomeIcons("fonts/MiSans-Normal.ttf", 16.f,
                                                 io.Fonts->GetGlyphRangesJapanese()); // will be loaded from the assets folder
}

// The Gui of the status bar
void StatusBarGui(AppState &app_state)
{
    ImGui::Text("Using backend: %s", HelloImGui::GetBackendDescription().c_str());
    ImGui::SameLine();
    //    if (app_state.rocket_state == AppState::RocketState::Preparing)
    //    {
    //        ImGui::Text("  -  Rocket completion: ");
    //        ImGui::SameLine();
    //        ImGui::ProgressBar(app_state.rocket_progress, HelloImGui::EmToVec2(7.0f, 1.0f));
    //    }
}

void ShowAboutWindow(bool *p_open)
{
    if (*p_open)
        ImGui::OpenPopup("关于");
    ImGui::SetNextWindowSize(ImVec2(400, 200)); // 设置窗口大小为宽度400，高度200
    if (ImGui::BeginPopupModal("关于", nullptr, ImGuiWindowFlags_MenuBar))
    {

        ImGui::Text("远峰科技 2024");
        ImGui::Separator();
        if (ImGui::Button("关闭"))
        {
            ImGui::CloseCurrentPopup();
            *p_open = false;
        }
        ImGui::EndPopup();
    }
}

// The menu gui
void ShowMenuGui(HelloImGui::RunnerParams &runnerParams, bool &show_tool_about, AppState &appState)
{
    HelloImGui::ShowAppMenu(runnerParams);
    HelloImGui::ShowViewMenu(runnerParams);

    if (ImGui::BeginMenu("配置"))
    {
        bool clicked = ImGui::MenuItem("打开全局配置", "", false);
        if (clicked)
        {
            std::string default_path;
            std::vector<std::string> filters = {"xml Config Files", "*.xml"};
            pfd::open_file dialog("打开全局配置", default_path, filters,
                                  pfd::opt::none);
            if (dialog.result().size() > 0)
            {
                std::cout << "open:" << dialog.result()[0] << std::endl;
                appState.configPath = dialog.result()[0];
                appState.loadConfigFromXml(dialog.result()[0]);
            }
        }
        bool clicked2 = ImGui::MenuItem("保存全局配置", "", false);
        if (clicked2)
        {
            if (appState.configPath.empty())
            {
                appState.saveConfigToXml("aoi_config.xml");
            }
            else
            {
                appState.saveConfigToXml(appState.configPath);
            }
        }
        ImGui::EndMenu();
    }
    ShowAboutWindow(&show_tool_about);
}

void ShowAppMenuItems(bool &show_tool_about)
{
    if (ImGui::MenuItem("关于应用", nullptr, &show_tool_about))
        ;
    //    if (ImGui::MenuItem("打开配置", nullptr,&show_tool_about));
}

void ShowTopToolbar(AppState &appState)
{
    //    ImGui::PushFont(appState.LargeIconFont);
    //    if (ImGui::Button(ICON_FA_POWER_OFF))
    //        HelloImGui::GetRunnerParams()->appShallExit = true;
    //
    //    ImGui::SameLine(ImGui::GetWindowWidth() - HelloImGui::EmSize(7.f));
    //    if (ImGui::Button(ICON_FA_HOUSE))
    //        HelloImGui::Log(HelloImGui::LogLevel::Info, "Clicked on Home in the top toolbar");
    //    ImGui::SameLine();
    //    if (ImGui::Button(ICON_FA_FLOPPY_DISK))
    //        HelloImGui::Log(HelloImGui::LogLevel::Info, "Clicked on Save in the top toolbar");
    //    ImGui::SameLine();
    //    if (ImGui::Button(ICON_FA_ADDRESS_BOOK))
    //        HelloImGui::Log(HelloImGui::LogLevel::Info, "Clicked on Address Book in the top toolbar");
    //
    //    ImGui::SameLine(ImGui::GetWindowWidth() - HelloImGui::EmSize(2.f));
    //    ImGui::Text(ICON_FA_BATTERY_THREE_QUARTERS);
    //    ImGui::PopFont();
}

void ShowRightToolbar(AppState &appState)
{
    //    ImGui::PushFont(appState.LargeIconFont);
    //    if (ImGui::Button(ICON_FA_CIRCLE_ARROW_LEFT))
    //        HelloImGui::Log(HelloImGui::LogLevel::Info, "Clicked on Circle left in the right toolbar");
    //
    //    if (ImGui::Button(ICON_FA_CIRCLE_ARROW_RIGHT))
    //        HelloImGui::Log(HelloImGui::LogLevel::Info, "Clicked on Circle right in the right toolbar");
    //    ImGui::PopFont();
}

//////////////////////////////////////////////////////////////////////////
//    Docking Layouts and Docking windows
//////////////////////////////////////////////////////////////////////////

//
// 1. Define the Docking splits (two versions are available)
//
std::vector<HelloImGui::DockingSplit> CreateDefaultDockingSplits()
{
    //    Define the default docking splits,
    //    i.e. the way the screen space is split in different target zones for the dockable windows
    //     We want to split "MainDockSpace" (which is provided automatically) into three zones, like this:
    //
    //    ___________________________________________
    //    |        |                                |
    //    | Command|                                |
    //    | Space  |    MainDockSpace               |
    //    |        |                                |
    //    |        |                                |
    //    |        |                                |
    //    -------------------------------------------
    //    |     MiscSpace                           |
    //    -------------------------------------------
    //

    // Then, add a space named "MiscSpace" whose height is 25% of the app height.
    // This will split the preexisting default dockspace "MainDockSpace" in two parts.
    HelloImGui::DockingSplit splitMainMisc;
    splitMainMisc.initialDock = "MainDockSpace";
    splitMainMisc.newDock = "MiscSpace";
    splitMainMisc.direction = ImGuiDir_Right;
    splitMainMisc.ratio = 0.2f;

    // Then, add a space to the left which occupies a column whose width is 25% of the app width
    HelloImGui::DockingSplit splitMainCommand;
    splitMainCommand.initialDock = "MainDockSpace";
    splitMainCommand.newDock = "CommandSpace";
    splitMainCommand.direction = ImGuiDir_Left;
    splitMainCommand.ratio = 0.15f;

    HelloImGui::DockingSplit splitMainTable;
    splitMainTable.initialDock = "MainDockSpace";
    splitMainTable.newDock = "TableSpace";
    splitMainTable.direction = ImGuiDir_Right;
    splitMainTable.ratio = 0.25f;

    HelloImGui::DockingSplit splitMainTable2;
    splitMainTable2.initialDock = "MainDockSpace";
    splitMainTable2.newDock = "TableSpace2";
    splitMainTable2.direction = ImGuiDir_Right;
    splitMainTable2.ratio = 0.3f;

    std::vector<HelloImGui::DockingSplit> splits{splitMainMisc, splitMainCommand, splitMainTable, splitMainTable2};
    return splits;
}

std::vector<HelloImGui::DockingSplit> CreateAlternativeDockingSplits()
{
    //    Define alternative docking splits for the "Alternative Layout"
    //    ___________________________________________
    //    |                |                        |
    //    | Misc           |                        |
    //    | Space          |    MainDockSpace       |
    //    |                |                        |
    //    -------------------------------------------
    //    |                                         |
    //    |                                         |
    //    |     CommandSpace                        |
    //    |                                         |
    //    -------------------------------------------

    HelloImGui::DockingSplit splitMainCommand;
    splitMainCommand.initialDock = "MainDockSpace";
    splitMainCommand.newDock = "CommandSpace";
    splitMainCommand.direction = ImGuiDir_Right;
    splitMainCommand.ratio = 0.25f;

    HelloImGui::DockingSplit splitMainMisc;
    splitMainMisc.initialDock = "MainDockSpace";
    splitMainMisc.newDock = "MiscSpace";
    splitMainMisc.direction = ImGuiDir_Left;
    splitMainMisc.ratio = 0.15f;

    HelloImGui::DockingSplit splitMainTable;
    splitMainTable.initialDock = "MainDockSpace";
    splitMainTable.newDock = "TableSpace";
    splitMainTable.direction = ImGuiDir_Right;
    splitMainTable.ratio = 0.25f;

    HelloImGui::DockingSplit splitMainTable2;
    splitMainTable2.initialDock = "MainDockSpace";
    splitMainTable2.newDock = "TableSpace2";
    splitMainTable2.direction = ImGuiDir_Right;
    splitMainTable2.ratio = 0.25f;

    std::vector<HelloImGui::DockingSplit> splits{splitMainCommand, splitMainMisc, splitMainTable, splitMainTable2};
    return splits;
}

//
// 2. Define the Dockable windows
//
std::vector<HelloImGui::DockableWindow> CreateDockableWindows(AppState &appState)
{
    // A window named "FeaturesDemo" will be placed in "CommandSpace". Its Gui is provided by "GuiWindowDemoFeatures"
    HelloImGui::DockableWindow configWindow;
    configWindow.label = "配置";
    configWindow.dockSpaceName = "CommandSpace";
    configWindow.GuiFunction = [&]
    { AOIParams(appState); };

    // A layout customization window will be placed in "MainDockSpace". Its Gui is provided by "GuiWindowLayoutCustomization"
    HelloImGui::DockableWindow imageWindow;
    imageWindow.label = "查看";
    imageWindow.dockSpaceName = "MiscSpace";
    imageWindow.GuiFunction = [&appState]()
    { ImageView(appState); };

    // A Log window named "Logs" will be placed in "MiscSpace". It uses the HelloImGui logger gui
    HelloImGui::DockableWindow resultWindow;
    resultWindow.label = "结果";
    resultWindow.dockSpaceName = "MainDockSpace";
    resultWindow.GuiFunction = [&appState]
    {
        ResView(appState);
        //        HelloImGui::LogGui();
    };

    // A Window named "Dear ImGui Demo" will be placed in "MainDockSpace"
    HelloImGui::DockableWindow measureWindow;
    measureWindow.label = "距离测量";
    measureWindow.dockSpaceName = "TableSpace";
    //    measureWindow.imGuiWindowFlags = ImGuiWindowFlags_MenuBar;
    measureWindow.GuiFunction = [&appState]
    {
        DistMeasureTable(appState);
        //        ImGui::ShowDemoWindow();
    };
    HelloImGui::DockableWindow measureWindow2;
    measureWindow2.label = "角度测量";
    measureWindow2.dockSpaceName = "TableSpace2";
    //    measureWindow.imGuiWindowFlags = ImGuiWindowFlags_MenuBar;
    measureWindow2.GuiFunction = [&appState]
    {
        AngleMeasureTable(appState);
        //        ImGui::ShowDemoWindow();
    };

    // additionalWindow is initially not visible (and not mentioned in the view menu).
    // it will be opened only if the user chooses to display it
    HelloImGui::DockableWindow additionalWindow;
    additionalWindow.label = "Additional Window";
    additionalWindow.isVisible = false;           // this window is initially hidden,
    additionalWindow.includeInViewMenu = false;   // it is not shown in the view menu,
    additionalWindow.rememberIsVisible = false;   // its visibility is not saved in the settings file,
    additionalWindow.dockSpaceName = "MiscSpace"; // when shown, it will appear in BottomSpace.
    additionalWindow.GuiFunction = []
    {
        ImGui::Text("This is the additional window");
        //        ImGuiIO &io = ImGui::GetIO();
        // static std::string item_string ;
        //
        // #ifdef IMGUI_DISABLE_OBSOLETE_KEYIO
        //        struct funcs { static bool IsLegacyNativeDupe(ImGuiKey) { return false; } };
        //            ImGuiKey start_key = ImGuiKey_NamedKey_BEGIN;
        // #else
        //        struct funcs {
        //            static bool IsLegacyNativeDupe(ImGuiKey key) {
        //                return key >= 0 && key < 512 && ImGui::GetIO().KeyMap[key] != -1;
        //            }
        //        }; // Hide Native<>ImGuiKey duplicates when both exists in the array
        //        ImGuiKey start_key = (ImGuiKey) 0;
        // #endif
        //        ImGui::Text("Keys down:");
        //        for (ImGuiKey key = start_key; key < ImGuiKey_NamedKey_END; key = (ImGuiKey) (key + 1)) {
        //            if (funcs::IsLegacyNativeDupe(key) || !ImGui::IsKeyDown(key))continue;
        //            ImGui::SameLine();
        //            ImGui::Text((key < ImGuiKey_NamedKey_BEGIN) ? "\"%s\"" : "\"%s\" %d", ImGui::GetKeyName(key), key);
        //            if(key == ImGuiKey_Enter){
        //                completeData = item_string;
        //                item_string="";
        //            }else if(key == ImGuiKey_LeftShift || key == ImGuiKey_ModShift) {
        //
        //            }else{
        ////                ImWchar c = ImGui::GetIO().KeyMap[key];
        //////                if(c > ' ' && c <= 255)
        ////                c = std::tolower(c);
        //                auto str = ImGui::GetKeyName(key);
        //                item_string += str;
        //            }
        //        }
    };

    std::vector<HelloImGui::DockableWindow> dockableWindows{
            configWindow,
            imageWindow,
            resultWindow,
            measureWindow,
            measureWindow2,
            additionalWindow,
    };
    return dockableWindows;
}

//
// 3. Define the layouts:
//        A layout is stored inside DockingParams, and stores the splits + the dockable windows.
//        Here, we provide the default layout, and two alternative layouts.
//
HelloImGui::DockingParams CreateDefaultLayout(AppState &appState)
{
    HelloImGui::DockingParams dockingParams;
    // dockingParams.layoutName = "Default"; // By default, the layout name is already "Default"
    dockingParams.dockingSplits = CreateDefaultDockingSplits();
    dockingParams.dockableWindows = CreateDockableWindows(appState);
    return dockingParams;
}

std::vector<HelloImGui::DockingParams> CreateAlternativeLayouts(AppState &appState)
{
    HelloImGui::DockingParams alternativeLayout;
    {
        alternativeLayout.layoutName = "Alternative Layout";
        alternativeLayout.dockingSplits = CreateAlternativeDockingSplits();
        alternativeLayout.dockableWindows = CreateDockableWindows(appState);
    }
    HelloImGui::DockingParams tabsLayout;
    {
        tabsLayout.layoutName = "Tabs Layout";
        tabsLayout.dockableWindows = CreateDockableWindows(appState);
        // Force all windows to be presented in the MainDockSpace
        for (auto &window : tabsLayout.dockableWindows)
            window.dockSpaceName = "MainDockSpace";
        // In "Tabs Layout", no split is created
        tabsLayout.dockingSplits = {};
    }
    return {alternativeLayout, tabsLayout};
}

int main()
{
    try
    {
        // #############################################################################################
        //  Part 1: Define the application state, fill the status and menu bars, load additional font
        // #############################################################################################
        std::cout << "SQlite3 version " << SQLite::VERSION << " (" << SQLite::getLibVersion() << ")" << std::endl;
        std::cout << "SQliteC++ version " << SQLITECPP_VERSION << std::endl;

        // Our application state
        AppState appState;
        KeyInputListener listener([&appState](const std::vector<std::string> &input)
                                  {
                                      int sz = appState.measure_boxes.size();
                                      int inputSize = input.size();
//        appState.scanResVec.resize(inputSize);
                                      if (inputSize >= sz) {
                                          appState.scanResVec.assign(input.end() - sz, input.end());
                                      } else {
                                          appState.scanResVec.assign(input.begin(), input.end());
                                      } },
                                  [&appState](const std::string &input)
                                  {
                                      appState.scanResJson = input;
                                  },
                                  &scannerIsRunning);

        listener.start();
        // listener.startIfNotRunning();

        appState.loadBoxSettings();
        appState.last_measure_boxes.resize(appState.measure_boxes.size());
        //        appState.loadPairSettings();
        //        appState.loadAngleSettings();
        //        appState.loadLoginInfo();
        //        appState.loadCalibrationParameters();
        appState.loadConfigFromXml("aoi_config.xml");

        // camera
        cameraController.SetResolution(4056, 3040);
        cameraController.ToggleCamera(0);
        appState.image_points_seq.clear();
        appState.object_points.clear();
        appState.paramsSummary.clear();

        // Hello ImGui params (they hold the settings as well as the Gui callbacks)
        HelloImGui::RunnerParams runnerParams;
        runnerParams.appWindowParams.windowTitle = "AOI PINs";
        runnerParams.imGuiWindowParams.menuAppTitle = "AOI";
        //    runnerParams.appWindowParams.windowGeometry.size = {1200, 1000};
        // runnerParams.appWindowParams.windowGeometry.fullScreenMode = HelloImGui::FullScreenMode::FullScreenDesktopResolution;
        runnerParams.appWindowParams.windowGeometry.fullScreenMode = HelloImGui::FullScreenMode::FullMonitorWorkArea;
        runnerParams.appWindowParams.windowGeometry.position = {0, 0};
        runnerParams.appWindowParams.restorePreviousGeometry = true;

        // Our application uses a borderless window, but is movable/resizable
        //    runnerParams.appWindowParams.borderless = true;
        //    runnerParams.appWindowParams.borderlessMovable = true;
        //    runnerParams.appWindowParams.borderlessResizable = true;
        //    runnerParams.appWindowParams.borderlessClosable = true;

        // Load additional font
        runnerParams.callbacks.LoadAdditionalFonts = []()
        { MyLoadFonts(); };

        //
        // Status bar
        //
        // We use the default status bar of Hello ImGui
        runnerParams.imGuiWindowParams.showStatusBar = true;
        // uncomment next line in order to hide the FPS in the status bar
        // runnerParams.imGuiWindowParams.showStatusFps = false;
        runnerParams.callbacks.ShowStatus = [&appState]()
        { StatusBarGui(appState); };

        //
        // Menu bar
        //
        // Here, we fully customize the menu bar:
        // by setting `showMenuBar` to true, and `showMenu_App` and `showMenu_View` to false,
        // HelloImGui will display an empty menu bar, which we can fill with our own menu items via the callback `ShowMenus`
        runnerParams.imGuiWindowParams.showMenuBar = true;
        runnerParams.imGuiWindowParams.showMenu_App = false;
        runnerParams.imGuiWindowParams.showMenu_View = false;
        // Inside `ShowMenus`, we can call `HelloImGui::ShowViewMenu` and `HelloImGui::ShowAppMenu` if desired
        static bool show_tool_about = false;
        runnerParams.callbacks.ShowMenus = [&runnerParams, &appState]()
        {
            ShowMenuGui(runnerParams, show_tool_about, appState);
        };
        // Optional: add items to Hello ImGui default App menu

        runnerParams.callbacks.ShowAppMenuItems = []()
        { ShowAppMenuItems(show_tool_about); };

        //
        // Top and bottom toolbars
        //
        // toolbar options
        HelloImGui::EdgeToolbarOptions edgeToolbarOptions;
        edgeToolbarOptions.sizeEm = 2.5f;
        edgeToolbarOptions.WindowBg = ImVec4(0.8f, 0.8f, 0.8f, 0.35f);
        // top toolbar
        //    runnerParams.callbacks.AddEdgeToolbar(
        //            HelloImGui::EdgeToolbarType::Top,
        //            [&appState]() { ShowTopToolbar(appState); },
        //            edgeToolbarOptions
        //    );
        // right toolbar
        edgeToolbarOptions.WindowBg.w = 0.4f;
        //    runnerParams.callbacks.AddEdgeToolbar(
        //            HelloImGui::EdgeToolbarType::Right,
        //            [&appState]() { ShowRightToolbar(appState); },
        //            edgeToolbarOptions
        //    );

        //
        // Load user settings at `PostInit` and save them at `BeforeExit`
        //
        //    runnerParams.callbacks.PostInit = [&appState]   { LoadMyAppSettings(appState);};
        //    runnerParams.callbacks.BeforeExit = [&appState] { SaveMyAppSettings(appState);};

        //
        // Change style
        //
        // 1. Change theme
        auto &tweakedTheme = runnerParams.imGuiWindowParams.tweakedTheme;
        tweakedTheme.Theme = ImGuiTheme::ImGuiTheme_MaterialFlat;
        tweakedTheme.Tweaks.Rounding = 10.f;
        // 2. Customize ImGui style at startup
        runnerParams.callbacks.SetupImGuiStyle = []()
        {
            // Reduce spacing between items ((8, 4) by default)
            ImGui::GetStyle().ItemSpacing = ImVec2(6.f, 4.f);
        };

        // ###############################################################################################
        //  Part 2: Define the application layout and windows
        // ###############################################################################################

        // First, tell HelloImGui that we want full screen dock space (this will create "MainDockSpace")
        runnerParams.imGuiWindowParams.defaultImGuiWindowType = HelloImGui::DefaultImGuiWindowType::ProvideFullScreenDockSpace;
        // In this demo, we also demonstrate multiple viewports: you can drag windows outside out the main window in order to put their content into new native windows
        runnerParams.imGuiWindowParams.enableViewports = true;
        // Set the default layout
        runnerParams.dockingParams = CreateDefaultLayout(appState);
        // Add alternative layouts
        runnerParams.alternativeDockingLayouts = CreateAlternativeLayouts(appState);

        // uncomment the next line if you want to always start with the layout defined in the code
        //     (otherwise, modifications to the layout applied by the user layout will be remembered)
        // runnerParams.dockingParams.layoutCondition = HelloImGui::DockingLayoutCondition::ApplicationStart;

        // ###############################################################################################
        //  Part 3: Where to save the app settings
        // ###############################################################################################
        //  By default, HelloImGui will save the settings in the current folder. This is convenient when developing,
        //  but not so much when deploying the app.
        //      You can tell HelloImGui to save the settings in a specific folder: choose between
        //          CurrentFolder
        //          AppUserConfigFolder
        //          AppExecutableFolder
        //          HomeFolder
        //          TempFolder
        //          DocumentsFolder
        //
        //      Note: AppUserConfigFolder is:
        //          AppData under Windows (Example: C:\Users\[Username]\AppData\Roaming)
        //          ~/.config under Linux
        //          "~/Library/Application Support" under macOS or iOS
        runnerParams.iniFolderType = HelloImGui::IniFolderType::AppUserConfigFolder;

        // runnerParams.iniFilename: this will be the name of the ini file in which the settings
        // will be stored.
        // In this example, the subdirectory Docking_Demo will be created under the folder defined
        // by runnerParams.iniFolderType.
        //
        // Note: if iniFilename is left empty, the name of the ini file will be derived
        // from appWindowParams.windowTitle
        runnerParams.iniFilename = "Docking_AOI.ini";

        // ###############################################################################################
        //  Part 4: Run the app
        // ###############################################################################################
        HelloImGui::DeleteIniSettings(runnerParams);

        // Optional: choose the backend combination
        // ----------------------------------------
        //    runnerParams.platformBackendType = HelloImGui::PlatformBackendType::Sdl;
        //    runnerParams.rendererBackendType = HelloImGui::RendererBackendType::OpenGL3;

        HelloImGui::Run(runnerParams); // Note: with ImGuiBundle, it is also possible to use ImmApp::Run(...)
    }
    catch (const std::exception &e)
    {
        // 在此处处理异常
        std::cout << "main发生异常: " << e.what() << std::endl;
    }
    return 0;
};