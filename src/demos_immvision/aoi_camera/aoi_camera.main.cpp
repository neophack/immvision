#include "immvision/immvision.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <thread>
#include <atomic>
#include <iostream>
#include <filesystem>
#include <unordered_set>
#include <mutex>
#include <string>
#include <numeric>
#include <fstream>
#include "imgui_freetype.h"

//#include "deep_model.h"
//#include "cmdline.h"
//#include "utils.h"
#include "immvision/internal/misc/portable_file_dialogs.h"
#include "immvision/internal/image_cache.h"

extern unsigned char font_data[];
extern size_t font_data_size;

// Poor man's fix for C++ late arrival in the unicode party:
//    - C++17: "my string" is of type const char*
//    - C++20: "my string" is of type const char8_t*
// However, ImGui text functions expect const char*.
#ifdef __cpp_char8_t
#define U8_TO_CHAR(x) reinterpret_cast<const char*>(x)
#else
#define U8_TO_CHAR(x) x
#endif
// And then, we need to tell gcc to stop validating format string (it gets confused by the "" string)
#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wformat"
#endif

// 常见分辨率
std::vector<std::pair<int, int>> resolutions = {
        {800,  600},
        {1024, 768},
        {1280, 720},
        {1920, 1080},
        {2560, 1440},
        {3840, 2160},
        {4056, 3040}
};

enum Pattern {
    NOT_EXISTING, CHESSBOARD, CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID, CHARUCO
};

enum CAMERA_MODEL {
    PINHOLE,
    FISHEYE
};

static int calibstatus = 0;

std::string ResourcesDir() {
    std::filesystem::path this_file(__FILE__);
    return (this_file.parent_path().parent_path() / "resources").string();
}

enum class Orientation {
    Horizontal,
    Vertical
};



class CameraController {
public:
    CameraController() : isRunning(false) {}

    ~CameraController() {
        Stop();
    }

    std::vector<int> GetAvailableCameras() {
        std::vector<int> availableCameras;
        if (!isRunning) {
            for (int i = 0; i < 10; i++) {
                cv::VideoCapture cap(i);
                if (cap.isOpened()) {
                    availableCameras.push_back(i);
                    cap.release();
                }
            }
        }
        return availableCameras;
    }

    std::pair<int, int> GetResolution(int cam) {
        if (!isRunning) {
            cv::VideoCapture cap(cam);
            if (cap.isOpened()) {
                int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
                int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
                cap.release();
                return std::make_pair(width, height);
            }
        }
        return std::make_pair(-1, -1);
    }

    double GetFrameRate(int cam) {
        if (!isRunning) {
            cv::VideoCapture cap(cam);
            if (cap.isOpened()) {
                double frameRate = cap.get(cv::CAP_PROP_FPS);
                cap.release();
                return frameRate;
            }
        }
        return -1.0;
    }

    void Start(int cam) {
        if (!isRunning) {
            isRunning = true;
            cameraThread = std::thread(&CameraController::CameraThreadFunc, this, cam);
        }
    }

    void Stop() {
        if (isRunning) {
            isRunning = false;
            cameraThread.join();
        }
    }

    void ToggleCamera(int cam) {
        if (isRunning) {
            Stop();
        } else {
            Start(cam);
        }
    }

    cv::Mat GetFrame() {
        std::lock_guard<std::mutex> lock(frameMutex);
        return frame;
    }

    bool SetResolution(int widtht, int heightt) {
        if (!isRunning) {
            width = widtht;
            height = heightt;
            return true;

        }
        return false;
    }

    std::atomic<bool> isRunning;
private:
    cv::Mat frame;

    std::thread cameraThread;
    std::mutex frameMutex;
    int width;
    int height;


    void CameraThreadFunc(int cam) {
        cv::VideoCapture cap(cam);
        if (!cap.isOpened()) {
            std::cout << "Failed to open the camera." << std::endl;
            return;
        }
        cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);

        while (isRunning) {
            cv::Mat currentFrame;
            cap >> currentFrame;

            {
                std::lock_guard<std::mutex> lock(frameMutex);
                frame = currentFrame;
            }
        }
    }
};


struct AppState {
    cv::Mat image;
    cv::Mat imageDraw;
    cv::Mat imageCap;
    cv::Mat imageProcess;
    cv::Mat imageProcess2;
    cv::Mat imageProcess3;
    cv::Mat imageSobel;
//    SobelParams sobelParams;

    ImmVision::ImageParams immvisionParams;
    ImmVision::ImageParams immvisionParamsStatic;
    ImmVision::ImageParams immvisionParamsStaticSub;
    CameraController cameraController;  // Moved cameraController declaration here

    int validcam;
    int seli = 4;
    int width;
    int height;

    int calibPatten = CHESSBOARD;
    int boardRows = 11;
    int boardCols = 8;
    int gridMM = 3;

    std::vector<std::tuple<std::string, double, double, double>> paramsSummary;

    std::string savePath = "./aoi_pos.yml";
    std::string savePathPair = "./pin_pair.bin";

    int calibModel = 0;
    std::vector<cv::Point2f> image_points_buf;
    std::vector<std::vector<cv::Point2f>> image_points_seq;
    std::vector<std::vector<cv::Point3f>> object_points;

    cv::Mat intrinsics_in;
    cv::Mat dist_coeffs;
    double reproj_err;

    std::vector<std::vector<int>> circle_info;

    std::vector<std::vector<int>> pin_info;
    std::vector<int> connectedIndices;

    int large_circle_id;

    float rect_x = 0.f;
    float rect_y = 0.f;
    float rect_w = 0.f;
    float rect_h = 0.f;

    float bin_threshold = 127.f;

    float relativeRectX = 0.f;
    float relativeRectY = 0.f;
    float relativeRectW = 0.f;
    float relativeRectH = 0.f;

    struct MEASURE_SETTING {
        int pin0 = 0;
        int pin1 = 1;
        float minDistance = 30.;
        float maxDistance = 60.;
        float distance = -1.;
    };

    bool needRefresh = false;

    std::vector<MEASURE_SETTING> pin_measure_settings;

    AppState() : cameraController() {
//        cameraController.Start();
        image = cv::Mat(600, 800, CV_8UC3, cv::Scalar(0, 0, 0));

//        sobelParams = SobelParams();
//        imageSobel = ComputeSobel(image, sobelParams);

        immvisionParams = ImmVision::ImageParams();
        immvisionParams.ImageDisplaySize = cv::Size(500, 0);
        immvisionParams.ZoomKey = "z";
        immvisionParams.RefreshImage = true;
        immvisionParams.forceFullView = true;
        immvisionParams.ShowZoomButtons = false;

        immvisionParamsStatic = ImmVision::ImageParams();
        immvisionParamsStatic.ImageDisplaySize = cv::Size(500, 0);
        immvisionParamsStatic.ZoomKey = "z";
        immvisionParamsStatic.RefreshImage = false;
        immvisionParamsStatic.forceFullView = true;
        immvisionParamsStatic.ShowZoomButtons = false;

        immvisionParamsStaticSub = ImmVision::ImageParams();
//        immvisionParamsSub.ImageDisplaySize = cv::Size(0, 0);
        immvisionParamsStaticSub.ZoomKey = "c";
//        immvisionParamsSub.ShowOptionsPanel = true;
        immvisionParamsStaticSub.RefreshImage = false;
        immvisionParamsStaticSub.forceFullView = true;
        immvisionParamsStaticSub.ShowZoomButtons = false;
//        immvisionParamsSub.ShowOptionsInTooltip = false;
        immvisionParamsStaticSub.ShowOptionsButton = false;
    }


    void savePairSettings() {
        std::ofstream out(savePathPair, std::ios::binary);
        if (out.is_open()) {
            size_t size = pin_measure_settings.size();
            out.write((char *) &size, sizeof(size));
            out.write((char *) pin_measure_settings.data(), size * sizeof(MEASURE_SETTING));
            out.close();
        }
    }

    void loadPairSettings() {
        std::ifstream in(savePathPair, std::ios::binary);
        if (in.is_open()) {
            size_t size;
            in.read((char *) &size, sizeof(size));
            pin_measure_settings.resize(size);
            in.read((char *) pin_measure_settings.data(), size * sizeof(MEASURE_SETTING));
            in.close();
        }
    }
};


int getLargerCircleIfRadiusDouble(AppState &appState) {
    // 确保至少有两个圆
    if (appState.circle_info.size() != 2) {
        return -1;
    }

    // 获取两个圆的信息
    std::vector<int> circle1 = appState.circle_info[0];
    std::vector<int> circle2 = appState.circle_info[1];

    // 计算两个圆的半径
    int radius1 = circle1[2] / 2;
    int radius2 = circle2[2] / 2;

    // 检查半径是否接近两倍
    if (std::abs(static_cast<double>(radius1) / radius2 - 2.0) <= 0.5 ||
        std::abs(static_cast<double>(radius2) / radius1 - 2.0) <= 0.5) {
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

double calculateDistance(const cv::Point &p1, const cv::Point &p2) {
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Function to calculate the angle between three points
double calculateAngle(const cv::Point &p1, const cv::Point &p2, const cv::Point &p3) {
    cv::Point v1 = p2 - p1;
    cv::Point v2 = p3 - p1;
    double angle = std::atan2(v2.y, v2.x) - std::atan2(v1.y, v1.x);
    angle = angle * 180.0 / CV_PI; // Convert to degree
    if (angle < 0) angle += 360.0; // Make it positive
    if (angle > 180) angle = 360 - angle; // Make it less than or equal to 180
    return angle;
}

std::vector<int> findMatchingRelationships(const std::vector<std::vector<int>> &pin_info) {
    cv::Scalar color(255, 0, 0);
    int thickness = 1;
    int font = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.3;
    int baseline = 0;

    // Create a vector to store the matching relationships
    std::vector<int> matching(pin_info.size(), -1);
    std::vector<int> matching2(pin_info.size(), -1);

    // Find the matching relationships
    for (size_t i = 0; i < pin_info.size(); ++i) {
        cv::Point p1(pin_info[i][0], pin_info[i][1]);

        // Initialize minDistance and secondMinDistance with maximum possible value
        double minDistance = std::numeric_limits<double>::max();
        double secondMinDistance = std::numeric_limits<double>::max();
        cv::Point nearestPoint;
        cv::Point secondNearestPoint;
        int lastNearInd = -1;

        // Find the nearest and second nearest point
        for (size_t j = 0; j < pin_info.size(); ++j) {
            if (i == j) continue; // Skip the same point
            cv::Point p2(pin_info[j][0], pin_info[j][1]);
            double distance = calculateDistance(p1, p2);
            if (distance < minDistance) {
                secondMinDistance = minDistance;
                secondNearestPoint = nearestPoint;
                minDistance = distance;
                nearestPoint = p2;
                matching[i] = j;

                // Calculate the angle
                cv::Point v1 = nearestPoint - p1;
                cv::Point v2 = secondNearestPoint - p1;
                double angle = std::atan2(v2.y, v2.x) - std::atan2(v1.y, v1.x);
                angle = angle * 180.0 / CV_PI; // Convert to degree
                if (angle < 0) angle += 360.0; // Make it positive
                if (angle > 180) angle = 360 - angle; // Make it less than or equal to 180
                if (angle > 70) {
                    matching2[i] = lastNearInd;
                }

                lastNearInd = j;
            } else if (distance < secondMinDistance) {
                secondMinDistance = distance;
                secondNearestPoint = p2;

                // Calculate the angle
                cv::Point v1 = nearestPoint - p1;
                cv::Point v2 = secondNearestPoint - p1;
                double angle = std::atan2(v2.y, v2.x) - std::atan2(v1.y, v1.x);
                angle = angle * 180.0 / CV_PI; // Convert to degree
                if (angle < 0) angle += 360.0; // Make it positive
                if (angle > 180) angle = 360 - angle; // Make it less than or equal to 180
                if (angle > 70) {
                    matching2[i] = j;
                }

            }
        }

    }

    std::vector<int> connectedIndices;
    int startIndex = -1;

// 找到一个端点作为起始点
    for (size_t i = 0; i < pin_info.size(); ++i) {
        if (matching2[i] == -1) {
            startIndex = i;
            break;
        }
        if (matching[i] == -1) {
            startIndex = i;
            break;
        }
    }

    if (startIndex == -1) {
        std::cout << "No start point found." << std::endl;

    } else {

// 从起始点开始，按照匹配关系遍历所有的点
        std::unordered_set<int> connectedIndicesSet;
        int currentIndex = startIndex;
        while (currentIndex != -1 && connectedIndicesSet.size() < pin_info.size()) {
            connectedIndices.push_back(currentIndex);

            if (matching[currentIndex] == -1 ||
                connectedIndicesSet.find(matching[currentIndex]) != connectedIndicesSet.end()) {
                // The point has already been matched, so we skip it.
                connectedIndicesSet.insert(currentIndex);
                std::cout << "currentIndex:" << currentIndex << "next:" << matching2[currentIndex] << std::endl;
                currentIndex = matching2[currentIndex];
            } else {
                connectedIndicesSet.insert(currentIndex);
                currentIndex = matching[currentIndex];
            }
        }


// 现在，connectedIndices向量中包含了所有串联起来的pin端点的索引
// 比较首尾两个点的坐标值
        cv::Point startPoint(pin_info[connectedIndices.front()][0], pin_info[connectedIndices.front()][1]);
        cv::Point endPoint(pin_info[connectedIndices.back()][0], pin_info[connectedIndices.back()][1]);

// 如果终点的坐标值小于起始点的坐标值，则反转 connectedIndices 向量
        if ((endPoint.x < startPoint.x) || (endPoint.x == startPoint.x && endPoint.y < startPoint.y)) {
            std::reverse(connectedIndices.begin(), connectedIndices.end());
        }

    }
    return connectedIndices;
}


// Function to draw nearest distances between pin points
void drawNearestDistances(cv::Mat &image, const std::vector<std::vector<int>> &pin_info,
                          const std::vector<int> connectedIndices) {

    cv::Scalar color(255, 0, 0);
    int thickness = 1;
    int font = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.3;


    std::vector<cv::Scalar> colors = {cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255)};
    int colorIndex = 0;
    if (connectedIndices.size() > 1)
        for (size_t i = 0; i < connectedIndices.size() - 1; ++i) {
            int ind = connectedIndices[i];
            int ind_next = connectedIndices[i + 1];
            cv::Point p1(pin_info[ind][0], pin_info[ind][1]);
            cv::Point p2(pin_info[ind_next][0], pin_info[ind_next][1]);
            double distance = calculateDistance(p1, p2);

            cv::line(image, p1, p2, colors[colorIndex], thickness);
            std::stringstream stream;
            stream << std::fixed << std::setprecision(1) << distance;
            std::string distanceText = stream.str();
            cv::putText(image, distanceText, (p1 + p2) / 2, font, fontScale, colors[colorIndex], thickness / 2);
        }

    for (size_t i = 0; i < connectedIndices.size(); ++i) {
        int ind = connectedIndices[i];
        cv::Point p1(pin_info[ind][0] + 5, pin_info[ind][1] - 5);
        cv::putText(image, std::to_string(i), p1, font, fontScale, colors[colorIndex + 1], thickness / 2);
        cv::circle(image, cv::Point(pin_info[ind][0], pin_info[ind][1]),
                   (pin_info[ind][2] + pin_info[ind][3]) / 4,
                   cv::Scalar(0, 255, 0), 1);

        printf("圆半径：%f %f，中心坐标：%d %d\n", float(pin_info[ind][0]), float(pin_info[ind][1]), pin_info[ind][2],
               pin_info[ind][3]);
    }

}

void drawMeasureDistances(cv::Mat &image, const std::vector<std::vector<int>> &pin_info,
                          const std::vector<int> connectedIndices,
                          std::vector<AppState::MEASURE_SETTING> &measure_settings) {

    cv::Scalar color(255, 0, 0);
    int thickness = 1;
    int font = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.3;

    std::vector<cv::Scalar> colors = {cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255)};
    int colorIndex = 0;

    for (size_t i = 0; i < measure_settings.size(); ++i) {
        auto measure_setting = measure_settings[i];
        if(measure_setting.pin0>=connectedIndices.size()||measure_setting.pin1>=connectedIndices.size()){
            continue;
        }
        int ind = connectedIndices[measure_setting.pin0];
        int ind_next = connectedIndices[measure_setting.pin1];
        cv::Point p1(pin_info[ind][0], pin_info[ind][1]);
        cv::Point p2(pin_info[ind_next][0], pin_info[ind_next][1]);
        double distance = calculateDistance(p1, p2);
        std::cout << "distance:" << distance << std::endl;
        measure_settings[i].distance = distance;

        if (distance < measure_setting.minDistance || distance > measure_setting.maxDistance) {
            cv::line(image, p1, p2, cv::Scalar(0, 0, 255), thickness); // Red color
            std::stringstream stream;
            stream << std::fixed << std::setprecision(1) << distance;
            std::string distanceText = stream.str();
            cv::putText(image, distanceText, (p1 + p2) / 2, font, fontScale, cv::Scalar(0, 0, 255), thickness / 2); // Red color
        } else {
            cv::line(image, p1, p2, colors[colorIndex], thickness);
            std::stringstream stream;
            stream << std::fixed << std::setprecision(1) << distance;
            std::string distanceText = stream.str();
            cv::putText(image, distanceText, (p1 + p2) / 2, font, fontScale, colors[colorIndex], thickness / 2);
        }
    }

    for (size_t i = 0; i < connectedIndices.size(); ++i) {
        int ind = connectedIndices[i];
        cv::Point p1(pin_info[ind][0] + 5, pin_info[ind][1] - 5);
        cv::putText(image, std::to_string(i), p1, font, fontScale, colors[colorIndex + 1], thickness / 2);
        cv::circle(image, cv::Point(pin_info[ind][0], pin_info[ind][1]),
                   (pin_info[ind][2] + pin_info[ind][3]) / 4,
                   cv::Scalar(0, 255, 0), 1);

        printf("圆半径：%f %f，中心坐标：%d %d\n", float(pin_info[ind][2]), float(pin_info[ind][3]), pin_info[ind][0],
               pin_info[ind][1]);
    }
}

bool AOIParams(AppState &appState) {
//    SobelParams params = appState.sobelParams;
    bool changed = false;
    static bool viewarea = false;

    if (ImGui::Button("刷新摄像头列表")) {
        if (!appState.cameraController.isRunning) {
            appState.validcam = appState.cameraController.GetAvailableCameras().size();
        }
    }

//    for (int i = 0; i < appState.validcam; i++) {
//        if (!ImGui::CollapsingHeader(("摄像头 " + std::to_string(i)).c_str())) {
//            // 显示所有分辨率选项
////            for (const auto& resolution : resolutions)
//            std::vector<std::string> items;
//
//            for (int j = 0; j < resolutions.size(); j++) {
//                auto &resolution = resolutions[j];
//                std::string label = std::to_string(resolution.first) + "x" + std::to_string(resolution.second);
//                items.emplace_back(label);
//            }
////            static int item_current = 0;
//            std::vector<const char *> itemLabels;
//            for (const auto &item: items) {
//                itemLabels.push_back(item.c_str());
//            }
//            if (ImGui::Combo("分辨率", &appState.seli, itemLabels.data(), itemLabels.size())) {
//                auto resolution = resolutions[appState.seli];
//                appState.width = resolution.first;
//                appState.height = resolution.second;
//                // 选择了分辨率，执行相应操作
//                std::cout << "Selected resolution: " << resolution.first << "x" << resolution.second << std::endl;
//
//            }
////            ImGui::Combo("分辨率", &item_current, items.data()->c_str(), items.data()->size());
//
//            if (ImGui::Button("打开关闭")) {
//                appState.cameraController.SetResolution(appState.width, appState.height);
//                appState.cameraController.ToggleCamera(i);
//                appState.image_points_seq.clear();
//                appState.object_points.clear();
//                appState.paramsSummary.clear();
//            }
//        }
//        ImGui::Spacing();
//    }

    if (!ImGui::CollapsingHeader("采集图像")) {
        if (ImGui::Button("抓取图像")) {
            appState.imageCap = appState.image;
        }
        ImGui::SameLine();
        if (ImGui::Button("实时图像")) {
            appState.imageCap = cv::Mat();
        }
        if (ImGui::Button("打开文件")) {
            std::string title = "打开图片";
            std::string default_path;
            std::vector<std::string> filters = {"Image Files", "*.png *.jpg *.jpeg *.bmp"};
            pfd::open_file dialog(title, default_path, filters,
                                  pfd::opt::none);
            if (dialog.result().size() > 0) {
                std::cout << "open:" << dialog.result()[0] << std::endl;
                appState.imageCap = cv::imread(dialog.result()[0]);
                appState.imageProcess = cv::Mat();
            }
        }

    }

    if (!ImGui::CollapsingHeader("图像处理")) {

        if (ImGui::Button("检测红圆")) {
            if (!appState.imageCap.empty()) {
                appState.imageCap.copyTo(appState.imageProcess);
                // 转换到HSV色彩空间
                cv::Mat hsv;
                cv::cvtColor(appState.imageCap, hsv, cv::COLOR_BGR2HSV);

                // 定义红色的HSV颜色范围
                cv::Mat mask1, mask2;
                cv::inRange(hsv, cv::Scalar(0, 70, 50), cv::Scalar(10, 255, 255), mask1);
                cv::inRange(hsv, cv::Scalar(170, 70, 50), cv::Scalar(180, 255, 255), mask2);

                // 合并两个颜色范围的掩码
                cv::Mat mask = mask1 | mask2;
//                appState.imageProcess = mask;
                // 对图像进行开操作，去除噪点
//                cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));
//                cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

                // 寻找圆形
                std::vector<std::vector<cv::Point> > contours;
                cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
                appState.circle_info.clear();
                std::vector<cv::Point> approx;
                for (int i = 0; i < contours.size(); i++) {
                    // 对每个找到的轮廓进行多边形逼近
                    cv::approxPolyDP(cv::Mat(contours[i]), approx, cv::arcLength(cv::Mat(contours[i]), true) * 0.02,
                                     true);

                    // 如果逼近的点的数量大于6，我们假定找到了一个圆
                    if (approx.size() > 6) {
                        // 计算轮廓的面积
                        double area = cv::contourArea(contours[i]);

                        // 计算对象的边界矩形
                        cv::Rect r = cv::boundingRect(contours[i]);

                        // 计算矩形的面积
                        int rect_area = r.height * r.width;

                        // 检查面积比以确定是否是圆
                        double ratio = double(area) / rect_area;
                        if (ratio > 0.5) {
                            cv::circle(appState.imageProcess, cv::Point(r.x + r.width / 2, r.y + r.height / 2),
                                       (r.width + r.height) / 4,
                                       cv::Scalar(0, 255, 0), 2);
                            cv::circle(appState.imageProcess, cv::Point(r.x + r.width / 2, r.y + r.height / 2), 5,
                                       cv::Scalar(255, 0, 0), 5);
                            printf("圆半径：%f %f，中心坐标：%d %d\n", r.width / 2., r.height / 2., r.x + r.width / 2,
                                   r.y + r.height / 2);
                            appState.circle_info.emplace_back(
                                    std::vector<int>({r.x + r.width / 2, r.y + r.height / 2, r.width, r.height}));
                        }
                    }
                }
                int largerCircleIndex = getLargerCircleIfRadiusDouble(appState);

                if (largerCircleIndex == -1) {
                    std::cerr << "The radii of the first two circles are not approximately double" << std::endl;
                    return -1;
                }
                appState.large_circle_id = largerCircleIndex;
                // 获取并绘制大圆（位于新坐标系的原点）
                std::vector<int> largerCircle = appState.circle_info[largerCircleIndex];
                std::vector<int> circle = appState.circle_info[1 - largerCircleIndex];

                // 在图像中心绘制坐标系
                cv::line(appState.imageProcess, cv::Point(largerCircle[0], largerCircle[1]),
                         cv::Point(circle[0], circle[1]), cv::Scalar(255, 255, 255), 1);
                cv::Point v = cv::Point(circle[0] - largerCircle[0], circle[1] - largerCircle[1]);
                cv::Point v_perpendicular = cv::Point(-v.y, v.x);

// 选择一个长度scale，例如，你可以设置为图像的宽度或者高度
                double scale = appState.imageProcess.size().width;
// 计算垂线上的一个点
                cv::Point point_on_perpendicular_line = cv::Point(largerCircle[0] + scale * v_perpendicular.x,
                                                                  largerCircle[1] + scale * v_perpendicular.y);
// 画出垂线
                cv::line(appState.imageProcess, cv::Point(largerCircle[0], largerCircle[1]),
                         point_on_perpendicular_line, cv::Scalar(255, 255, 255), 1);

            }
        }
        ImGui::SameLine();
        if (ImGui::Button("加载参数")) {
//            appState.imageProcess = cv::Mat();
            cv::FileStorage fs(appState.savePath.data(), cv::FileStorage::READ);
            if (fs.isOpened()) {
                fs["rect_x"] >> appState.rect_x;
                fs["rect_y"] >> appState.rect_y;
                fs["rect_w"] >> appState.rect_w;
                fs["rect_h"] >> appState.rect_h;
                fs["bin_threshold"] >> appState.bin_threshold;

//                fs["circle_info"] >> appState.circle_info;
//                fs["large_circle_id"] >> appState.large_circle_id;

                fs.release();
                viewarea = true;
            }
        }


        // Keep track of the previous values
        static float prev_rect_x = appState.rect_x;
        static float prev_rect_y = appState.rect_y;
        static float prev_rect_w = appState.rect_w;
        static float prev_rect_h = appState.rect_h;
        static float prev_bin_threshold = appState.bin_threshold;

        ImGui::SliderFloat("X", &appState.rect_x, -3.0f, 3.0f);
        ImGui::SliderFloat("Y", &appState.rect_y, -3.0f, 3.0f);
        ImGui::SliderFloat("Width", &appState.rect_w, 0.0f, 1.0f);
        ImGui::SliderFloat("Height", &appState.rect_h, 0.0f, 1.0f);
        ImGui::SliderFloat("threshold", &appState.bin_threshold, -0.0f, 255.0f);
        if (prev_rect_x != appState.rect_x || prev_rect_y != appState.rect_y ||
            prev_rect_w != appState.rect_w || prev_rect_h != appState.rect_h ||
            prev_bin_threshold != appState.bin_threshold) {
            viewarea = true;
        }

        if (viewarea) {
            appState.immvisionParamsStatic.RefreshImage = true;
            appState.immvisionParamsStaticSub.RefreshImage = true;
        }
// Check if any slider value has changed
        if (appState.circle_info.size() > 0) {
            if (viewarea) {
                viewarea = false;
                appState.imageCap.copyTo(appState.imageProcess);
                // Get the image width and height
//                int imageWidth = appState.imageCap.cols;
//                int imageHeight = appState.imageCap.rows;

                // 获取并绘制大圆（位于新坐标系的原点）
                std::vector<int> largerCircle = appState.circle_info[appState.large_circle_id];
                std::vector<int> circle = appState.circle_info[1 - appState.large_circle_id];

                // 在图像中心绘制坐标系
                cv::line(appState.imageProcess, cv::Point(largerCircle[0], largerCircle[1]),
                         cv::Point(circle[0], circle[1]),
                         cv::Scalar(255, 255, 255), 1);
                cv::Point v = cv::Point(circle[0] - largerCircle[0], circle[1] - largerCircle[1]);
                cv::Point v_perpendicular = cv::Point(-v.y, v.x);

                // 选择一个长度scale，例如，你可以设置为图像的宽度或者高度
                double scale = appState.imageProcess.size().width;

                // 计算垂线上的一个点
                cv::Point point_on_perpendicular_line = cv::Point(largerCircle[0] + scale * v_perpendicular.x,
                                                                  largerCircle[1] + scale * v_perpendicular.y);

                // 画出垂线
                cv::line(appState.imageProcess, cv::Point(largerCircle[0], largerCircle[1]),
                         point_on_perpendicular_line,
                         cv::Scalar(255, 255, 255), 1);

                float vnorm = cv::norm(v);
                // Calculate the relative rectangle parameters
                appState.relativeRectX = appState.rect_x * vnorm;
                appState.relativeRectY = appState.rect_y * vnorm;
                appState.relativeRectW = appState.rect_w * vnorm;
                appState.relativeRectH = appState.rect_h * vnorm;

                // Define the rectangle's position and size relative to the largerCircle
                cv::Rect rect(appState.relativeRectX + largerCircle[0], appState.relativeRectY + largerCircle[1],
                              appState.relativeRectW, appState.relativeRectH);

                // Calculate the rotation angle based on the coordinate system
//                double rotationAngle = std::acos(v.x / cv::norm(v)) * 180 / CV_PI;
                double rotationAngle = -std::atan2(v.y, v.x) * 180 / CV_PI;
                std::cout << "rotationAngle:" << rotationAngle << std::endl;

                // Perform affine transformation to rotate the rectangle
                cv::Mat rotationMatrix = cv::getRotationMatrix2D(cv::Point(largerCircle[0], largerCircle[1]),
                                                                 rotationAngle,
                                                                 1.0);

                // Warp the rectangle using the rotation matrix
                std::vector<cv::Point2f> srcPoints = {cv::Point2f(rect.x, rect.y),
                                                      cv::Point2f(rect.x + rect.width, rect.y),
                                                      cv::Point2f(rect.x + rect.width, rect.y + rect.height),
                                                      cv::Point2f(rect.x, rect.y + rect.height)};
                std::vector<cv::Point2f> dstPoints(4);
                cv::transform(srcPoints, dstPoints, rotationMatrix);

                // Draw the rotated rectangle
                std::vector<cv::Point> rotatedRectPoints;
                for (const auto &point: dstPoints) {
                    rotatedRectPoints.push_back(cv::Point(static_cast<int>(point.x), static_cast<int>(point.y)));
                }

                cv::polylines(appState.imageProcess, rotatedRectPoints, true, cv::Scalar(0, 255, 0), 2);

                // 裁剪图像
                cv::Rect boundingRect = cv::boundingRect(rotatedRectPoints);
                cv::Mat croppedImage = appState.imageCap(boundingRect);

// 旋转图像

                cv::Mat rotatedImage;
                cv::Mat rotationMatrix2 = cv::getRotationMatrix2D(
                        cv::Point(croppedImage.size().width / 2, croppedImage.size().height / 2), -rotationAngle,
                        1.0);
                cv::warpAffine(croppedImage, rotatedImage, rotationMatrix2, croppedImage.size(), cv::INTER_LINEAR);
                std::vector<cv::Point2f> dstPoints2(4);
//                cv::transform(dstPoints, dstPoints2, rotationMatrix2);

                std::vector<cv::Point> rotatedRectPoints2;
                for (const auto &point: dstPoints) {
                    cv::Point rotatedPoint;
                    rotatedPoint.x = static_cast<int>(
                            (point.x - boundingRect.x - boundingRect.width / 2) * cos(-rotationAngle * CV_PI / 180.0) -
                            (point.y - boundingRect.y - boundingRect.height / 2) * sin(-rotationAngle * CV_PI / 180.0) +
                            croppedImage.size().width / 2);
                    rotatedPoint.y = static_cast<int>(
                            (point.x - boundingRect.x - boundingRect.width / 2) * sin(-rotationAngle * CV_PI / 180.0) +
                            (point.y - boundingRect.y - boundingRect.height / 2) * cos(-rotationAngle * CV_PI / 180.0) +
                            croppedImage.size().height / 2);
                    rotatedRectPoints2.push_back(rotatedPoint);
                }

                // 裁剪图像
                cv::Rect boundingRect2 = cv::boundingRect(rotatedRectPoints2);

// 检查边界矩形是否超出图像范围
                cv::Rect imageRect(0, 0, rotatedImage.cols, rotatedImage.rows);
                boundingRect2 = boundingRect2 & imageRect;


// 创建一个与边界矩形大小一致的新图像
                cv::Mat imageProcess2;//(boundingRect2.size(), rotatedImage.type());

                imageProcess2 = rotatedImage(boundingRect2);

                appState.imageProcess2 = imageProcess2;


// 将结果赋值给 appState.imageProcess2
//                rotatedImage.copyTo(appState.imageProcess2);

// 进行二值化处理

                cv::Mat binaryImage;
                cv::Mat grayImage;
                cv::cvtColor(imageProcess2, grayImage, cv::COLOR_BGR2GRAY);
                cv::threshold(grayImage, binaryImage, appState.bin_threshold, 255, cv::THRESH_BINARY);
                // 检查二值化图像的通道数
                if (binaryImage.channels() == 1) {
                    // 如果二值化图像是灰度图像，将其转换为RGB图像
                    cv::cvtColor(binaryImage, appState.imageProcess2, cv::COLOR_GRAY2RGB);
                } else {
                    // 如果二值化图像已经是RGB图像，直接使用
                    appState.imageProcess2 = binaryImage;
                }

                // 寻找圆形
                std::vector<std::vector<cv::Point> > contours;
                cv::findContours(binaryImage.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
                appState.pin_info.clear();
                std::vector<cv::Point> approx;
                for (int i = 0; i < contours.size(); i++) {
                    // 对每个找到的轮廓进行多边形逼近
                    cv::approxPolyDP(cv::Mat(contours[i]), approx, cv::arcLength(cv::Mat(contours[i]), true) * 0.02,
                                     true);

                    // 如果逼近的点的数量大于6，我们假定找到了一个圆
                    if (approx.size() > 6) {
                        // 计算轮廓的面积
                        double area = cv::contourArea(contours[i]);

                        // 计算对象的边界矩形
                        cv::Rect r = cv::boundingRect(contours[i]);

                        // 计算矩形的面积
                        int rect_area = r.height * r.width;

                        // 检查面积比以确定是否是圆
                        double ratio = double(area) / rect_area;
                        if (ratio > 0.4) {

                            appState.pin_info.emplace_back(
                                    std::vector<int>({r.x + r.width / 2, r.y + r.height / 2, r.width, r.height}));


                        }
                    }
                }
                auto connectedIndices = findMatchingRelationships(appState.pin_info);
                appState.connectedIndices = connectedIndices;
                appState.needRefresh = true;
            }
        }


// Update the previous values
        prev_rect_x = appState.rect_x;
        prev_rect_y = appState.rect_y;
        prev_rect_w = appState.rect_w;
        prev_rect_h = appState.rect_h;
        prev_bin_threshold = appState.bin_threshold;

        if (ImGui::Button("保存参数")) {
//            appState.imageProcess = cv::Mat();
            cv::FileStorage fs(appState.savePath.data(), cv::FileStorage::WRITE);
            if (fs.isOpened()) {
                fs << "rect_x" << appState.rect_x;
                fs << "rect_y" << appState.rect_y;
                fs << "rect_w" << appState.rect_w;
                fs << "rect_h" << appState.rect_h;
                fs << "bin_threshold" << appState.bin_threshold;
//
//                fs << "circle_info" << appState.circle_info;
//                fs << "large_circle_id" << appState.large_circle_id;

                fs.release();
            }
            calibstatus = 3;
        }


        if (ImGui::Button("原始图像")) {
            appState.imageProcess = cv::Mat();
        }
    }
    return changed;
}


void RightTable(AppState &appState) {
    size_t pin_num = appState.pin_info.size();
    if (ImGui::Button("添加测量"))
        ImGui::OpenPopup("添加测量");
    if (ImGui::BeginPopupModal("添加测量", NULL, ImGuiWindowFlags_MenuBar)) {

        ImGui::Text("选择两个pin针编号，设置容许范围");

        // Testing behavior of widgets stacking their own regular popups over the modal.
        static int pin0 = 0;
        static int pin1 = 1;
        static float minDistance = 30;
        static float maxDistance = 60;


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
        if(pin0<0){
            pin0 = 0;
        }
        if(pin1<0){
            pin1 = 0;
        }


        ImGui::Separator();
        if (ImGui::Button("关闭"))
            ImGui::CloseCurrentPopup();
        ImGui::SameLine();
        if (ImGui::Button("添加")) {
            appState.pin_measure_settings.emplace_back(
                    AppState::MEASURE_SETTING{pin0, pin1, minDistance, maxDistance});
            ImGui::CloseCurrentPopup();
            appState.needRefresh = true;
        }
        ImGui::EndPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("加载设置")) {
        appState.loadPairSettings();
        appState.needRefresh = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("保存设置")) {
        appState.savePairSettings();
    }

    ImGui::BeginTable("WatchedPixels", 6);
    ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
    ImGui::TableNextColumn();
    ImGui::Text("IND");
    ImGui::TableNextColumn();
    ImGui::Text("pins");
    ImGui::TableNextColumn();
    ImGui::Text("min");
    ImGui::TableNextColumn();
    ImGui::Text("max");
    ImGui::TableNextColumn();
    ImGui::Text("dist");
    ImGui::TableNextColumn();
    ImGui::Text("Edit");


    static size_t indices_to_remove = -1;
    static size_t indices_to_edit = -1;
    for (size_t i = 0; i < appState.pin_measure_settings.size(); ++i) {
        auto pin_measure_setting = appState.pin_measure_settings[i];
        ImGui::TableNextRow();
        if(pin_measure_setting.distance<0){
            ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0,
                                   ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 1.0f)));
        }else if(pin_measure_setting.distance<pin_measure_setting.minDistance || pin_measure_setting.distance>pin_measure_setting.maxDistance) {
            ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0,
                                   ImGui::GetColorU32(ImVec4(1.0f, 0.0f, 0.0f, 1.0f))); // 设置一行的背景颜色为红色
        }

        // index
        ImGui::TableNextColumn();
        ImGui::Text("#%i: ", (int) i);

        // (x,y)
        ImGui::TableNextColumn();
        std::string posStr =
                std::to_string(pin_measure_setting.pin0) + "<->" + std::to_string(pin_measure_setting.pin1);
        ImGui::Text("%s", posStr.c_str());

        ImGui::TableNextColumn();
        ImGui::Text("%0.1f", pin_measure_setting.minDistance);
        ImGui::TableNextColumn();
        ImGui::Text("%0.1f", pin_measure_setting.maxDistance);

        // Show Color Cell
        ImGui::TableNextColumn();
        ImGui::Text("%0.1f", pin_measure_setting.distance);

        // Actions
        ImGui::TableNextColumn();
        std::string lblRemove = "x##" + std::to_string(i);
        if (ImGui::SmallButton(lblRemove.c_str())) {
            indices_to_remove = i;
        }
        ImGui::SameLine();
        std::string lblEdit = "e##" + std::to_string(i);
        if (ImGui::SmallButton(lblEdit.c_str())) {
            indices_to_edit = i;
        }
        ImGui::SameLine();
    }
    ImGui::EndTable();
    // Remove elements in reverse order
    if (indices_to_remove != -1) {
        appState.pin_measure_settings.erase(appState.pin_measure_settings.begin() + indices_to_remove);
        indices_to_remove = -1;
        appState.needRefresh = true;
    }
    if (indices_to_edit != -1) {
        ImGui::OpenPopup("编辑测量");
        if (ImGui::BeginPopupModal("编辑测量")) {
            ImGui::Text("选择两个pin针编号，设置容许范围");
            auto setting = (appState.pin_measure_settings.begin() + indices_to_edit);
            // Testing behavior of widgets stacking their own regular popups over the modal.
            static int pin0 = setting->pin0;
            static int pin1 = setting->pin1;
            static float minDistance = setting->minDistance;
            static float maxDistance = setting->maxDistance;

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
            if(pin0<0){
                pin0 = 0;
            }
            if(pin1<0){
                pin1 = 0;
            }

            ImGui::Separator();
            if (ImGui::Button("关闭")) {
                ImGui::CloseCurrentPopup();
                indices_to_edit = -1;
            }
            ImGui::SameLine();
            if (ImGui::Button("更新")) {
                setting->pin0 = pin0;
                setting->pin1 = pin1;
                setting->minDistance = minDistance;
                setting->maxDistance = maxDistance;
                ImGui::CloseCurrentPopup();
                indices_to_edit = -1;
                appState.needRefresh = true;
            }
            ImGui::EndPopup();
        }
    }

    if (appState.needRefresh) {
        appState.imageProcess2.copyTo(appState.imageProcess3);
//        drawNearestDistances(appState.imageProcess3, appState.pin_info, appState.connectedIndices);
        drawMeasureDistances(appState.imageProcess3, appState.pin_info, appState.connectedIndices,
                             appState.pin_measure_settings);
        appState.immvisionParamsStaticSub.RefreshImage = true;
//
    }
}




class Checker {
private:
    std::vector<std::pair<std::vector<double>, cv::Mat>> db;
    double max_chessboard_speed;
    std::vector<std::string> param_names;
    std::vector<double> param_ranges;
    bool goodenough;
    std::thread checkerThread;

public:

    std::atomic<bool> isRunning;

    Checker(double max_speed) : max_chessboard_speed(max_speed) {
        param_names = {"X", "Y", "Size", "Skew"};
        param_ranges = {1.0, 1.0, 1.0, 1.0};
        goodenough = false;
    }


// Define a struct for storing corner coordinates
    typedef cv::Point2f Corner;

// Define a struct for storing the corners of the board
    struct BoardCorners {
        Corner up_left;
        Corner up_right;
        Corner down_right;
        Corner down_left;
    };

// Define a struct for storing the parameters
    struct Parameters {
        float x;
        float y;
        float size;
        float skew;
    };

// Function to calculate the angle between three points
    float angle(const Corner &a, const Corner &b, const Corner &c) {
        float ab_x = a.x - b.x;
        float ab_y = a.y - b.y;
        float cb_x = c.x - b.x;
        float cb_y = c.y - b.y;
        float dot_product = ab_x * cb_x + ab_y * cb_y;
        float norm_ab = std::sqrt(ab_x * ab_x + ab_y * ab_y);
        float norm_cb = std::sqrt(cb_x * cb_x + cb_y * cb_y);
        return std::acos(dot_product / (norm_ab * norm_cb));
    }

// Function to calculate the skew for given checkerboard detection
    float calculateSkew(const BoardCorners &corners) {
        const Corner &up_left = corners.up_left;
        const Corner &up_right = corners.up_right;
        const Corner &down_right = corners.down_right;
        float skew = std::min(1.0f, float(2.0f * std::abs((M_PI / 2.0f) - angle(up_left, up_right, down_right))));
        return skew;
    }

// Function to calculate the area of the detected checkerboard
    float calculateArea(const BoardCorners &corners) {
        const Corner &up_left = corners.up_left;
        const Corner &up_right = corners.up_right;
        const Corner &down_right = corners.down_right;
        const Corner &down_left = corners.down_left;
        float a_x = up_right.x - up_left.x;
        float a_y = up_right.y - up_left.y;
        float b_x = down_right.x - up_right.x;
        float b_y = down_right.y - up_right.y;
        float c_x = down_left.x - down_right.x;
        float c_y = down_left.y - down_right.y;
        float p_x = b_x + c_x;
        float p_y = b_y + c_y;
        float q_x = a_x + b_x;
        float q_y = a_y + b_y;
        return std::abs(p_x * q_y - p_y * q_x) / 2.0f;
    }

// Function to get the outside corners of the board
    BoardCorners getOutsideCorners(const std::vector<Corner> &corners, AppState &appState) {
        int xdim = appState.boardCols;
        int ydim = appState.boardRows;

        if (appState.calibPatten != CHARUCO && corners.size() != xdim * ydim) {
            throw std::runtime_error("Invalid number of corners!");
        }

        if (appState.calibPatten == CHARUCO && corners.size() != (xdim - 1) * (ydim - 1)) {
            throw std::runtime_error("Invalid number of corners for ChArUco boards!");
        }

        BoardCorners outside_corners;
        outside_corners.up_left = corners[0];
        outside_corners.up_right = corners[xdim - 1];
        outside_corners.down_right = corners[corners.size() - 1];
        outside_corners.down_left = corners[corners.size() - xdim];
        return outside_corners;
    }

// Function to get the largest rectangle corners in a partial view of a ChArUco board
    BoardCorners
    getLargestRectangleCorners(const std::vector<Corner> &corners, const std::vector<int> &ids, AppState &appState) {
        int xdim = appState.boardCols - 1;
        int ydim = appState.boardRows - 1;
        std::vector<std::vector<bool>> board_vis(ydim, std::vector<bool>(xdim, false));

        // Initialize board visibility
        for (int i = 0; i < ids.size(); i++) {
            int id = ids[i];
            int row = id / xdim;
            int col = id % xdim;
            board_vis[row][col] = true;
        }

        int best_area = 0;
        int best_rect[4] = {-1, -1, -1, -1};

        for (int x1 = 0; x1 < xdim; x1++) {
            for (int x2 = x1; x2 < xdim; x2++) {
                for (int y1 = 0; y1 < ydim; y1++) {
                    for (int y2 = y1; y2 < ydim; y2++) {
                        if (board_vis[y1][x1] && board_vis[y1][x2] && board_vis[y2][x1] && board_vis[y2][x2] &&
                            (x2 - x1 + 1) * (y2 - y1 + 1) > best_area) {
                            best_area = (x2 - x1 + 1) * (y2 - y1 + 1);
                            best_rect[0] = x1;
                            best_rect[1] = x2;
                            best_rect[2] = y1;
                            best_rect[3] = y2;

                        }
                    }
                }
            }
        }

        int x1 = best_rect[0];
        int x2 = best_rect[1];
        int y1 = best_rect[2];
        int y2 = best_rect[3];
        std::vector<int> corner_ids = {y2 * xdim + x1, y2 * xdim + x2, y1 * xdim + x2, y1 * xdim + x1};
        BoardCorners largest_rectangle_corners;
        largest_rectangle_corners.up_left = corners[corner_ids[0]];
        largest_rectangle_corners.up_right = corners[corner_ids[1]];
        largest_rectangle_corners.down_right = corners[corner_ids[2]];
        largest_rectangle_corners.down_left = corners[corner_ids[3]];
        return largest_rectangle_corners;
    }

// Function to get the parameters describing the checkerboard view
    const std::vector<double>
    getParameters(const std::vector<Corner> &corners, const std::vector<int> &ids, AppState &appState,
                  const std::pair<int, int> &size) {
        int width = size.first;
        int height = size.second;
        std::vector<float> Xs;
        std::vector<float> Ys;

        for (const auto &corner: corners) {
            Xs.push_back(corner.x);
            Ys.push_back(corner.y);
        }

        BoardCorners outside_corners;
        if (appState.calibPatten == CHARUCO) {
            outside_corners = getLargestRectangleCorners(corners, ids, appState);
        } else {
            outside_corners = getOutsideCorners(corners, appState);
        }

        float area = calculateArea(outside_corners);
        float skew = calculateSkew(outside_corners);
        float border = std::sqrt(area);
        float p_x = std::min(1.0f, std::max(0.0f,
                                            (std::accumulate(Xs.begin(), Xs.end(), 0.0f) / Xs.size() - border / 2.0f) /
                                            (width - border)));
        float p_y = std::min(1.0f, std::max(0.0f,
                                            (std::accumulate(Ys.begin(), Ys.end(), 0.0f) / Ys.size() - border / 2.0f) /
                                            (height - border)));
        float p_size = std::sqrt(area / (width * height));

        std::vector<double> params;
        params.emplace_back(p_x);
        params.emplace_back(p_y);
        params.emplace_back(p_size);
        params.emplace_back(skew);

        return params;
    }

    bool is_slow_moving(const std::vector<Corner> &corners, const std::vector<int> &ids,
                        const std::vector<Corner> &last_frame_corners,
                        const std::vector<int> &last_frame_ids) {
        if (last_frame_corners.empty())
            return false;

        std::vector<std::vector<double>> corner_deltas;
        if (ids.empty()) {
            int num_corners = corners.size();
            corner_deltas.resize(num_corners, std::vector<double>(2));
            for (int i = 0; i < num_corners; i++) {
                corner_deltas[i][0] = corners[i].x - last_frame_corners[i].x;
                corner_deltas[i][1] = corners[i].y - last_frame_corners[i].y;
            }
        } else {
            std::vector<int> last_frame_ids_copy = last_frame_ids;
            corner_deltas.reserve(ids.size());
            for (int i = 0; i < ids.size(); i++) {
                try {
                    auto it = std::find(last_frame_ids_copy.begin(), last_frame_ids_copy.end(), ids[i]);
                    if (it != last_frame_ids_copy.end()) {
                        int last_i = std::distance(last_frame_ids_copy.begin(), it);
                        corner_deltas.push_back({corners[i].x - last_frame_corners[last_i].x,
                                                 corners[i].y - last_frame_corners[last_i].y});
                        last_frame_ids_copy.erase(it);
                    }
                } catch (const std::exception &) {
                    // Ignore exception and continue to the next iteration
                }
            }
        }

        double average_motion = 0.0;
        for (const auto &delta: corner_deltas) {
            average_motion += std::sqrt(delta[0] * delta[0] + delta[1] * delta[1]);
        }
        average_motion /= corner_deltas.size();

        return average_motion <= max_chessboard_speed;
    }

    bool is_good_sample(const std::vector<double> &params, const std::vector<Corner> &corners,
                        const std::vector<int> &ids, const std::vector<Corner> &last_frame_corners,
                        const std::vector<int> &last_frame_ids) {
        if (db.empty())
            return true;

        auto param_distance = [](const std::vector<double> &p1, const std::vector<double> &p2) {
            double distance = 0.0;
            for (int i = 0; i < p1.size(); i++) {
                distance += std::abs(p1[i] - p2[i]);
            }
            return distance;
        };

        std::vector<std::vector<double>> db_params;
        db_params.reserve(db.size());
        for (const auto &sample: db) {
            db_params.push_back(sample.first);
        }

        double d = std::numeric_limits<double>::max();
        for (const auto &p: db_params) {
            double distance = param_distance(params, p);
            if (distance < d)
                d = distance;
        }
        std::cout << "d:" << d << std::endl;
        if (d <= 0.2)
            return false;

        if (max_chessboard_speed > 0 && !is_slow_moving(corners, ids, last_frame_corners, last_frame_ids))
            return false;

        return true;
    }

    std::vector<std::tuple<std::string, double, double, double>> compute_goodenough() {
        if (db.empty())
            return {};

        std::vector<std::vector<double>> all_params;
        all_params.reserve(db.size());
        for (const auto &sample: db) {
            all_params.push_back(sample.first);
        }

        std::vector<double> min_params = all_params[0];
        std::vector<double> max_params = all_params[0];
        for (int i = 1; i < all_params.size(); i++) {
            for (int j = 0; j < min_params.size(); j++) {
                min_params[j] = std::min(min_params[j], all_params[i][j]);
                max_params[j] = std::max(max_params[j], all_params[i][j]);
            }
        }
        min_params[2] = 0.0; // Don't reward small size
        min_params[3] = 0.0; // Don't reward skew

        std::vector<double> progress;
        progress.reserve(min_params.size());
        for (int i = 0; i < min_params.size(); i++) {
            double p = std::min((max_params[i] - min_params[i]) / param_ranges[i], 1.0);
            progress.push_back(p);
        }

        goodenough =
                (db.size() >= 40) || std::all_of(progress.begin(), progress.end(), [](double p) { return p == 1.0; });

        std::vector<std::tuple<std::string, double, double, double>> result;
        for (int i = 0; i < min_params.size(); i++) {
            result.push_back(std::make_tuple(param_names[i], min_params[i], max_params[i], progress[i]));
        }

        return result;
    }


    void Start(AppState &appState) {
        if (!isRunning) {
            isRunning = true;
            checkerThread = std::thread(&Checker::CheckerThreadFunc, this, std::ref(appState));
        }
    }

    void Stop() {
        if (isRunning) {
            isRunning = false;
            checkerThread.join();
            db.clear();
        }
    }

    std::vector<int> generateIds(AppState &appState) {
        int xdim = appState.boardCols;
        int ydim = appState.boardRows;

        int total_corners = xdim * ydim;
        std::vector<int> ids(total_corners);

        // Generate sequential IDs starting from 0
        for (int i = 0; i < total_corners; i++) {
            ids[i] = i;
        }

        return ids;
    }


    void CheckerThreadFunc(AppState &appState) {
        const float confThreshold = 0.3f;
        const float iouThreshold = 0.4f;
        const float maskThreshold = 0.05f;
        bool div255 = true;


        float anchors_640[3][6] = {{10.0,   10.0,      20.0,    20.0,     40.0,  40.0},
                                   {412.75, 17.921875, 196.875, 60.84375, 189.5, 74.0625},
                                   {301.0,  104.25,    660.5,   52.3125,  358.5, 153.75}};
        const std::vector<std::string> classNames{"l", "p", "o"};


        bool isGPU = false;
        const std::string modelPath = "/home/nn/Documents/immvision/deep_perception_engine/best.onnx";

//        Perception::YOLOv5 *detector(nullptr);
//        const std::vector<int64_t> in_size{800, 800};
//        const std::vector<std::string> output_names{"l80", "l40", "l20"};
//        try {
//            detector = new Perception::YOLOv5(modelPath, isGPU, in_size, output_names);
//            std::cout << "Model was initialized." << std::endl;
//        }
//        catch (const std::exception &e) {
//            std::cerr << e.what() << std::endl;
//            return;
//        }


//        while (isRunning) {
//            cv::Mat image;
//            cv::Size image_size;
//            std::vector<std::shared_ptr<std::vector<float>>> infer_result;
//            cv::Mat segres;
//            try {
//                image = appState.image;
//                image_size = image.size();
//
//                std::vector<float> blob = detector->imagePreprocessing(image, in_size, div255);
//                std::vector<std::vector<float>> blobs;
//                blobs.emplace_back(blob);
//                uint64_t startPostTime = Perception::utils::getMonotonicTimeMs();
//                infer_result = detector->execute(blobs);
//
//                std::cerr << "infer Process elapsed: " << Perception::utils::getMonotonicTimeMs() - startPostTime
//                          << "(ms)"
//                          << std::endl;
//                std::cout << "execute ok !!" << std::endl;
//            } catch (const std::exception &e) {
//                std::cerr << e.what() << std::endl;
//                return;
//            }
//
////    std::vector<std::vector<int64_t>> in_shapes = detector->getInputShape();
//            std::vector<std::vector<int64_t>> out_shapes = detector->getOutputShape();
//            std::vector<std::string> out_names = detector->getOutputNames();
//            uint64_t startPostTime = Perception::utils::getMonotonicTimeMs();
//            std::vector<Perception::utils::Detection> result = detector->postProcessing(infer_result, out_shapes,
//                                                                                        out_names, in_size,
//                                                                                        {image_size.width,
//                                                                                         image_size.height},
//                                                                                        (float *) anchors_640,
//                                                                                        confThreshold, iouThreshold);
//            detector->visualizeDetection(image, result, classNames);
//            std::cerr << "post Process elapsed: " << Perception::utils::getMonotonicTimeMs() - startPostTime << "(ms)"
//                      << std::endl;
////    cv::imshow("result", image);
////    cv::waitKey(0);
////        cv::imwrite("result.jpg", image);
//            appState.imageDraw = image;
//            usleep(10);
//        }
//        delete detector;
    }
};


ImVec2 set_ImageSize(float listWidth, bool showOptionsColumn) {
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


Checker checker(-0.5);

#include "immvision/internal/imgui/image_widgets.h"

void ImageView(AppState &appState) {
//    static AppState appState;
    ImmVision::ImageWidgets::s_CollapsingHeader_CacheState_Sync = true;

    static float initialListWidth = ImGui::GetFontSize() * 8.5f;
    static float currentListWidth = initialListWidth;

    bool showOptionsColumn = true;
    if ((appState.immvisionParams.ShowOptionsInTooltip) || (!appState.immvisionParams.ShowOptionsPanel))
        showOptionsColumn = false;
    ImVec2 imageSize = set_ImageSize(currentListWidth, showOptionsColumn);

//    std::cout<<showOptionsColumn<<" "<<imageSize.x<<" "<<imageSize.y<<std::endl;

    {
//        ImGui::Separator();
        if (appState.cameraController.isRunning) {
            cv::Mat tim = appState.cameraController.GetFrame();
            if (!tim.empty()) {
                appState.image = tim;
            }


//            std::cout<<"num of points :"<<appState.image_points_seq.size()<<std::endl;
            if (!checker.isRunning) {
                if (!tim.empty())
                    checker.Start(appState);
            }
            if (!appState.cameraController.isRunning) {
                checker.Stop();
            }
        }

        {
//            appState.immvisionParams.ImageDisplaySize = cv::Size((int) imageSize.x - 540, 0);
            if (checker.isRunning) {
                ImmVision::Image("Checker", appState.imageDraw, &appState.immvisionParams);
                usleep(1e3);
//                appState.imageDraw = cv::Mat();
            } else if (appState.cameraController.isRunning) {
                ImmVision::Image("Original", appState.image, &appState.immvisionParams);
//        ImmVision::Image("Deriv", appState.imageSobel, &appState.immvisionParamsSobel);
            } else if (!appState.imageProcess.empty()) {
                ImmVision::Image("Process", appState.imageProcess, &appState.immvisionParamsStatic);
            } else {
                ImmVision::Image("File", appState.imageCap, &appState.immvisionParamsStatic);

            }

            ImGui::SameLine();
//            ImGui::NextColumn(); // 切换到下一列
// 在窗口中添加横向分隔

            appState.immvisionParamsStaticSub.ImageDisplaySize.height = (int) imageSize.y / 2 - 60;
            {
                ImGui::BeginGroup();
//                ImmVision::Image("Process2", appState.imageProcess2, &appState.immvisionParamsSub);
                ImmVision::Image("Process3", appState.imageProcess3, &appState.immvisionParamsStaticSub);
                ImGui::EndGroup();
            }
            if (appState.needRefresh) {
                appState.needRefresh = false;
                appState.immvisionParamsStaticSub.RefreshImage = false;
            }
            appState.immvisionParamsStatic.RefreshImage = false;
        }
    }

    ImmVision::ImageWidgets::s_CollapsingHeader_CacheState_Sync = false;
}


std::atomic<bool> isThreadRunning(false);

void status() {
    if (calibstatus != 0 && !isThreadRunning) {
        isThreadRunning = true;
        std::thread t([&]() {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            calibstatus = 0;
            isThreadRunning = false;
        });
        t.detach();
    }

    switch (calibstatus) {
        case 1:
            ImGui::Text("标定中。。。");
            break;
        case 2:
            ImGui::Text("标定完成");
            break;
        case 3:
            ImGui::Text("保存完成");
            break;
    }


}

#include "hello_imgui/hello_imgui.h"
#include "hello_imgui/hello_imgui_font.h"
#include "hello_imgui/hello_imgui_assets.h"
#include "hello_imgui/hello_imgui.h"
#include "hello_imgui/imgui_default_settings.h"

// Demonstrate how to load additional fonts (fonts - part 1/3)
ImFont *gCustomFont = nullptr;

void MyLoadFonts() {

//    HelloImGui::ImGuiDefaultSettings::LoadDefaultFont_WithFontAwesomeIcons(); // The font that is loaded first is the default font
    ImGuiIO &io = ImGui::GetIO();
//    HelloImGui::SetAssetsFolder("/home/nn/Documents/immvision/src/demos_immvision/calib_camera/assets");
    gCustomFont = HelloImGui::LoadFontTTF_WithFontAwesomeIcons("fonts/MiSans-Normal.ttf", 16.f,
                                                               io.Fonts->GetGlyphRangesJapanese()); // will be loaded from the assets folder

    // 将字体从字节数组加载到ImFontAtlas中
//    ImFontAtlas* fontAtlas = io.Fonts;
//    ImFontConfig font_cfg;
//
//    font_cfg.FontDataOwnedByAtlas = false;
//    font_cfg.MergeMode            = true;
////    font_cfg.OversampleH = font_cfg.OversampleV = 1;
//    font_cfg.FontBuilderFlags |= ImGuiFreeTypeBuilderFlags_LoadColor;
//
//    fontAtlas->AddFontFromMemoryTTF(font_data, font_data_size, 16.f, &font_cfg, fontAtlas->GetGlyphRangesChineseFull());
    // See ImGuiFreeType::RasterizationFlags
//    float dpiFactor = HelloImGui::DpiFontLoadingFactor();
//    ImGui::GetIO().FontGlobalScale = dpiFactor;

}


// The Gui of the status bar
void StatusBarGui(AppState &app_state) {
    ImGui::Text("Using backend: %s", HelloImGui::GetBackendDescription().c_str());
    ImGui::SameLine();
//    if (app_state.rocket_state == AppState::RocketState::Preparing)
//    {
//        ImGui::Text("  -  Rocket completion: ");
//        ImGui::SameLine();
//        ImGui::ProgressBar(app_state.rocket_progress, HelloImGui::EmToVec2(7.0f, 1.0f));
//    }
}

void ShowAboutWindow(bool *p_open) {
    if (*p_open)
        ImGui::OpenPopup("关于");
    ImGui::SetNextWindowSize(ImVec2(400, 200));  // 设置窗口大小为宽度400，高度200
    if (ImGui::BeginPopupModal("关于", NULL, ImGuiWindowFlags_MenuBar)) {

        ImGui::Text("远峰科技 2024");
        ImGui::Separator();
        if (ImGui::Button("关闭")) {
            ImGui::CloseCurrentPopup();
            *p_open = false;
        }
        ImGui::EndPopup();
    }
}

// The menu gui
void ShowMenuGui(HelloImGui::RunnerParams &runnerParams, bool &show_tool_about) {
    HelloImGui::ShowAppMenu(runnerParams);
    HelloImGui::ShowViewMenu(runnerParams);

//    static bool show_tool_about = false;
//    if (ImGui::BeginMenu("Other")) {
//        ImGui::MenuItem("About", NULL, &show_tool_about);
//        ImGui::EndMenu();
//    }
    ShowAboutWindow(&show_tool_about);
}

void ShowAppMenuItems(bool &show_tool_about) {
    if (ImGui::MenuItem("关于应用", NULL, &show_tool_about));
}

void ShowTopToolbar(AppState &appState) {
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

void ShowRightToolbar(AppState &appState) {
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
std::vector<HelloImGui::DockingSplit> CreateDefaultDockingSplits() {
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
    splitMainMisc.direction = ImGuiDir_Down;
    splitMainMisc.ratio = 0.25f;

    // Then, add a space to the left which occupies a column whose width is 25% of the app width
    HelloImGui::DockingSplit splitMainCommand;
    splitMainCommand.initialDock = "MainDockSpace";
    splitMainCommand.newDock = "CommandSpace";
    splitMainCommand.direction = ImGuiDir_Left;
    splitMainCommand.ratio = 0.25f;

    HelloImGui::DockingSplit splitMainTable;
    splitMainTable.initialDock = "MainDockSpace";
    splitMainTable.newDock = "TableSpace";
    splitMainTable.direction = ImGuiDir_Right;
    splitMainTable.ratio = 0.35f;

    std::vector<HelloImGui::DockingSplit> splits{splitMainMisc, splitMainCommand, splitMainTable};
    return splits;
}

std::vector<HelloImGui::DockingSplit> CreateAlternativeDockingSplits() {
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
    splitMainCommand.direction = ImGuiDir_Down;
    splitMainCommand.ratio = 0.5f;

    HelloImGui::DockingSplit splitMainMisc;
    splitMainMisc.initialDock = "MainDockSpace";
    splitMainMisc.newDock = "MiscSpace";
    splitMainMisc.direction = ImGuiDir_Left;
    splitMainMisc.ratio = 0.3f;

    HelloImGui::DockingSplit splitMainTable;
    splitMainTable.initialDock = "MainDockSpace";
    splitMainTable.newDock = "TableSpace";
    splitMainTable.direction = ImGuiDir_Right;
    splitMainTable.ratio = 0.4f;

    std::vector<HelloImGui::DockingSplit> splits{splitMainCommand, splitMainMisc, splitMainTable};
    return splits;
}

//
// 2. Define the Dockable windows
//
std::vector<HelloImGui::DockableWindow> CreateDockableWindows(AppState &appState) {
    // A window named "FeaturesDemo" will be placed in "CommandSpace". Its Gui is provided by "GuiWindowDemoFeatures"
    HelloImGui::DockableWindow featuresDemoWindow;
    featuresDemoWindow.label = "配置";
    featuresDemoWindow.dockSpaceName = "CommandSpace";
    featuresDemoWindow.GuiFunction = [&] { AOIParams(appState); };

    // A layout customization window will be placed in "MainDockSpace". Its Gui is provided by "GuiWindowLayoutCustomization"
    HelloImGui::DockableWindow layoutCustomizationWindow;
    layoutCustomizationWindow.label = "查看";
    layoutCustomizationWindow.dockSpaceName = "MainDockSpace";
    layoutCustomizationWindow.GuiFunction = [&appState]() { ImageView(appState); };

    // A Log window named "Logs" will be placed in "MiscSpace". It uses the HelloImGui logger gui
    HelloImGui::DockableWindow logsWindow;
    logsWindow.label = "Logs";
    logsWindow.dockSpaceName = "MiscSpace";
//    logsWindow.GuiFunction = [] { HelloImGui::LogGui(); };

    // A Window named "Dear ImGui Demo" will be placed in "MainDockSpace"
    HelloImGui::DockableWindow dearImGuiDemoWindow;
    dearImGuiDemoWindow.label = "测量";
    dearImGuiDemoWindow.dockSpaceName = "TableSpace";
    dearImGuiDemoWindow.imGuiWindowFlags = ImGuiWindowFlags_MenuBar;
    dearImGuiDemoWindow.GuiFunction = [&appState] {
        RightTable(appState);
//        ImGui::ShowDemoWindow();
    };

    // additionalWindow is initially not visible (and not mentioned in the view menu).
    // it will be opened only if the user chooses to display it
    HelloImGui::DockableWindow additionalWindow;
    additionalWindow.label = "Additional Window";
    additionalWindow.isVisible = false;               // this window is initially hidden,
    additionalWindow.includeInViewMenu = false;       // it is not shown in the view menu,
    additionalWindow.rememberIsVisible = false;       // its visibility is not saved in the settings file,
    additionalWindow.dockSpaceName = "MiscSpace";     // when shown, it will appear in BottomSpace.
    additionalWindow.GuiFunction = [] { ImGui::Text("This is the additional window"); };

    std::vector<HelloImGui::DockableWindow> dockableWindows{
            featuresDemoWindow,
            layoutCustomizationWindow,
            logsWindow,
            dearImGuiDemoWindow,
            additionalWindow,
    };
    return dockableWindows;
}

//
// 3. Define the layouts:
//        A layout is stored inside DockingParams, and stores the splits + the dockable windows.
//        Here, we provide the default layout, and two alternative layouts.
//
HelloImGui::DockingParams CreateDefaultLayout(AppState &appState) {
    HelloImGui::DockingParams dockingParams;
    // dockingParams.layoutName = "Default"; // By default, the layout name is already "Default"
    dockingParams.dockingSplits = CreateDefaultDockingSplits();
    dockingParams.dockableWindows = CreateDockableWindows(appState);
    return dockingParams;
}

std::vector<HelloImGui::DockingParams> CreateAlternativeLayouts(AppState &appState) {
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
        for (auto &window: tabsLayout.dockableWindows)
            window.dockSpaceName = "MainDockSpace";
        // In "Tabs Layout", no split is created
        tabsLayout.dockingSplits = {};
    }
    return {alternativeLayout, tabsLayout};
}

int main() {
//
//    HelloImGui::RunnerParams params;
//    params.appWindowParams.windowGeometry.size = {1280, 720};
//    params.appWindowParams.windowTitle = "AOI CAMERA";
//    params.imGuiWindowParams.defaultImGuiWindowType = HelloImGui::DefaultImGuiWindowType::ProvideFullScreenWindow;
//
//    // Fonts need to be loaded at the appropriate moment during initialization (fonts - part 2/3)
//    params.callbacks.LoadAdditionalFonts = MyLoadFonts; // LoadAdditionalFonts is a callback that we set with our own font loading function
//
//    params.callbacks.ShowGui = gui;
//
//    params.imGuiWindowParams.showMenuBar = false;
//    params.imGuiWindowParams.showMenu_App = false;
//
//    params.imGuiWindowParams.showStatusBar = true;
//
//    params.callbacks.ShowStatus = status;
//
//    HelloImGui::Run(params);
//    return 0;

    //#############################################################################################
    // Part 1: Define the application state, fill the status and menu bars, load additional font
    //#############################################################################################

    // Our application state
    AppState appState;

    // Hello ImGui params (they hold the settings as well as the Gui callbacks)
    HelloImGui::RunnerParams runnerParams;
    runnerParams.appWindowParams.windowTitle = "AOI PINs";
    runnerParams.imGuiWindowParams.menuAppTitle = "AOI";
    runnerParams.appWindowParams.windowGeometry.size = {1200, 1000};
    runnerParams.appWindowParams.restorePreviousGeometry = true;

    // Our application uses a borderless window, but is movable/resizable
//    runnerParams.appWindowParams.borderless = true;
//    runnerParams.appWindowParams.borderlessMovable = true;
//    runnerParams.appWindowParams.borderlessResizable = true;
//    runnerParams.appWindowParams.borderlessClosable = true;

    // Load additional font
    runnerParams.callbacks.LoadAdditionalFonts = [&appState]() { MyLoadFonts(); };

    //
    // Status bar
    //
    // We use the default status bar of Hello ImGui
    runnerParams.imGuiWindowParams.showStatusBar = true;
    // uncomment next line in order to hide the FPS in the status bar
    // runnerParams.imGuiWindowParams.showStatusFps = false;
    runnerParams.callbacks.ShowStatus = [&appState]() { StatusBarGui(appState); };

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
    runnerParams.callbacks.ShowMenus = [&runnerParams]() { ShowMenuGui(runnerParams, show_tool_about); };
    // Optional: add items to Hello ImGui default App menu

    runnerParams.callbacks.ShowAppMenuItems = []() { ShowAppMenuItems(show_tool_about); };

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
    runnerParams.callbacks.SetupImGuiStyle = []() {
        // Reduce spacing between items ((8, 4) by default)
        ImGui::GetStyle().ItemSpacing = ImVec2(6.f, 4.f);
    };

    //###############################################################################################
    // Part 2: Define the application layout and windows
    //###############################################################################################

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

    //###############################################################################################
    // Part 3: Where to save the app settings
    //###############################################################################################
    // By default, HelloImGui will save the settings in the current folder. This is convenient when developing,
    // but not so much when deploying the app.
    //     You can tell HelloImGui to save the settings in a specific folder: choose between
    //         CurrentFolder
    //         AppUserConfigFolder
    //         AppExecutableFolder
    //         HomeFolder
    //         TempFolder
    //         DocumentsFolder
    //
    //     Note: AppUserConfigFolder is:
    //         AppData under Windows (Example: C:\Users\[Username]\AppData\Roaming)
    //         ~/.config under Linux
    //         "~/Library/Application Support" under macOS or iOS
    runnerParams.iniFolderType = HelloImGui::IniFolderType::AppUserConfigFolder;

    // runnerParams.iniFilename: this will be the name of the ini file in which the settings
    // will be stored.
    // In this example, the subdirectory Docking_Demo will be created under the folder defined
    // by runnerParams.iniFolderType.
    //
    // Note: if iniFilename is left empty, the name of the ini file will be derived
    // from appWindowParams.windowTitle
    runnerParams.iniFilename = "Docking_Demo/Docking_demo.ini";

    //###############################################################################################
    // Part 4: Run the app
    //###############################################################################################
    HelloImGui::DeleteIniSettings(runnerParams);

    // Optional: choose the backend combination
    // ----------------------------------------
//    runnerParams.platformBackendType = HelloImGui::PlatformBackendType::Sdl;
//    runnerParams.rendererBackendType = HelloImGui::RendererBackendType::OpenGL3;

    HelloImGui::Run(runnerParams); // Note: with ImGuiBundle, it is also possible to use ImmApp::Run(...)

    return 0;

};