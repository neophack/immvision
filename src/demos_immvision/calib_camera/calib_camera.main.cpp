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

#include <mutex>
#include <string>
#include <numeric>
#include <unistd.h>


// Poor man's fix for C++ late arrival in the unicode party:
//    - C++17: u8"my string" is of type const char*
//    - C++20: u8"my string" is of type const char8_t*
// However, ImGui text functions expect const char*.
#ifdef __cpp_char8_t
#define U8_TO_CHAR(x) reinterpret_cast<const char*>(x)
#else
#define U8_TO_CHAR(x) x
#endif
// And then, we need to tell gcc to stop validating format string (it gets confused by the u8"" string)
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

struct SobelParams {
    float blur_size = 1.25f;
    int deriv_order = 1;  // order of the derivative
    int k_size = 7;  // size of the extended Sobel kernel it must be 1, 3, 5, or 7 (or -1 for Scharr)
    Orientation orientation = Orientation::Vertical;
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

cv::Mat ComputeSobel(const cv::Mat &image, const SobelParams &params) {
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::Mat img_float;
    gray.convertTo(img_float, CV_32F, 1.0 / 255.0);
    cv::Mat blurred;
    cv::GaussianBlur(img_float, blurred, cv::Size(), params.blur_size, params.blur_size);

    double good_scale = 1.0 / std::pow(2.0, (params.k_size - 2 * params.deriv_order - 2));

    int dx, dy;
    if (params.orientation == Orientation::Vertical) {
        dx = params.deriv_order;
        dy = 0;
    } else {
        dx = 0;
        dy = params.deriv_order;
    }
    cv::Mat r;
    cv::Sobel(blurred, r, CV_64F, dx, dy, params.k_size, good_scale);
    return r;
}

struct AppState {
    cv::Mat image;
    cv::Mat imageDraw;
    cv::Mat imageSobel;
    SobelParams sobelParams;

    ImmVision::ImageParams immvisionParams;
    ImmVision::ImageParams immvisionParamsSobel;
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

    std::string savePath = "./camera_intrinsics.yml";

    int calibModel = 0;
    std::vector<cv::Point2f> image_points_buf;
    std::vector<std::vector<cv::Point2f>> image_points_seq;
    std::vector<std::vector<cv::Point3f>> object_points;

    cv::Mat intrinsics_in;
    cv::Mat dist_coeffs;
    double reproj_err;

    AppState(const int cam) : cameraController() {
//        cameraController.Start();
        image = cv::Mat(600, 800, CV_8UC3, cv::Scalar(0, 0, 0));

        sobelParams = SobelParams();
        imageSobel = ComputeSobel(image, sobelParams);

        immvisionParams = ImmVision::ImageParams();
//        immvisionParams.ImageDisplaySize = cv::Size(500, 0);
        immvisionParams.ZoomKey = "z";
        immvisionParams.RefreshImage = true;
        immvisionParams.forceFullView = true;
        immvisionParams.ShowZoomButtons = false;

        immvisionParamsSobel = ImmVision::ImageParams();
//        immvisionParamsSobel.ImageDisplaySize = cv::Size(500, 0);
        immvisionParamsSobel.ZoomKey = "z";
        immvisionParamsSobel.ShowOptionsPanel = true;
        immvisionParamsSobel.RefreshImage = true;
    }
};


double calibrateCameraModel(CAMERA_MODEL camera_model,
                            std::vector<std::vector<cv::Point3f>> opts,
                            std::vector<std::vector<cv::Point2f>> ipts,
                            cv::Size size,
                            cv::Mat &intrinsics_in,
                            cv::Mat &distortion,
                            int calib_flags,
                            int fisheye_calib_flags) {
    intrinsics_in = cv::Mat();
    distortion = cv::Mat();
    if (camera_model == CAMERA_MODEL::PINHOLE) {
        std::cout << "mono pinhole calibration..." << std::endl;
        std::vector<cv::Mat> rvecs, tvecs;
        cv::Mat dist_coeffs;
        double reproj_err = cv::calibrateCamera(opts, ipts, size, intrinsics_in, dist_coeffs, rvecs, tvecs,
                                                calib_flags);


        if (calib_flags & cv::CALIB_RATIONAL_MODEL) {
            distortion = dist_coeffs(cv::Range::all(), cv::Range(0, 8)); // rational polynomial
        } else {
            distortion = dist_coeffs(cv::Range::all(), cv::Range(0, 5)); // plumb bob
        }
        return reproj_err;
    } else if (camera_model == CAMERA_MODEL::FISHEYE) {
        std::cout << "mono fisheye calibration..." << std::endl;
        std::vector<std::vector<cv::Point2f>> ipts64;
        std::vector<std::vector<cv::Point3f>> opts64;
        for (const auto &i: ipts) {
            ipts64.push_back(std::vector<cv::Point2f>(i.begin(), i.end()));
        }
        for (const auto &o: opts) {
            opts64.push_back(std::vector<cv::Point3f>(o.begin(), o.end()));
        }
        std::vector<cv::Mat> rvecs, tvecs;
        double reproj_err = cv::fisheye::calibrate(opts64, ipts64, size, intrinsics_in, distortion, rvecs, tvecs,
                                                   fisheye_calib_flags);
        return reproj_err;
    }
}


bool CalibParams(SobelParams &params, AppState &appState) {
    bool changed = false;

    if (ImGui::Button(u8"刷新摄像头列表")) {
        if (!appState.cameraController.isRunning) {
            appState.validcam = appState.cameraController.GetAvailableCameras().size();
        }
    }

    for (int i = 0; i < appState.validcam; i++) {
        if (!ImGui::CollapsingHeader(("摄像头 " + std::to_string(i)).c_str())) {
            // 显示所有分辨率选项
//            for (const auto& resolution : resolutions)
            std::vector<std::string> items;

            for (int j = 0; j < resolutions.size(); j++) {
                auto &resolution = resolutions[j];
                std::string label = std::to_string(resolution.first) + "x" + std::to_string(resolution.second);
                items.emplace_back(label);
            }
//            static int item_current = 0;
            std::vector<const char *> itemLabels;
            for (const auto &item: items) {
                itemLabels.push_back(item.c_str());
            }
            if (ImGui::Combo("分辨率", &appState.seli, itemLabels.data(), itemLabels.size())) {
                auto resolution = resolutions[appState.seli];
                appState.width = resolution.first;
                appState.height = resolution.second;
                // 选择了分辨率，执行相应操作
                std::cout << "Selected resolution: " << resolution.first << "x" << resolution.second << std::endl;

            }
//            ImGui::Combo("分辨率", &item_current, items.data()->c_str(), items.data()->size());



            if (ImGui::Button(u8"打开关闭")) {
                appState.cameraController.SetResolution(appState.width, appState.height);
                appState.cameraController.ToggleCamera(i);
                appState.image_points_seq.clear();
                appState.object_points.clear();
                appState.paramsSummary.clear();
            }
        }
        ImGui::Spacing();
    }

    if (!ImGui::CollapsingHeader(u8"内参标定")) {
        int inputFlag = ImGuiInputTextFlags_None;
        int selFlag = ImGuiSelectableFlags_None;
        if (appState.cameraController.isRunning) {
            inputFlag = ImGuiInputTextFlags_ReadOnly;
            selFlag = ImGuiSelectableFlags_Disabled;
        }

        if (ImGui::Selectable(u8"棋盘格", CHESSBOARD == appState.calibPatten, selFlag)) {
            appState.calibPatten = CHESSBOARD;

        }
        if (ImGui::Selectable(u8"圆点", CIRCLES_GRID == appState.calibPatten, selFlag)) {
            appState.calibPatten = CIRCLES_GRID;
        }


        ImGui::InputInt("行", &appState.boardRows, 1, 100, inputFlag);
        ImGui::InputInt("列", &appState.boardCols, 1, 100, inputFlag);
        ImGui::InputInt("尺寸mm", &appState.gridMM, 1, 100, inputFlag);

        ImGui::Spacing();
        ImGui::Text("检测到%d张", appState.image_points_seq.size());
        ImDrawList *draw_list = ImGui::GetWindowDrawList();

        for (int i = 0; i < appState.paramsSummary.size(); i++) {
            // 获取参数信息
            std::string label = std::get<0>(appState.paramsSummary[i]);
            double lo = std::get<1>(appState.paramsSummary[i]);
            double hi = std::get<2>(appState.paramsSummary[i]);
            double progress = std::get<3>(appState.paramsSummary[i]);

            // 显示标签
            ImGui::Text("%s", label.c_str());

            // 计算线条颜色
            ImU32 color = ImGui::GetColorU32(ImVec4(progress, 0.0f, 1.0f, 1.0f));

            // 计算线条的起点和终点
            ImVec2 p1 = ImGui::GetCursorScreenPos();
            p1.x += lo * 100;
            p1.y += i;
            ImVec2 p2 = p1;
            p2.x += hi * 100;

            // 绘制线条
            draw_list->AddLine(p1, p2, color, 4.0f);

            // 添加一些间隔
            ImGui::Spacing();
        }
        ImGui::Spacing();
        if (ImGui::Selectable(u8"针孔模型", CAMERA_MODEL::PINHOLE == appState.calibModel)) {
            appState.calibModel = 0;

        }
        if (ImGui::Selectable(u8"鱼眼模型", CAMERA_MODEL::FISHEYE == appState.calibModel)) {
            appState.calibModel = 1;
        }

        if (ImGui::Button("标定")) {
            ImGui::Text("标定中。。。");
            calibstatus=1;
            if (appState.object_points.size() > 0) {
                int calib_flags = 0; // 标定的标志
                int fisheye_calib_flags = 0; // 鱼眼标定的标志
                appState.reproj_err = calibrateCameraModel(appState.calibModel, appState.object_points,
                                                           appState.image_points_seq,
                                                           appState.image.size(), appState.intrinsics_in,
                                                           appState.dist_coeffs,
                                                           calib_flags,
                                                           fisheye_calib_flags);
            }

            calibstatus=2;
        }
        if (appState.reproj_err > 1e-6) {
            std::stringstream ss;
            ss << "Intrinsics: \n" << appState.intrinsics_in << std::endl;
            ss << "Distortion coefficients: \n" << appState.dist_coeffs << std::endl;
            ss << "Reprojection error: \n" << appState.reproj_err << std::endl;

            char buffer[1000];
            strncpy(buffer, ss.str().c_str(), sizeof(buffer));
            ImGui::InputTextMultiline("内参", buffer, sizeof(buffer));
        }
        ImGui::InputText("路径", appState.savePath.data(), 1000);
        if (ImGui::Button("保存标定")) {
            // Save calibration parameters to YAML file
            cv::FileStorage fs(appState.savePath.data(), cv::FileStorage::WRITE);
            if (fs.isOpened()) {
                fs << "Intrinsics" << appState.intrinsics_in;
                fs << "DistortionCoefficients" << appState.dist_coeffs;
                fs << "ReprojectionError" << appState.reproj_err;
                fs.release();
            }
            calibstatus=3;

        }
    }
    return changed;
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
        while (isRunning) {
            if (CHESSBOARD == appState.calibPatten) {
                cv::Size board_size = cv::Size(appState.boardRows, appState.boardCols);
                if (0 == findChessboardCorners(appState.image, cv::Size(appState.boardRows, appState.boardCols),
                                               appState.image_points_buf)) {

//            std::cout << "can not find corners" << std::endl;
//            exit(1);
                } else {
                    cv::Mat view_gray;
                    cvtColor(appState.image, view_gray, cv::COLOR_RGB2GRAY);
                    cornerSubPix(view_gray, appState.image_points_buf, cv::Size(7, 7), cv::Size(-1, -1),
                                 cv::TermCriteria(
                                         cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
                                         40,
                                         0.001));//7 7zuihao

                    auto ids = generateIds(appState);
                    auto params = getParameters(appState.image_points_buf, ids, appState,
                                                {appState.image.cols, appState.image.rows});
                    bool isgood = is_good_sample(params, appState.image_points_buf, ids, appState.image_points_buf,
                                                 ids);
                    if (isgood) {
                        db.emplace_back(std::make_pair(params, view_gray));

                        std::cout << "isgood:" << isgood << std::endl;
                        int i, j, t;
                        std::vector<cv::Point3f> tempPointSet;
                        for (i = 0; i < board_size.height; i++) {
                            for (j = 0; j < board_size.width; j++) {
                                cv::Point3f realPoint;
                                /* 假设标定板放在世界坐标系中z=0的平面上 */
                                realPoint.x = i * appState.gridMM;
                                realPoint.y = j * appState.gridMM;
                                realPoint.z = 0;
                                tempPointSet.push_back(realPoint);
                            }
                        }
                        appState.object_points.push_back(tempPointSet);

                        //find4QuadCornerSubpix(view_gray,image_points_buf,Size(5,5));// diyushangbiande 0.5pixel//5 5 zuihao
                        appState.image_points_seq.push_back(appState.image_points_buf);
                        appState.image.copyTo(appState.imageDraw);
                        drawChessboardCorners(appState.imageDraw, cv::Size(appState.boardRows, appState.boardCols),
                                              appState.image_points_buf, true);

                        appState.paramsSummary = compute_goodenough();

                    }
                }
            }
            usleep(10000);
        }
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

void gui() {
    static AppState appState(0);

    static float initialListWidth = ImGui::GetFontSize() * 8.5f;
    static float currentListWidth = initialListWidth;

    bool showOptionsColumn = true;
    if ((appState.immvisionParams.ShowOptionsInTooltip) || (!appState.immvisionParams.ShowOptionsPanel))
        showOptionsColumn = false;
    ImVec2 imageSize = set_ImageSize(currentListWidth, showOptionsColumn);

//    std::cout<<showOptionsColumn<<" "<<imageSize.x<<" "<<imageSize.y<<std::endl;


    ImGui::Columns(2);

    //
    // First column: image list
    //
    {
        // Set column width
        {
            static int idxFrame = 0;
            ++idxFrame;
            if (idxFrame <= 2) // The column width is not set at the first frame
                ImGui::SetColumnWidth(0, initialListWidth);

//            ImGui::Text(u8"????????? %d", 123);
            currentListWidth = ImGui::GetColumnWidth(0);
        }
        // Show image list
        ImGui::TextWrapped(R"(
        Cameras
    )");
//        ImGui::Separator();
        cv::Mat tim = appState.cameraController.GetFrame();
        if (!tim.empty()) {
            appState.image = tim;
        }
//    appState.imageSobel = appState.image;
        CalibParams(appState.sobelParams, appState);


//            std::cout<<"num of points :"<<appState.image_points_seq.size()<<std::endl;
        if (!checker.isRunning) {
            if (!tim.empty())
                checker.Start(appState);
        }
        if (!appState.cameraController.isRunning) {
            checker.Stop();
        }

        ImGui::NextColumn();

        //
        // Second column : image
        //
        {
            appState.immvisionParams.ImageDisplaySize = cv::Size((int) imageSize.x, (int) imageSize.y);
            if (!appState.imageDraw.empty()) {
                ImmVision::Image("Checked", appState.imageDraw, &appState.immvisionParams);
                usleep(10e5);
                appState.imageDraw = cv::Mat();
            } else {
                ImmVision::Image("Original", appState.image, &appState.immvisionParams);
//        ImmVision::Image("Deriv", appState.imageSobel, &appState.immvisionParamsSobel);
            }
        }

        ImGui::Columns(1);

//    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    }
}

std::atomic<bool> isThreadRunning(false);

void status() {
    if(calibstatus!=0 && !isThreadRunning){
        isThreadRunning = true;
        std::thread t([&](){
            std::this_thread::sleep_for(std::chrono::seconds(5));
            calibstatus = 0;
            isThreadRunning = false;
        });
        t.detach();
    }

    switch(calibstatus){
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
    HelloImGui::SetAssetsFolder("/home/nn/Documents/immvision/src/demos_immvision/calib_camera/assets");
//    HelloImGui::ImGuiDefaultSettings::LoadDefaultFont_WithFontAwesomeIcons(); // The font that is loaded first is the default font
    ImGuiIO &io = ImGui::GetIO();
    gCustomFont = HelloImGui::LoadFontTTF_WithFontAwesomeIcons("fonts/MiSans-Normal.ttf", 16.f,
                                                               io.Fonts->GetGlyphRangesJapanese()); // will be loaded from the assets folder
}

int main() {

    HelloImGui::RunnerParams params;
    params.appWindowParams.windowGeometry.size = {1280, 720};
    params.appWindowParams.windowTitle = "CALIB CAMERA";
    params.imGuiWindowParams.defaultImGuiWindowType = HelloImGui::DefaultImGuiWindowType::ProvideFullScreenWindow;

    // Fonts need to be loaded at the appropriate moment during initialization (fonts - part 2/3)
    params.callbacks.LoadAdditionalFonts = MyLoadFonts; // LoadAdditionalFonts is a callback that we set with our own font loading function

    params.callbacks.ShowGui = gui;

    params.imGuiWindowParams.showMenuBar = false;
    params.imGuiWindowParams.showMenu_App = false;

    params.imGuiWindowParams.showStatusBar = true;

    params.callbacks.ShowStatus = status;

    HelloImGui::Run(params);
    return 0;
};