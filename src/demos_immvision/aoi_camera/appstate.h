//
// Created by nn on 4/3/24.
//

#ifndef IMMVISION_APPSTATE_H
#define IMMVISION_APPSTATE_H

#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>
#include "immvision/immvision.h"
#include "camera.h"

struct AppState {
    cv::Mat image;
    cv::Mat imageDraw;
    cv::Mat imageCap;
    cv::Mat imageProcess;
    std::vector<cv::Mat> imageProcess2;
    std::vector<cv::Mat> imageProcess3;
    std::vector<cv::Mat> imageProcess4;
    cv::Mat imageSobel;
    //    SobelParams sobelParams;

    ImmVision::ImageParams immvisionParams;
    ImmVision::ImageParams immvisionParamsStatic;
    std::vector<ImmVision::ImageParams> immvisionParamsStaticSub;
    std::vector<ImmVision::ImageParams> immvisionParamsStaticSub2;

    int validcam;
    int seli = 4;
    int width;
    int height;

    std::vector<std::tuple<std::string, double, double, double>> paramsSummary;

    std::string savePath = "./aoi_calib.yml";
    std::string savePathPair = "./pin_pair.bin";
    std::string savePathBox = "./pin_box.bin";
    std::string savePathAngle = "./pin_angle.bin";
    std::string savePathLogin = "./pin_login.bin";
    std::string configPath = "aoi_config.xml";

    int calibModel = 0;
    std::vector<cv::Point2f> image_points_buf;
    std::vector<std::vector<cv::Point2f>> image_points_seq;
    std::vector<std::vector<cv::Point3f>> object_points;

    cv::Mat intrinsics_in;
    cv::Mat dist_coeffs;
    double reproj_err;
    double mmPrePixel = 0.03734429785177405;

    std::vector<std::vector<float>> circle_info;
    int largerCircleMoved = -1;
    // checker
    int noiseThreshold = 5; // 设置噪声阈值
    int qsize = 5;

    float scale = 1 / 28.0;

    std::vector<std::vector<std::vector<float>>> pin_info;
    std::vector<int> connectedIndices;

    int large_circle_id;
    double circleDistance = 0;
    std::vector<int> pass_status;
    std::vector<int> angle_pass_status;

    struct MEASURE_BOX {
        float rect_x = 0.f;
        float rect_y = 0.f;
        float rect_w = 0.f;
        float rect_h = 0.f;

        float bin_threshold = 127.f;
        bool flip = false;
    };
    std::vector<MEASURE_BOX> measure_boxes;
    std::vector<MEASURE_BOX> last_measure_boxes;
    int sel_box = 0;

    float relativeRectX = 0.f;
    float relativeRectY = 0.f;
    float relativeRectW = 0.f;
    float relativeRectH = 0.f;

    struct MEASURE_SETTING {
        int box = 0;
        int pin0 = 0;
        int pin1 = 1;
        float minDistance = 30.;
        float maxDistance = 60.;
        float distance = -1.;
        std::vector<float> distanceVec;
    };

    bool needRefresh = false;
    bool algorithmRunning = false;
    bool algorithmFinished = true;
    int needMeasure = -1;
    bool measureRunning = false;
    bool measureFinished = true;

    std::vector<MEASURE_SETTING> pin_measure_settings;

    struct ANGLE_MEASURE_SETTING {
        int box = 0;
        int pin0 = 0;
        int pin1 = 1;
        int pin2 = 1;
        int pin3 = 2;
        float minAngle = 30.;
        float maxAngle = 60.;
        float angle = -1.;
        std::vector<float> angleVec;
    };
    std::vector<ANGLE_MEASURE_SETTING> angle_measure_settings;

    // 存储所有圆点的容器
    std::vector<cv::Point2f> cad_points;

// 公差范围
    float cad_tolerance = 0.125f;


    std::vector<std::string> scanResVec;
    std::string scanResJson;

    struct LoginInfo {
        //        std::string code;
        std::string mo;
        std::string line;
        std::string station;
        std::string workshop;
        std::string model;
        std::string result;
        std::string user;
        std::string url;
    };
    LoginInfo loginInfo;

    std::string response;

    bool scaner_running = false;

    // 设置参数
    float gamma = 0.5; // gamma值，控制曲线形状
    float alpha = 2.0; // 增益系数，控制亮度增强程度

    // 创建查找表
    cv::Mat lookup_table;

    AppState() {
        lookup_table = cv::Mat(1, 256, CV_8U);
        for (int i = 0; i < 256; ++i) {
            float normalized_pixel = i / 255.0;
            float enhanced_pixel = alpha * pow(normalized_pixel, gamma) * 255.0;
            lookup_table.at<uchar>(i) = static_cast<uchar>(std::min(enhanced_pixel, 255.0f));
        }
        //        cameraController.Start();
        // image = cv::Mat(600, 800, CV_8UC3, cv::Scalar(0, 0, 0));

        immvisionParams = ImmVision::ImageParams();
        immvisionParams.ImageDisplaySize = cv::Size(500, 0);
        immvisionParams.ZoomKey = "z";
        immvisionParams.RefreshImage = true;
        immvisionParams.forceFullView = true;
        immvisionParams.ShowZoomButtons = false;
        immvisionParams.ShowOptionsInTooltip = true;

        immvisionParamsStatic = ImmVision::ImageParams();
        immvisionParamsStatic.ImageDisplaySize = cv::Size(500, 0);
        immvisionParamsStatic.ZoomKey = "z";
        immvisionParamsStatic.RefreshImage = false;
        immvisionParamsStatic.forceFullView = true;
        immvisionParamsStatic.ShowZoomButtons = false;
        immvisionParamsStatic.ShowOptionsInTooltip = true;

        //         immvisionParamsStaticSub = ImmVision::ImageParams();
        // //        immvisionParamsSub.ImageDisplaySize = cv::Size(0, 0);
        //         immvisionParamsStaticSub.ZoomKey = "c";
        // //        immvisionParamsSub.ShowOptionsPanel = true;
        //         immvisionParamsStaticSub.RefreshImage = false;
        //         immvisionParamsStaticSub.forceFullView = true;
        //         immvisionParamsStaticSub.ShowZoomButtons = false;
        // //        immvisionParamsSub.ShowOptionsInTooltip = false;
        //         immvisionParamsStaticSub.ShowOptionsButton = false;
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
        try {
            std::ifstream in(savePathPair, std::ios::binary);
            if (in.is_open()) {
                size_t size;
                in.read((char *) &size, sizeof(size));
                pin_measure_settings.resize(size);
                in.read((char *) pin_measure_settings.data(), size * sizeof(MEASURE_SETTING));
                in.close();
            }
        }
        catch (const std::exception &e) {
            // 在此处处理异常
            std::cout << "loadBIN时发生异常: " << e.what() << std::endl;
        }
    }

    void saveBoxSettings() {
        std::ofstream out(savePathBox, std::ios::binary);
        if (out.is_open()) {
            size_t size = measure_boxes.size();
            out.write((char *) &size, sizeof(size));
            out.write((char *) measure_boxes.data(), size * sizeof(MEASURE_BOX));
            out.close();
        }
    }

    void loadBoxSettings() {
        try {
            std::ifstream in(savePathBox, std::ios::binary);
            if (in.is_open()) {
                size_t size;
                in.read((char *) &size, sizeof(size));
                measure_boxes.resize(size);
                in.read((char *) measure_boxes.data(), size * sizeof(MEASURE_BOX));
                in.close();
            }
        }
        catch (const std::exception &e) {
            // 在此处处理异常
            std::cout << "loadBIN时发生异常: " << e.what() << std::endl;
        }
    }

    void saveAngleSettings() {
        std::ofstream out(savePathAngle, std::ios::binary);
        if (out.is_open()) {
            size_t size = angle_measure_settings.size();
            out.write((char *) &size, sizeof(size));
            out.write((char *) angle_measure_settings.data(), size * sizeof(ANGLE_MEASURE_SETTING));
            out.close();
        }
    }

    void loadAngleSettings() {
        try {
            std::ifstream in(savePathAngle, std::ios::binary);
            if (in.is_open()) {
                size_t size;
                in.read((char *) &size, sizeof(size));
                angle_measure_settings.resize(size);
                in.read((char *) angle_measure_settings.data(), size * sizeof(ANGLE_MEASURE_SETTING));
                in.close();
            }
        }
        catch (const std::exception &e) {
            // 在此处处理异常
            std::cout << "loadBIN时发生异常: " << e.what() << std::endl;
        }
    }

    void saveLoginInfo() {
        std::ofstream file(savePathLogin);
        if (file.is_open()) {
            file << loginInfo.mo << '\n';
            file << loginInfo.line << '\n';
            file << loginInfo.station << '\n';
            file << loginInfo.workshop << '\n';
            file << loginInfo.model << '\n';
            file << loginInfo.result << '\n';
            file << loginInfo.user << '\n';
            file << loginInfo.url << '\n';
            file.close();
        } else {
            std::cerr << "Unable to open file " << savePathLogin << std::endl;
        }
    }

    void loadLoginInfo() {
        try {
            std::ifstream file(savePathLogin);
            if (file.is_open()) {
                std::getline(file, loginInfo.mo);
                std::getline(file, loginInfo.line);
                std::getline(file, loginInfo.station);
                std::getline(file, loginInfo.workshop);
                std::getline(file, loginInfo.model);
                std::getline(file, loginInfo.result);
                std::getline(file, loginInfo.user);
                std::getline(file, loginInfo.url);
                file.close();
            } else {
                std::cerr << "Unable to open file " << savePathLogin << std::endl;
            }
        }
        catch (const std::exception &e) {
            // 在此处处理异常
            std::cout << "loadBIN时发生异常: " << e.what() << std::endl;
        }
    }

    void saveCalibrationParameters() {
        // Save calibration parameters to YAML file
        cv::FileStorage fs(savePath, cv::FileStorage::WRITE);
        if (fs.isOpened()) {
            fs << "Intrinsics" << intrinsics_in;
            fs << "DistortionCoefficients" << dist_coeffs;
            fs << "ReprojectionError" << reproj_err;
            fs << "mmPrePixel" << mmPrePixel;

            fs.release();
        }
    }

    void loadCalibrationParameters() {
        try {
            cv::FileStorage fs(savePath, cv::FileStorage::READ);
            if (fs.isOpened()) {
                fs["Intrinsics"] >> intrinsics_in;
                fs["DistortionCoefficients"] >> dist_coeffs;
                fs["ReprojectionError"] >> reproj_err;
                fs["mmPrePixel"] >> mmPrePixel;


                fs.release();
            }
        }
        catch (const std::exception &e) {
            // 在此处处理异常
            std::cout << "load标定参数时发生异常: " << e.what() << std::endl;
        }
    }

    void loadConfigFromXml(const std::string &filename) {
        try {

            cv::FileStorage fs(filename, cv::FileStorage::READ);
            if (!fs.isOpened()) {
                std::cerr << "Failed to open XML configuration file: " << filename << std::endl;
                return;
            }

            fs["reproj_err"] >> reproj_err;
            fs["mmPrePixel"] >> mmPrePixel;

            fs["noiseThreshold"] >> noiseThreshold;
            fs["qsize"] >> qsize;
            fs["scale"] >> scale;

            // Load measure_boxes
            cv::FileNode measure_boxes_node = fs["measure_boxes"];
            measure_boxes.clear();
            for (cv::FileNodeIterator it = measure_boxes_node.begin(); it != measure_boxes_node.end(); ++it) {
                cv::FileNode box_node = *it;
                float rect_x, rect_y, rect_w, rect_h, bin_threshold;
                bool flip;
                box_node["rect_x"] >> rect_x;
                box_node["rect_y"] >> rect_y;
                box_node["rect_w"] >> rect_w;
                box_node["rect_h"] >> rect_h;
                box_node["bin_threshold"] >> bin_threshold;
                box_node["flip"] >> flip;
                measure_boxes.emplace_back(rect_x, rect_y, rect_w, rect_h, bin_threshold, flip);
            }

            // Load last_measure_boxes
            cv::FileNode last_measure_boxes_node = fs["last_measure_boxes"];
            last_measure_boxes.clear();
            for (cv::FileNodeIterator it = last_measure_boxes_node.begin();
                 it != last_measure_boxes_node.end(); ++it) {
                cv::FileNode box_node = *it;
                float rect_x, rect_y, rect_w, rect_h, bin_threshold;
                bool flip;
                box_node["rect_x"] >> rect_x;
                box_node["rect_y"] >> rect_y;
                box_node["rect_w"] >> rect_w;
                box_node["rect_h"] >> rect_h;
                box_node["bin_threshold"] >> bin_threshold;
                box_node["flip"] >> flip;
                last_measure_boxes.emplace_back(rect_x, rect_y, rect_w, rect_h, bin_threshold, flip);
            }

            // Load pin_measure_settings
            cv::FileNode pin_measure_settings_node = fs["pin_measure_settings"];
            pin_measure_settings.clear();
            for (cv::FileNodeIterator it = pin_measure_settings_node.begin();
                 it != pin_measure_settings_node.end(); ++it) {
                cv::FileNode setting_node = *it;
                int box, pin0, pin1;
                float minDistance, maxDistance, distance;
                std::vector<float> distanceVec;
                setting_node["box"] >> box;
                setting_node["pin0"] >> pin0;
                setting_node["pin1"] >> pin1;
                setting_node["minDistance"] >> minDistance;
                setting_node["maxDistance"] >> maxDistance;
                setting_node["distance"] >> distance;
                cv::FileNode distanceVec_node = setting_node["distanceVec"];
                for (cv::FileNodeIterator subIt = distanceVec_node.begin();
                     subIt != distanceVec_node.end(); ++subIt) {
                    float dist;
                    *subIt >> dist;
                    distanceVec.push_back(dist);
                }
                pin_measure_settings.emplace_back(box, pin0, pin1, minDistance, maxDistance, distance, distanceVec);
            }

            // Load angle_measure_settings
            cv::FileNode angle_measure_settings_node = fs["angle_measure_settings"];
            angle_measure_settings.clear();
            for (cv::FileNodeIterator it = angle_measure_settings_node.begin();
                 it != angle_measure_settings_node.end(); ++it) {
                cv::FileNode setting_node = *it;
                int box, pin0, pin1, pin2, pin3;
                float minAngle, maxAngle, angle;
                std::vector<float> angleVec;
                setting_node["box"] >> box;
                setting_node["pin0"] >> pin0;
                setting_node["pin1"] >> pin1;
                setting_node["pin2"] >> pin2;
                setting_node["pin3"] >> pin3;
                setting_node["minAngle"] >> minAngle;
                setting_node["maxAngle"] >> maxAngle;
                setting_node["angle"] >> angle;
                cv::FileNode angleVec_node = setting_node["angleVec"];
                for (cv::FileNodeIterator subIt = angleVec_node.begin(); subIt != angleVec_node.end(); ++subIt) {
                    float ang;
                    *subIt >> ang;
                    angleVec.push_back(ang);
                }
                angle_measure_settings.emplace_back(box, pin0, pin1, pin2, pin3, minAngle, maxAngle, angle,
                                                    angleVec);
            }

            fs["intrinsics_in"] >> intrinsics_in;
            fs["dist_coeffs"] >> dist_coeffs;

            fs["loginInfo"]["mo"] >> loginInfo.mo;
            fs["loginInfo"]["line"] >> loginInfo.line;
            fs["loginInfo"]["station"] >> loginInfo.station;
            fs["loginInfo"]["workshop"] >> loginInfo.workshop;
            fs["loginInfo"]["model"] >> loginInfo.model;
            fs["loginInfo"]["result"] >> loginInfo.result;
            fs["loginInfo"]["user"] >> loginInfo.user;
            fs["loginInfo"]["url"] >> loginInfo.url;

            fs["gamma"] >> gamma;
            fs["alpha"] >> alpha;

            fs["cad_points"] >> cad_points;
            fs["cad_tolerance"] >> cad_tolerance;

            fs.release();
        }
        catch (const std::exception &e) {
            // 在此处处理异常
            std::cout << "load xml config时发生异常: " << e.what() << std::endl;
        }
    }

    void saveConfigToXml(const std::string &filename) {
        cv::FileStorage fs(filename, cv::FileStorage::WRITE);
        if (!fs.isOpened()) {
            std::cerr << "Failed to open XML configuration file for saving: " << filename << std::endl;
            return;
        }

        fs << "reproj_err" << reproj_err;
        fs << "mmPrePixel" << mmPrePixel;

        fs << "noiseThreshold" << noiseThreshold;
        fs << "qsize" << qsize;

        // Save measure_boxes
        fs << "measure_boxes" << "[";
        for (const auto &box: measure_boxes) {
            fs << "{";
            fs << "rect_x" << box.rect_x;
            fs << "rect_y" << box.rect_y;
            fs << "rect_w" << box.rect_w;
            fs << "rect_h" << box.rect_h;
            fs << "bin_threshold" << box.bin_threshold;
            fs << "flip" << box.flip;
            fs << "}";
        }
        fs << "]";

        // Save last_measure_boxes
        fs << "last_measure_boxes" << "[";
        for (const auto &box: last_measure_boxes) {
            fs << "{";
            fs << "rect_x" << box.rect_x;
            fs << "rect_y" << box.rect_y;
            fs << "rect_w" << box.rect_w;
            fs << "rect_h" << box.rect_h;
            fs << "bin_threshold" << box.bin_threshold;
            fs << "flip" << box.flip;
            fs << "}";
        }
        fs << "]";

        // Save pin_measure_settings
        fs << "pin_measure_settings" << "[";
        for (const auto &setting: pin_measure_settings) {
            fs << "{";
            fs << "box" << setting.box;
            fs << "pin0" << setting.pin0;
            fs << "pin1" << setting.pin1;
            fs << "minDistance" << setting.minDistance;
            fs << "maxDistance" << setting.maxDistance;
            fs << "distance" << setting.distance;
            fs << "distanceVec" << setting.distanceVec;
            fs << "}";
        }
        fs << "]";

        fs << "needRefresh" << needRefresh;
        fs << "algorithmRunning" << algorithmRunning;
        fs << "algorithmFinished" << algorithmFinished;
        fs << "needMeasure" << needMeasure;
        fs << "measureRunning" << measureRunning;
        fs << "measureFinished" << measureFinished;

        // Save angle_measure_settings
        fs << "angle_measure_settings" << "[";
        for (const auto &setting: angle_measure_settings) {
            fs << "{";
            fs << "box" << setting.box;
            fs << "pin0" << setting.pin0;
            fs << "pin1" << setting.pin1;
            fs << "pin2" << setting.pin2;
            fs << "pin3" << setting.pin3;
            fs << "minAngle" << setting.minAngle;
            fs << "maxAngle" << setting.maxAngle;
            fs << "angle" << setting.angle;
            fs << "angleVec" << setting.angleVec;
            fs << "}";
        }
        fs << "]";

        fs << "intrinsics_in" << intrinsics_in;
        fs << "dist_coeffs" << dist_coeffs;

        fs << "loginInfo" << "{";
        fs << "mo" << loginInfo.mo;
        fs << "line" << loginInfo.line;
        fs << "station" << loginInfo.station;
        fs << "workshop" << loginInfo.workshop;
        fs << "model" << loginInfo.model;
        fs << "result" << loginInfo.result;
        fs << "user" << loginInfo.user;
        fs << "url" << loginInfo.url;
        fs << "}";

        fs << "gamma" << gamma;
        fs << "alpha" << alpha;

        fs << "cad_points" << cad_points;
        fs << "cad_tolerance" << cad_tolerance;


        fs.release();
    }


};

#endif // IMMVISION_APPSTATE_H
