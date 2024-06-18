

//#include <lccv.hpp>
#include <opencv2/opencv.hpp>
#include <thread>
#include <atomic>

//class CameraControllerLCCV {
//public:
//    CameraControllerLCCV() : isRunning(false) {}
//
//    ~CameraControllerLCCV() {
//        Stop();
//    }
//
//    std::vector<int> GetAvailableCameras() {
//        std::vector<int> availableCameras;
//        if (!isRunning) {
//            for (int i = 0; i < 10; i++) {
//
//            }
//        }
//        return availableCameras;
//    }
//
//    std::pair<int, int> GetResolution(int cam) {
//        if (!isRunning) {
//
//        }
//        return std::make_pair(-1, -1);
//    }
//
//    double GetFrameRate(int cam) {
//        if (!isRunning) {
//
//        }
//        return -1.0;
//    }
//
//    void Start(int cam) {
//        if (!isRunning) {
//            isRunning = true;
//            cameraThread = std::thread(&CameraControllerLCCV::CameraThreadFunc, this, cam);
//            cameraThread.detach();
//        }
//    }
//
//    void Stop() {
//        if (isRunning) {
//            isRunning = false;
//            // cameraThread.join();
//        }
//    }
//
//    void ToggleCamera(int cam) {
//        if (isRunning) {
//            Stop();
//        } else {
//            Start(cam);
//        }
//    }
//
//    cv::Mat GetFrame() {
//        std::lock_guard<std::mutex> lock(frameMutex);
//        return frame;
//    }
//
//    cv::Mat GetThumbnail() {
//        std::lock_guard<std::mutex> lock(frameMutex);
//
//        if(!frame.empty()){
//            cv::resize(frame, thumbnail,  cv::Size(), 0.25, 0.25);
//        }
//        return thumbnail;
//    }
//
//
//    bool SetResolution(int widtht, int heightt) {
//        if (!isRunning) {
//            width = widtht;
//            height = heightt;
//            return true;
//
//        }
//        return false;
//    }
//
//    std::atomic<bool> isRunning;
//private:
//    cv::Mat frame;
//
//    std::thread cameraThread;
//    std::mutex frameMutex;
//    int width;
//    int height;
//    cv::Mat thumbnail;
//
//    void CameraThreadFunc(int cam) {
//        cv::Mat image;
//        lccv::PiCamera cap;
//        cap.options->video_width=width;
//        cap.options->video_height=height;
//        cap.options->framerate=30;
//        cap.options->setExposureMode(Exposure_Modes::EXPOSURE_CUSTOM);
//        cap.options->shutter = 10000;
//        cap.options->verbose=true;
//        cap.startVideo();
//        while (isRunning) {
//            if(!cap.getVideoFrame(image,1000)){
//                // std::cout<<"Timeout error"<<std::endl;
//            }
//            else{
//                std::lock_guard<std::mutex> lock(frameMutex);
//                cv::flip(image, frame, -1); // -1 means flipping around both axes
//                // frame = image;
//            }
//
//        }
//        cap.stopVideo();
//    }
//};

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
            cameraThread.detach();
        }
    }

    void Stop() {
        if (isRunning) {
            isRunning = false;
            // cameraThread.join();
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

        cv::Mat GetThumbnail() {
        std::lock_guard<std::mutex> lock(frameMutex);

        if(!frame.empty()){
            cv::resize(frame, thumbnail,  cv::Size(), 0.25, 0.25);
        }
        return thumbnail;
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
    cv::Mat thumbnail;
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
