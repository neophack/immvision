//
// Created by nn on 4/9/24.
//

#ifndef IMMVISION_SCANER_H
#define IMMVISION_SCANER_H

#include <stdio.h>
#include <wchar.h>
#include <string>
#include <hidapi.h>
#include <map>
#include <iostream>
#include <thread>
#include <mutex> // 添加互斥锁头文件


#define MAX_STR 1024
#define VENDOR_ID 0x0c2e
#define PRODUCT_ID 0x0901

std::map<int, char> hidNoShiftTable = {
        {0x04, 'a'}, {0x05, 'b'}, {0x06, 'c'}, {0x07, 'd'}, {0x08, 'e'}, {0x09, 'f'}, {0x0a, 'g'}, {0x0b, 'h'}, {0x0c, 'i'}, {0x0d, 'j'},
        {0x0e, 'k'}, {0x0f, 'l'}, {0x10, 'm'}, {0x11, 'n'}, {0x12, 'o'}, {0x13, 'p'}, {0x14, 'q'}, {0x15, 'r'}, {0x16, 's'}, {0x17, 't'},
        {0x18, 'u'}, {0x19, 'v'}, {0x1a, 'w'}, {0x1b, 'x'}, {0x1c, 'y'}, {0x1d, 'z'}, {0x1e, '1'}, {0x1f, '2'}, {0x20, '3'}, {0x21, '4'},
        {0x22, '5'}, {0x23, '6'}, {0x24, '7'}, {0x25, '8'}, {0x26, '9'}, {0x27, '0'}, {0x28, '\n'}, {0x2c, ' '}, {0x2d, '-'}, {0x2e, '='},
        {0x2f, '['}, {0x30, ']'}, {0x31, '\\'}, {0x32, '#'}, {0x33, ';'}, {0x34, '\''}, {0x35, '`'}, {0x36, ','}, {0x37, '.'}, {0x38, '/'}
};

std::map<int, char> hidWithShiftTable = {
        {0x04, 'A'}, {0x05, 'B'}, {0x06, 'C'}, {0x07, 'D'}, {0x08, 'E'}, {0x09, 'F'}, {0x0a ,'G'}, {0x0b, 'H'}, {0x0c, 'I'}, {0x0d, 'J'},
        {0x0e, 'K'}, {0x0f, 'L'}, {0x10, 'M'}, {0x11, 'N'}, {0x12, 'O'}, {0x13, 'P'}, {0x14, 'Q'}, {0x15, 'R'}, {0x16, 'S'}, {0x17, 'T'},
        {0x18, 'U'}, {0x19, 'V'}, {0x1a, 'W'}, {0x1b, 'X'}, {0x1c, 'Y'}, {0x1d, 'Z'}, {0x1e, '!'}, {0x1f, '@'}, {0x20, '#'}, {0x21, '$'},
        {0x22, '%'}, {0x23, '^'}, {0x24, '&'}, {0x25, '*'}, {0x26, '('}, {0x27, ')'}, {0x28, '\n'}, {0x2c, ' '}, {0x2d, '_'}, {0x2e, '+'},
        {0x2f, '{'}, {0x30, '}'}, {0x31, '|'}, {0x32, '~'}, {0x33, ':'}, {0x34, '"'}, {0x35, '~'}, {0x36, '<'}, {0x37, '>'}, {0x38, '?'}
};



class KeyInputListener {
public:

    using ResultCallback = std::function<void(const std::vector<std::string>&)>;
    using ResultCallback2 = std::function<void(const std::string&)>;

    void start() {
        if (*isRunning) {
            std::cout << "KeyInputListener is already running." << std::endl;
            return;
        }

        *isRunning = true;
        scannerThread = std::thread(&KeyInputListener::run, this);
        scannerThread.detach(); // Add this line to detach the thread
    }

    void stop() {
        if (!*isRunning) {
            std::cout << "KeyInputListener is not running." << std::endl;
            return;
        }

        *isRunning = false;
//        scannerThread.join();
    }
    int is_device_online(unsigned short vendor_id, unsigned short product_id) {
        struct hid_device_info *devs, *cur_dev;
        devs = hid_enumerate(vendor_id, product_id);
        cur_dev = devs;
        int is_online = 0;

        while (cur_dev) {
            if (cur_dev->vendor_id == vendor_id && cur_dev->product_id == product_id) {
                is_online = 1;
                break;
            }
            cur_dev = cur_dev->next;
        }

//        hid_free_enumeration(devs);
        return is_online;
    }



    void startIfNotRunning() {
        thread_checker = std::thread([this]() {
            while (true) {
                bool shouldStart = false;
                {
                    std::lock_guard<std::mutex> lock(mtx); // 使用互斥锁保护共享变量
                    if (!*isRunning) {
                        shouldStart = true;
                    }
                }
                if (shouldStart) {
                    start();
                }
                if(!is_device_online(VENDOR_ID, PRODUCT_ID)){
                    std::lock_guard<std::mutex> lock(mtx); // 使用互斥锁保护共享变量
                    *isRunning = false;
                }
                std::this_thread::sleep_for(std::chrono::seconds(5)); // 每次检查间隔5秒
            }
        });
    }


    void stopChecker() {
        if (thread_checker.joinable()) {
            thread_checker.join();
        }
    }

    std::string scannerDevicePath="/dev/barscanner";
    bool *isRunning;
    ResultCallback resultCallback; // 添加回调函数成员变量
    ResultCallback2 resultCallback2;
    KeyInputListener(const ResultCallback& callback,const ResultCallback2& callback2,bool * isrunning) : isRunning(isrunning), resultCallback(callback),resultCallback2(callback2) {}

private:

    std::thread scannerThread;
    std::thread thread_checker;
    std::mutex mtx; // 定义互斥锁对象

    void run() {
        int res;
        unsigned char buf[256];
        wchar_t wstr[MAX_STR];
        hid_device *handle;

        // 初始化hidapi库
        res = hid_init();
        if (res < 0) {
            std::cout << "无法初始化hidapi库" << std::endl;
            *isRunning = false;
            return;
        }

        // 打开设备
        handle = hid_open(VENDOR_ID, PRODUCT_ID, NULL);
        // handle =hid_open_path(scannerDevicePath.c_str());
        if (handle == NULL) {
            std::cout << "无法打开Scanner设备" << std::endl;
            hid_exit();
            *isRunning = false;
            return;
        }

        std::string result;
        std::vector<std::string> results;

        while (*isRunning) {
            // if (access(scannerDevicePath.c_str(), F_OK) == -1) {
            //     std::cout << "Scanner device disconnected." << std::endl;
            //     *isRunning = false;
            //     break;
            // }
            // 读取输入报告
            res = hid_read(handle, buf, sizeof(buf));
            if (res > 0) {
                // 读取成功
                if (res == 8 && buf[2] != 0) {
                    char c;
                    if (buf[0] == 0x02) {
                        c = hidWithShiftTable[buf[2]];
                    } else {
                        c = hidNoShiftTable[buf[2]];
                    }

                    if (c == '\n') {
                        if(results.size()>30){
                            results.erase(results.begin());
                        }
                        if (result.starts_with("{") && result.ends_with("}")) {
                            if (resultCallback2) {
                                resultCallback2(result);
                            }
                        }else {
                            results.emplace_back(result);
                            if (resultCallback) {
                                resultCallback(results);
                            }
                        }
                        result="";
//                        break;
                    }else{
                        result += c;
                    }
                }
            } else if (res < 0) {
                std::cout << "读取失败: " << (char*)hid_error(handle) << std::endl;
                // break;
            }
        }

        // 关闭设备
        hid_close(handle);

        // 退出hidapi库
        hid_exit();
        *isRunning = false;
        // std::cout << "Received input: " << result << std::endl;
    }
};

#endif //IMMVISION_SCANER_H
