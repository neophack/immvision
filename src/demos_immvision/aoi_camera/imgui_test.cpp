//#include <iostream>
//#include <curl/curl.h>
//#include <string>
//#include <nlohmann/json.hpp>
//
//using json = nlohmann::json;
//
//// 回调函数，用于处理HTTP响应
//size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* response) {
//    size_t totalSize = size * nmemb;
//    response->append((char*)contents, totalSize);
//    return totalSize;
//}
//
//// 函数用于上传测试记录到MES并返回结果信息
//std::string uploadTestRecordToMES(const std::string &url,const json& jsonData) {
//    std::string response;
//
//
//
//    // 初始化CURL
//    CURL* curl = curl_easy_init();
//    if (curl) {
//        // 设置POST请求
//        curl_easy_setopt(curl, CURLOPT_POST, 1L);
//        // 设置请求的URL
//        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
//        // 设置POST数据
//        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonData.dump().c_str());
//
//        // 设置HTTP响应处理回调函数
//        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
//        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
//
//        // 发送HTTP请求
//        CURLcode res = curl_easy_perform(curl);
//        if (res != CURLE_OK) {
//            // HTTP请求失败
//            std::cout << "Failed to send HTTP request: " << curl_easy_strerror(res) << std::endl;
//        }
//
//        // 清理CURL资源
//        curl_easy_cleanup(curl);
//    } else {
//        // 初始化CURL失败
//        std::cout << "Failed to initialize CURL." << std::endl;
//    }
//
//    return response;
//}
//
//int main() {
//    // 构建测试记录的JSON对象
//    json jsonData;
//    jsonData["Code"] = "A1111111111";
//    jsonData["MO"] = "MO";
//    jsonData["Line"] = "A11";
//    jsonData["Station"] = "PIN针检查";
//    jsonData["Workshop"] = "PIN针检查";
//    jsonData["Model"] = "PIN针检查";
//
//    // 构建测试详细记录的JSON数组
//    json detailArray = json::array();
//    json detailItem1;
//    detailItem1["Item"] = "Code";
//    detailItem1["MinValue"] = "";
//    detailItem1["Value"] = "A1111111111";
//    detailItem1["MaxValue"] = "";
//    detailItem1["Result"] = "OK";
//    detailArray.push_back(detailItem1);
//    json detailItem2;
//    detailItem2["Item"] = "Pin针检查";
//    detailItem2["MinValue"] = "";
//    detailItem2["Value"] = "OK";
//    detailItem2["MaxValue"] = "";
//    detailItem2["Result"] = "OK";
//    detailArray.push_back(detailItem2);
//    jsonData["Detail"] = detailArray;
//
//    jsonData["Result"] = "PASS";
//    jsonData["User"] = "Mr";
//
//    // 设置请求的URL
//    std::string url = "http://192.168.39.250/PostATEData.ashx";
//    // 调用上传测试记录函数
//    std::string response = uploadTestRecordToMES(url,jsonData);
//
//    // 输出返回结果
//    std::cout << "Response: " << response << std::endl;
//
//    return 0;
//}
//
//#include <iostream>
//#include <string>
//#include <nlohmann/json.hpp>
//
//using json = nlohmann::json;
//
//int main() {
//    std::string jsonString = R"({"Code":"A1111111111","MO":"MO","Line":"A11","Station":"PIN","Workshop":"PIN","Model":"PIN","Result":"PASS","User":"Mr","url":"http://192.168.39.250/PostATEData.ashx"})";
//
//    // 解析JSON字符串
//    json data = json::parse(jsonString);
//
//    // 读取参数
//    std::string code = data["Code"];
//    std::string mo = data["MO"];
//    std::string line = data["Line"];
//    std::string station = data["Station"];
//    std::string workshop = data["Workshop"];
//    std::string model = data["Model"];
//    std::string result = data["Result"];
//    std::string user = data["User"];
//    std::string url = data["url"];
//
//    // 输出参数值
//    std::cout << "Code: " << code << std::endl;
//    std::cout << "MO: " << mo << std::endl;
//    std::cout << "Line: " << line << std::endl;
//    std::cout << "Station: " << station << std::endl;
//    std::cout << "Workshop: " << workshop << std::endl;
//    std::cout << "Model: " << model << std::endl;
//    std::cout << "Result: " << result << std::endl;
//    std::cout << "User: " << user << std::endl;
//    std::cout << "URL: " << url << std::endl;
//
//    return 0;
//}

#include <stdio.h>
#include <wchar.h>
#include <string>
#include <hidapi.h>
#include <map>

#define MAX_STR 255
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



std::string readStringFromDevice() {
    int res;
    unsigned char buf[256];
    wchar_t wstr[MAX_STR];
    hid_device *handle;

    // 初始化hidapi库
    res = hid_init();
    if (res < 0) {
        printf("无法初始化hidapi库\n");
        return "";
    }

    // 打开设备
//    handle = hid_open(VENDOR_ID, PRODUCT_ID, NULL);
    handle =hid_open_path("/dev/barscanner");
    if (handle == NULL) {
        printf("无法打开设备\n");
        hid_exit();
        return "";
    }

    std::string result;

    while (1) {
        // 读取输入报告
        res = hid_read(handle, buf, sizeof(buf));
        if (res > 0) {
            // 读取成功
//            printf("读取到 %d 个字节的数据\n", res);
            if(res==8 && buf[2]!=0){
                char c;
                if(buf[0]==0x02){
                    c= hidWithShiftTable[buf[2]];
                }else{
                    c= hidNoShiftTable[buf[2]];
                }
                if(c=='\n'){
                    break;
                }
                result+=c;
            }
        } else if (res < 0) {
            printf("读取失败: %ls\n", hid_error(handle));
            break;
        }
    }

    // 关闭设备
    hid_close(handle);

    // 退出hidapi库
    hid_exit();

    return result;
}

int main() {
    std::string str = readStringFromDevice();
    printf("读取到的字符串: %s\n", str.c_str());

    return 0;
}
