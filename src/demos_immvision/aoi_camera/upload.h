//
// Created by nn on 4/9/24.
//

#ifndef IMMVISION_UPLOAD_H
#define IMMVISION_UPLOAD_H
#include <iostream>
#include <curl/curl.h>
#include <string>
#include <nlohmann/json.hpp>
#include "appstate.h"

using json = nlohmann::json;

struct UploadRecord {
    int success=-1;
    std::string code;
    std::string response;
};

json gen_json(AppState& appState, int ind) {
    std::string code = appState.scanResVec[ind];
    int pass_status = std::max(appState.pass_status[ind], appState.angle_pass_status[ind]);
    std::vector<std::string> words = {"CHECK", "PASS", "NG", "MISS"};

    // 构建测试记录的JSON对象
    json jsonData;
    jsonData["Code"] = code;
    jsonData["MO"] = appState.loginInfo.mo;
    jsonData["Line"] = appState.loginInfo.line;
    jsonData["Station"] = "Pin针检查";
    jsonData["Workshop"] = "Pin针检查";
    jsonData["Model"] = "Pin针检查";

    // 构建测试详细记录的JSON数组
//    json detailArray = json::array();
    std::string detailString = "Item=Code,MinValue=,Value=" + code + ",MaxValue=,Result=" + words[pass_status] + ";";
    detailString += "Item=Pin针检查,MinValue=,Value=" + words[pass_status] + ",MaxValue=,Result=" + words[pass_status] + ";";
//    detailArray.push_back(detailString);
    jsonData["Detail"] = detailString;

    jsonData["Result"] = words[pass_status];
    jsonData["User"] = appState.loginInfo.user;

    return jsonData;
}
// 定义回调函数，用于处理返回的数据
size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* response) {
    size_t totalSize = size * nmemb;
    response->append((char*)contents, totalSize);
    return totalSize;
}

void upload(AppState& appState, int ind, UploadRecord& record) {
    CURL *curl;
    CURLcode res;

    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();
    record.code = appState.scanResVec[ind];
    record.success = 2;
    if (appState.loginInfo.url.empty()) {
        record.success = 0;
    } else if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, appState.loginInfo.url.c_str());
        // 设置超时时间为10秒
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);

        struct curl_slist *headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

        std::string jsonPayload = gen_json(appState, ind).dump();
        std::cout << jsonPayload << std::endl;
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonPayload.c_str());

        // 设置回调函数和数据
        std::string responseData;  // 用于保存返回的内容
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &responseData);

        res = curl_easy_perform(curl);

        if (res != CURLE_OK) {
            record.success = 0;
            record.response = curl_easy_strerror(res);
        } else {
            long response_code;
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);

            if (response_code == 200) {
                try {
                    nlohmann::json data = nlohmann::json::parse(responseData);
                    if (data.count("Msg") > 0 && data.count("Code") > 0) {
                        if(data["Code"]==1){
                            record.success = 1;
                        }else{
                            record.success = 0;
                        }
                        record.response = "server:"+data["Msg"].dump();
                    }else{
                        record.success = 0;
                    }

                    // 在此处处理解析后的 JSON 数据
                } catch (const std::exception& e) {
                    // 在此处处理异常
                    std::cout << "解析 JSON 时发生异常: " << e.what() << std::endl;
                }


            } else {
                record.success = 0;
                record.response = std::to_string(response_code);
            }
        }

        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    } else {
        record.success = 0;
    }

    curl_global_cleanup();
}


#endif //IMMVISION_UPLOAD_H
