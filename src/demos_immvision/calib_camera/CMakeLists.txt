set(CMAKE_CXX_STANDARD 20)

# 添加编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")

hello_imgui_add_app(calib_camera calib_camera.main.cpp)
target_link_libraries(calib_camera PRIVATE immvision ${OpenCV_LIBS})
