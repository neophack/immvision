set(CMAKE_CXX_STANDARD 20)

# 添加编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")

include_directories(../../../deep_perception_engine/core/include)

hello_imgui_add_app(aoi_camera aoi_camera.main.cpp )
target_link_libraries(aoi_camera PRIVATE  immvision ${OpenCV_LIBS})

hello_imgui_add_app(imguitest_camera imgui_test.cpp )
target_link_libraries(imguitest_camera PRIVATE immvision ${OpenCV_LIBS})