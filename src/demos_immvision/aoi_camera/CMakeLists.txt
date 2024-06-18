set(CMAKE_CXX_STANDARD 20)

# 添加编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")

include(${CMAKE_SOURCE_DIR}/cmake/fetch_SQLiteCpp.cmake)
include_directories(${SQLiteCpp_SOURCE_DIR}/include)
include(${CMAKE_SOURCE_DIR}/cmake/fetch_json.cmake)
include_directories(${json_SOURCE_DIR}/single_include)

# 添加CURL库依赖
find_package(CURL REQUIRED)
add_subdirectory(hidapi)
include_directories(hidapi/include)

hello_imgui_add_app(aoi_camera aoi_camera.main.cpp )
target_link_libraries(aoi_camera PRIVATE  immvision ${OpenCV_LIBS} SQLiteCpp CURL::libcurl hidapi)


include(FindPkgConfig)
# pkg_search_module(LIBUSB REQUIRED libusb-1.0)
# include_directories(${LIBUSB_INCLUDE_DIRS})

pkg_check_modules(UDEV REQUIRED libudev)

# Add the include directories for libudev
include_directories(${UDEV_INCLUDE_DIRS})


hello_imgui_add_app(imguitest_camera imgui_test.cpp )
target_link_libraries(imguitest_camera PRIVATE immvision ${OpenCV_LIBS}  CURL::libcurl hidapi ${UDEV_LIBRARIES})