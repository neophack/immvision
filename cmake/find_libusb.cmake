#
#
#include(FetchContent)
#FetchContent_Declare(libusb
#        GIT_REPOSITORY "https://github.com/libusb/libusb.git"
#        GIT_TAG "v1.0.27"
#
#        UPDATE_COMMAND ${CMAKE_COMMAND} -E copy
#        ${CMAKE_SOURCE_DIR}/cmake/libusb.txt
#        <SOURCE_DIR>/CMakeLists.txt
##        UPDATE_COMMAND ${CMAKE_COMMAND} -E copy
##        ${CMAKE_SOURCE_DIR}/cmake/config.h
##        <SOURCE_DIR>
#)
#
#FetchContent_GetProperties(libusb)
#if(NOT libusb_POPULATED)
#    FetchContent_Populate(libusb)
#    add_subdirectory(${libusb_SOURCE_DIR} ${libusb_BINARY_DIR})
#endif()
#
#set(LIBUSB1_LIBRARY_DIRS ${libusb_BINARY_DIR})
#link_directories(${LIBUSB1_LIBRARY_DIRS})
#
#set(LIBUSB1_LIBRARIES usb)
##set(LIBUSB_LOCAL_INCLUDE_PATH third-party/libusb)
#
#set(USE_EXTERNAL_USB ON)
#set(LIBUSB_LOCAL_INCLUDE_PATH ${libusb_SOURCE_DIR}/libusb)

include(FetchContent)
include(GenerateExportHeader)

#getting HIDAPI deps
FetchContent_Declare(hidapi
        GIT_REPOSITORY https://github.com/libusb/hidapi.git
)

FetchContent_MakeAvailable(hidapi)
set(HIDAPI_WITH_HIDRAW TRUE)
set(HIDAPI_WITH_LIBUSB FALSE)
FetchContent_GetProperties(hidapi)
if(NOT hidapi_POPULATED)
    FetchContent_Populate(hidapi)
endif()
