set(LIB_NAME SQLiteCpp)
set(LIB_REPO https://github.com/SRombauts/SQLiteCpp)

FetchContent_Declare(${LIB_NAME}
        GIT_REPOSITORY  ${LIB_REPO})
message(STATUS "Using ${LIB_NAME} as CMake FetchContent project")
FetchContent_MakeAvailable(${LIB_NAME})
#include_directories(${${LIB_NAME}_SOURCE_DIR}/include)
#message("include:" ${${LIB_NAME}_SOURCE_DIR}/include)