macro(fetch_hello_imgui)
    # Download hello_imgui if not available as a target
    if(NOT TARGET hello_imgui)
        include(FetchContent)
        message("hello_imgui could not be located in the CMake module search path. Downloading it from Git and building it locally")
        set(HELLOIMGUI_DOWNLOAD_FREETYPE_IF_NEEDED 1 CACHE BOOL "Download FreeType if needed")
        FetchContent_Declare(
            hello_imgui
            GIT_REPOSITORY https://github.com/neophack/hello_imgui.git
            # Enter the desired git tag below
            # GIT_TAG 0.8.0
        )
        FetchContent_MakeAvailable(hello_imgui)

        # Make cmake function `hello_imgui_add_app` available
        list(APPEND CMAKE_MODULE_PATH ${HELLOIMGUI_CMAKE_PATH})
        include(hello_imgui_add_app)
    endif()
endmacro()
