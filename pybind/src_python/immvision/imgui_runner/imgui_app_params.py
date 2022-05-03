import logging
from dataclasses import dataclass
from typing import Optional, List
import os
from .gui_types import Color_RGBA_Float, WindowSize, WindowPosition, WindowBounds
from . import _DEBUG_IMGUI_RUNNER


@dataclass
class ImguiAppParams:
    # Title and Size
    app_window_title: str = ""

    #
    # Application window size options
    #

    # app_window_size: Application window size:
    # if app_window_size==None and app_window_size_restore_last==False:
    #     then, the app window size will try to match the widgets size.
    app_window_size: Optional[WindowSize] = None
    # restore_last_window_size: should we restore the application window size of the last run
    # if app_window_size_restore_last==True, the app window size will not try to match the widgets size.
    app_window_size_restore_last: bool = False
    # Make the App window maximized (i.e almost full screen, except dock, taskbar, etc)
    # If app_window_size_maximized==True, then use the whole usable area on the screen
    # (i.e except the dock, start menu and taskbar)
    app_window_maximized: bool = False
    # Make the app full screen (i.e hide the taskbar, dock, etc).
    # By default, the resolution will not be changed, except if app_window_size is filled.
    # The application will be displayed full screen on the monitor given by app_window_position_monitor_index
    # (Note: there might be issues with Mac Retina displays when using Glfw backend: switch to sdl in this case)
    app_window_full_screen: bool = False

    #
    # Application window position options
    #

    # app_window_position: Application window position
    # if app_window_position==None the app window size will center on the screen
    app_window_position: Optional[WindowPosition] = None
    # restore_last_window_position: should we restore the application window position of the last run
    app_window_position_restore_last: bool = True
    # Monitor index, where the application will be shown:
    # used only if app_window_position is None and (app_window_position_restore_last==False or unknown)
    app_window_monitor_index: int = 0

    #
    # imgui options
    #

    # Provide a default full *imgui* window inside the app, i.e a background on which you can place any widget
    # if provide_default_window==False, then the application will *not* adapt it window size to its widgets.
    provide_default_window: bool = True

    #
    # Power save options
    #

    # Use power save: if this is true, you can increase manually the idle FPS by calling
    # imgui_runner.power_save.set_max_wait_before_next_frame( 1 / desired_fps)
    use_power_save: bool = True

    #
    # OpenGL options
    #

    # Anti-aliasing number of samples
    # - if the call to SDL_GL_CreateContext fails, try reducing the value to zero
    # - if you want better antialiasing, you can also increase it
    gl_multisamples = 4
    # OpenGl clear color (i.e glClearColor)
    gl_clear_color: Color_RGBA_Float = (1., 1., 1., 1)

    #
    # Flags to handle the application lifetime
    #

    # Change this bool to True when you want to exit
    app_shall_exit: bool = False
    # Change this bool to True when you want the window to be brought to the top
    # at one moment during the execution. It will be brought to top at the next frame
    # (warning, does not always work on MacOS with SDL)
    app_shall_raise_window = False
