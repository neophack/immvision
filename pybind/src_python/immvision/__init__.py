print("In init immvision (empty) V2")
import imgui
import cv2
from .cpp_immvision import __doc__, __version__
from .cpp_immvision import Image, ImageNavigator
from .cpp_immvision import InitGlProvider, ResetGlProvider
from .cpp_immvision import ImageParams, ColorAdjustmentsValues


# <autogen:ImageParams.docstring> // Autogenerated code below! Do not edit!

ImageParams.__doc__ = """Set of display parameters and options for an ImageNavigator


 ImageParams store the parameters for an ImageNavigator
 (as well as user selected watched pixels, selected channel, etc.)
 Its default constructor will give them reasonable choices, which you can adapt to your needs.
 Its values will be updated when the user pans or zooms the image, adds watched pixels, etc.

 Display size and title
    * image_display_size: Size = (0, 0)
            Size of the navigator (can be different from the image size)
    * legend: str = "Image Navigator"
            Title displayed in the border

 Zoom and Pan (represented by an affine transform matrix, of size 3x3)
    * zoom_pan_matrix: Matx33d = np.eye(3)
            ZoomPanMatrix can be created using MakeZoomPanMatrix to create a view centered around a given point
    * zoom_key: str = ""
            If displaying several navigators, those with the same ZoomKey will zoom and pan together

 Color adjustments
    * color_adjustments: ColorAdjustmentsValues = ColorAdjustmentsValues()
            Color adjustments for float matrixes
    * color_adjustments_key: str = ""
            If displaying several navigators, those with the same ColorAdjustmentsKey will adjust together

 Zoom and pan with the mouse
    * pan_with_mouse: bool = True
    * zoom_with_mouse_wheel: bool = True
    * is_color_order_bgr: bool = True
            Color Order: RGB or RGBA versus BGR or BGRA (Note: by default OpenCV uses BGR and BGRA)

 Image display options
    * selected_channel: int = -1
            if SelectedChannel >= 0 then only this channel is displayed
    * show_alpha_channel_checkerboard: bool = True
            show a checkerboard behind transparent portions of 4 channels RGBA images
    * show_grid: bool = True
            Grid displayed when the zoom is high
    * draw_values_on_zoomed_pixels: bool = True
            Pixel values show when the zoom is high

 Navigator display options
    * show_image_info: bool = True
            Show matrix type and size
    * show_pixel_info: bool = True
            Show pixel values
    * show_zoom_buttons: bool = True
            Show buttons that enable to zoom in/out (the mouse wheel also zoom)
    * show_legend_border: bool = True
            Show a rectangular border with the legend
    * show_options: bool = False
            Open the options panel
    * show_options_in_tooltip: bool = False
            If set to True, then the option panel will be displayed in a transient tooltip window

 Watched Pixels
    * watched_pixels: list[Point] = list[Point]()
            List of Watched Pixel coordinates
    * highlight_watched_pixels: bool = True
            Shall the watched pixels be drawn on the image
"""
# </autogen:ImageParams.docstring> // Autogenerated code end


# <autogen:ColorAdjustmentsValues.docstring> // Autogenerated code below! Do not edit!

ColorAdjustmentsValues.__doc__ = """Color adjustments (esp. useful for a float matrix)

    * factor: float = 1.
            Pre-multiply values by a Factor before displaying
    * delta: float = 0.
            Add a delta to the values before displaying
"""
# </autogen:ColorAdjustmentsValues.docstring> // Autogenerated code end

