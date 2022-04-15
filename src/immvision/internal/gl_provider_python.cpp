#ifdef IMMVISION_BUILDING_PYBIND

#include "immvision/internal/gl_provider.h"
#include "immvision/image.h"
#include "immvision/internal/opencv_pybind_converter.h"

#include <opencv2/core.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <memory>

namespace
{
    struct PythonFunctions
    {
        PythonFunctions()
        {
            PythonModule = pybind11::module_::import("gl_provider_python");
            Blit_RGBA_Buffer = PythonModule.attr("Blit_RGBA_Buffer");
            GenTexture = PythonModule.attr("GenTexture");
            DeleteTexture = PythonModule.attr("DeleteTexture");

            Image = PythonModule.attr("image");
            ImageButton = PythonModule.attr("image_button");
            GetWindowDrawList_AddImage = PythonModule.attr("get_window_draw_list_add_image");
        }
        pybind11::object PythonModule;
        pybind11::object Blit_RGBA_Buffer;
        pybind11::object GenTexture;
        pybind11::object DeleteTexture;
        pybind11::object Image;
        pybind11::object ImageButton;
        pybind11::object GetWindowDrawList_AddImage;
    };

    std::unique_ptr<PythonFunctions> gPythonFunctions; // = std::make_unique<PythonFunctions>();
}

namespace ImmVision_GlProvider
{

    void InitGlProvider()
    {
        // InitGlProvider must be called after the OpenGl Loader is initialized
        if (!gPythonFunctions)
            gPythonFunctions = std::make_unique<PythonFunctions>();
    }

    void ResetGlProvider()
    {
        // InitGlProvider must be called before the OpenGl Loader is reset
        ImmVision::ClearAllTextureCaches();
        gPythonFunctions.release();
    }


    void Blit_RGBA_Buffer(unsigned char *image_data, int image_width, int image_height, unsigned int textureId)
    {
//        std::cout << "C++ : Blit_RGBA_Buffer -> about to import gl_provider_python\n";
//        pybind11::object python_module = pybind11::module_::import("gl_provider_python");
//        (void)python_module;

        assert(gPythonFunctions);

        std::cout << "C++ : Blit_RGBA_Buffer about to create Mat\n";
        cv::Mat m(image_height, image_width, CV_8UC4, image_data);
        std::cout << "C++ : Blit_RGBA_Buffer about to call PythonFunctions::Blit_RGBA_Buffer\n";
        gPythonFunctions->Blit_RGBA_Buffer(m, textureId);
        std::cout << "C++ : Blit_RGBA_Buffer done!\n";
    }

    unsigned int GenTexture()
    {
        assert(gPythonFunctions);
        std::cout << "C++ About to call GenTexture\n";
        pybind11::object id_object = gPythonFunctions->GenTexture();
        auto texture_id = id_object.cast<unsigned int>();
        std::cout << "C++ After calling GenTexture, texture_id=%i" << texture_id << "\n";
        return  texture_id;
    }

    void DeleteTexture(unsigned int texture_id)
    {
        assert(gPythonFunctions);
        std::cout << "C++ About to call DeleteTexture\n";
        gPythonFunctions->DeleteTexture(texture_id);
        std::cout << "C++ After calling DeleteTexture\n";
    }

} // namespace ImmVision_GlProvider


namespace ImGuiImmGlImage
{
    //
    // Wrappers for ImGui::Image, ImGui::ImageButton and ImGui::GetWindowDrawList()->AddImage
    //
    // They have the same behavior under C++, but under python this is transferred to the python interpreter
    //
    // The reason for this is the strange signature of ImGui::Image and others (ImGui::Image(void* texture_id, ...)
    // The call is transferred to pyimgui (i.e imgui python bindings), but somehow texture_id seems to be
    // dereferenced in the process, which causes a seg fault.
    //
    // Room for improvement in a later step


    pybind11::tuple ImVec2_to_Tuple(ImVec2 v)
    {
        return pybind11::make_tuple(v.x, v.y);
    }
    pybind11::tuple ImVec4_to_Tuple(ImVec4 v)
    {
        return pybind11::make_tuple(v.x, v.y, v.z, v.w);
    }

    void  Image(unsigned int user_texture_id, const ImVec2& size, const ImVec2& uv0, const ImVec2& uv1, const ImVec4& tint_col, const ImVec4& border_col)
    {
        assert(gPythonFunctions);
        gPythonFunctions->Image(
            user_texture_id,
            size.x, size.y,
            ImVec2_to_Tuple(uv0), ImVec2_to_Tuple(uv1),
            ImVec4_to_Tuple(tint_col), ImVec4_to_Tuple(border_col));
    }

    bool  ImageButton(unsigned int user_texture_id, const ImVec2& size, const ImVec2& uv0,  const ImVec2& uv1, int frame_padding, const ImVec4& bg_col, const ImVec4& tint_col)
    {
        assert(gPythonFunctions);
        pybind11::object r = gPythonFunctions->ImageButton(
            user_texture_id,
            size.x, size.y,
            ImVec2_to_Tuple(uv0), ImVec2_to_Tuple(uv1),
            frame_padding,
            ImVec4_to_Tuple(bg_col),
            ImVec4_to_Tuple(tint_col));
        bool clicked = r.cast<bool>();
        return clicked;
    }

    void  GetWindowDrawList_AddImage(unsigned int user_texture_id, const ImVec2& p_min, const ImVec2& p_max, const ImVec2& uv_min, const ImVec2& uv_max, ImU32 col)
    {
        assert(gPythonFunctions);
        gPythonFunctions->GetWindowDrawList_AddImage(
            user_texture_id,
            ImVec2_to_Tuple(p_min), ImVec2_to_Tuple(p_max),
            ImVec2_to_Tuple(uv_min), ImVec2_to_Tuple(uv_max),
            col);
    }

} // namespace ImGuiImmGlImage


#endif // #ifdef IMMVISION_BUILDING_PYBIND