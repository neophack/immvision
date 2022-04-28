#pragma once
#include "immvision/image_navigator.h"
#include "immvision/internal/gl/gl_texture.h"
#include <opencv2/core.hpp>

namespace ImmVision
{
    namespace ImageNavigatorDrawing
    {
        cv::Mat DrawWatchedPixels(const cv::Mat& image, const ImageParams& params);

        void DrawGrid(cv::Mat& inOutImageRgba, const ImageParams& params);

        cv::Mat DrawValuesOnZoomedPixels(const cv::Mat& drawingImage, const cv::Mat& valuesImage,
                                         const ImageParams& params, bool drawPixelCoords);

        cv::Mat MakeSchoolPaperBackground(cv::Size s);

        void BlitImageNavigatorTexture(
            const ImageParams& params,
            const cv::Mat& image,
            cv::Mat& in_out_rgba_image_cache,
            bool shall_refresh_rgba,
            GlTextureCv* outTexture
        );

    } // namespace ImageNavigatorDrawing

} // namespace ImmVision