#include "immvision/internal/drawing/image_drawing.h"
#include "immvision/internal/cv/colormap.h"
#include "immvision/internal/cv/zoom_pan_transform.h"
#include "immvision/internal/cv/cv_drawing_utils.h"
#include "immvision/internal/cv/matrix_info_utils.h"
#include "immvision/internal/cv/colormap.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


namespace ImmVision
{
    namespace ImageDrawing
    {
        cv::Mat DrawWatchedPixels(const cv::Mat& image, const ImageParams& params)
        {
            cv::Mat r = image.clone();

            std::vector<std::pair<size_t, cv::Point2d>> visiblePixels;
            {
                for (size_t i = 0; i < params.WatchedPixels.size(); ++i)
                {
                    cv::Point w = params.WatchedPixels[i];
                    cv::Point2d p = ZoomPanTransform::Apply(params.ZoomPanMatrix, w);
                    if (cv::Rect(cv::Point(0, 0), params.ImageDisplaySize).contains(p))
                        visiblePixels.push_back({i, p});
                }
            }

            for (const auto& kv : visiblePixels)
            {
                CvDrawingUtils::draw_named_feature(
                    r,         // img
                    kv.second, // position,
                    std::to_string(kv.first),       // name
                    cv::Scalar(255, 255, 255, 255), // color
                    true, // add_cartouche
                    4.,   // size
                    2.5,  // size_hole
                    1     // thickness
                );
            }

            return r;
        }

        void DrawGrid(cv::Mat& inOutImageRgba, const ImageParams& params)
        {
            double x_spacing = (double) params.ZoomPanMatrix(0, 0);
            double y_spacing = (double) params.ZoomPanMatrix(1, 1);

            double x_start, y_start;
            {
                cv::Point2d origin_unzoomed = ZoomPanTransform::Apply(params.ZoomPanMatrix.inv(), cv::Point2d(0., 0.));
                origin_unzoomed = cv::Point2d(std::floor(origin_unzoomed.x) + 0.5, std::floor(origin_unzoomed.y) + 0.5);
                cv::Point2d origin_zoomed = ZoomPanTransform::Apply(params.ZoomPanMatrix, origin_unzoomed);
                x_start = origin_zoomed.x;
                y_start = origin_zoomed.y;
            }
            double x_end = (double)inOutImageRgba.cols - 1.;
            double y_end = (double)inOutImageRgba.rows - 1.;

            auto lineColor = cv::Scalar(255, 255, 0, 255);
            double alpha = 0.3;
            CvDrawingUtils::draw_grid(inOutImageRgba, lineColor, alpha, x_spacing, y_spacing, x_start, y_start, x_end, y_end);
        }

        cv::Mat DrawValuesOnZoomedPixels(const cv::Mat& drawingImage, const cv::Mat& valuesImage,
                                         const ImageParams& params, bool drawPixelCoords)
        {
            assert(drawingImage.type() == CV_8UC4);

            cv::Mat r = drawingImage;
            cv::Point tl, br;
            {
                cv::Point2d tld = ZoomPanTransform::Apply(params.ZoomPanMatrix.inv(), cv::Point2d(0., 0.));
                cv::Point2d brd = ZoomPanTransform::Apply(params.ZoomPanMatrix.inv(),
                                                          cv::Point2d((double)params.ImageDisplaySize.width,
                                                                      (double)params.ImageDisplaySize.height));
                tl = { (int)std::floor(tld.x), (int)std::floor(tld.y) };
                br = { (int)std::ceil(brd.x), (int)std::ceil(brd.y) };
            }

            for (int x = tl.x; x <= br.x; x+= 1)
            {
                for (int y = tl.y; y <= br.y; y+= 1)
                {
                    std::string pixelInfo = MatrixInfoUtils::MatPixelColorInfo(valuesImage, x, y, '\n', false);
                    if (drawPixelCoords)
                        pixelInfo = std::string("x:") + std::to_string(x) + "\n" + "y:" + std::to_string(y) + "\n" + pixelInfo;

                    cv::Point2d position = ZoomPanTransform::Apply(params.ZoomPanMatrix, cv::Point2d((double)x, (double )y));

                    cv::Scalar textColor;
                    {
                        cv::Scalar white(255, 255, 255, 255);
                        cv::Scalar black(0, 0, 0, 255);
                        cv::Vec4b backgroundColor(0, 0, 0, 0);
                        if ( cv::Rect(cv::Point(), drawingImage.size()).contains({(int)position.x, (int)position.y}))
                            backgroundColor = drawingImage.at<cv::Vec4b>((int)position.y, (int)position.x);
                        double luminance = backgroundColor[2] * 0.2126 + backgroundColor[1] * 0.7152 + backgroundColor[0] * 0.0722;
                        if (luminance > 170.)
                            textColor = black;
                        else
                            textColor = white;
                    }
                    CvDrawingUtils::text(
                        r,
                        position,
                        pixelInfo,
                        textColor,
                        true, // center_around_point
                        false, // add_cartouche
                        0.3,  //fontScale
                        1     //int thickness
                    );
                }
            }
            return r;
        };


        cv::Mat MakeSchoolPaperBackground(cv::Size s)
        {
            cv::Mat mat(s, CV_8UC4);

            auto paperColor = cv::Scalar(205, 215, 220, 255);
            auto lineColor = cv::Scalar(199, 196, 184, 255);
            mat = paperColor;
            int quadSize = 17;
            for (int y = 0; y < s.height; y+= quadSize)
            {
                auto linePtr = mat.ptr<cv::Vec4b>(y);
                for (int x = 0; x < s.width; ++x)
                {
                    *linePtr = lineColor;
                    linePtr++;
                }
            }
            for (int y = 0; y < s.height; y++)
            {
                auto linePtr = mat.ptr<cv::Vec4b>(y);
                for (int x = 0; x < s.width; x+=quadSize)
                {
                    *linePtr = lineColor;
                    linePtr += quadSize;
                }
            }
            return mat;
        }

        void BlitImageTexture(
            const ImageParams& params,
            const cv::Mat& image,
            cv::Mat& in_out_rgba_image_cache,
            bool shall_refresh_rgba,
            GlTextureCv* outTexture
        )
        {
            if (image.empty())
                return;

            cv::Mat finalImage = image.clone();

            //
            // Adjustements needed before conversion to rgba
            //
            auto fnSelectChannel = [&finalImage, params]()
            {
                // Selected channels
                if (finalImage.channels() > 1 && (params.SelectedChannel >= 0) && (params.SelectedChannel < finalImage.channels()))
                {
                    std::vector<cv::Mat> channels;
                    cv::split(finalImage, channels);
                    finalImage = channels[(size_t)params.SelectedChannel];
                }

            };
            auto fnAlphaCheckerboard = [&finalImage, params]()
            {
                if ((finalImage.type() == CV_8UC4) && params.ShowAlphaChannelCheckerboard)
                {
                    cv::Mat background = CvDrawingUtils::make_alpha_channel_checkerboard_image(finalImage.size());
                    finalImage = CvDrawingUtils::overlay_alpha_image_precise(background, finalImage, 1.);
                }
            };
            auto fnMakeBackground = [&params]() -> cv::Mat
            {
                if (params.ShowSchoolPaperBackground)
                    return MakeSchoolPaperBackground(params.ImageDisplaySize);
                else
                {
                    cv::Mat m(params.ImageDisplaySize, CV_8UC4);
                    m = cv::Scalar(0, 0, 0, 255);
                    return m;
                }

            };

            //
            // Convert to rgba with adjustments if needed
            //
            if (shall_refresh_rgba)
            {
                if (HasColormapParam(params) && Colormap::CanColormap(image))
                    finalImage = Colormap::ApplyColormap(finalImage, params.ColormapSettings); // returns a rgba image
                else
                {
                    fnSelectChannel();
                    fnAlphaCheckerboard();
                    finalImage = CvDrawingUtils::converted_to_rgba_image(finalImage, params.IsColorOrderBGR);
                }
                in_out_rgba_image_cache = finalImage;
                assert(finalImage.type() == CV_8UC4);
            }
            else
            {
                finalImage = in_out_rgba_image_cache;
                assert(finalImage.type() == CV_8UC4);
                assert(!finalImage.empty());
            }

            //
            // Zoom
            //
            {
                cv::Mat backgroundWithImage = fnMakeBackground();
                cv::warpAffine(finalImage, backgroundWithImage,
                               ZoomPanTransform::ZoomMatrixToM23(params.ZoomPanMatrix),
                               params.ImageDisplaySize,
                               cv::INTER_NEAREST,
                               cv::BorderTypes::BORDER_TRANSPARENT,
                               cv::Scalar(127, 127, 127, 127)
                );
                finalImage = backgroundWithImage;
            }

            //
            // Drawings on final image
            //
            {
                // Draw grid
                double gridMinZoomFactor = 12.;
                double zoomFactor = (double)params.ZoomPanMatrix(0, 0);
                if (params.ShowGrid && zoomFactor >= gridMinZoomFactor)
                    DrawGrid(finalImage, params);

                // Draw Pixel Values
                double drawPixelvaluesMinZoomFactor = (image.depth() == CV_8U) ? 36. : 48.;
                if (params.DrawValuesOnZoomedPixels && zoomFactor > drawPixelvaluesMinZoomFactor)
                {
                    double drawPixelCoordsMinZoomFactor = 60.;
                    bool drawPixelCoords = zoomFactor > drawPixelCoordsMinZoomFactor;
                    finalImage = DrawValuesOnZoomedPixels(finalImage, image, params, drawPixelCoords);
                }

                // Draw Watched Pixels
                if (params.HighlightWatchedPixels && (! params.WatchedPixels.empty()))
                    finalImage = DrawWatchedPixels(finalImage, params);

            }

            //
            // Blit
            //
            outTexture->BlitMat(finalImage, false);
        }

        bool HasColormapParam(const ImageParams &params)
        {
            return (!params.ColormapSettings.Colormap.empty() || !params.ColormapSettings.internal_ColormapHovered.empty());
        }

        cv::Mat _DrawLutGraph(const std::vector<double>& x, const std::vector<double>& y, cv::Size size)
        {
            cv::Mat image(size, CV_8UC4);
            assert(x.size() == y.size());
            size_t len = x.size();

            auto toPoint = [size](double x, double y) -> cv::Point2d {
                cv::Point2d r {
                    1. + x * (double)(size.width - 3),
                    1. + (1. - y) * (double)(size.height - 3),
                };
                return r;
            };

            image = cv::Scalar(200, 200, 200, 0);
            // cv::Scalar color(255, 80, 50, 255);
            cv::Scalar color(0, 255, 255, 255);
            for (size_t i = 0; i < len - 1; ++i)
            {
                double x0 = x[i], y0 = y[i];
                double x1 = x[i + 1], y1 = y[i + 1];
                CvDrawingUtils::line(image, toPoint(x0, y0), toPoint(x1, y1), color);
            }
            return image;
        }

    } // namespace ImageDrawing

} // namespace ImmVision
