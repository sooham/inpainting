#ifndef utils_hpp
#define utils_hpp

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <algorithm>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>

typedef std::vector<std::vector<cv::Point>> contours_t;
typedef std::vector<cv::Vec4i> hierarchy_t;
typedef std::vector<cv::Point> contour_t;


// the radius of a patch
// results in 9x9 patches
#define radius 4


// the radius of pixels around specified point in border
// resulting in borders of size 9
#define border_radius 4

void loadInpaintingImages(
                          const std::string& colorFilename,
                          const std::string& maskFilename,
                          cv::Mat& colorMat,
                          cv::Mat& maskMat,
                          cv::Mat& grayMat,
                          cv::Mat& cieMat);

void showMat(const cv::String &winname, const cv::Mat& mat);

void getContours(const cv::Mat& mask, contours_t& contours, hierarchy_t& hierarchy);

double computeConfidence(const cv::Mat& confidencePatch);

cv::Mat getPatch(const cv::Mat& image, const cv::Point& p);

void getDerivatives(const cv::Mat& grayMat, cv::Mat& dx, cv::Mat& dy);

cv::Vec2d getNormal(const contour_t& contour, const cv::Point& point);

template <typename T> cv::Point getMaxPosition(cv::Mat& mat);

void computePriority(const contours_t& contours, const cv::Mat& grayMat, const cv::Mat& confidenceMat, cv::Mat& priorityMat);

template <typename T> cv::Point getClosestPatchPoint(
                                              const cv::Mat& imageMat,
                                              const cv::Mat& psiHatP,
                                              const cv::Mat& mask
                                              );


#endif