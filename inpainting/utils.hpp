#ifndef utils_hpp
#define utils_hpp

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <assert.h>
#include <stdio.h>

#include <algorithm>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>

typedef std::vector<std::vector<cv::Point>> contours_t;
typedef std::vector<cv::Vec4i> hierarchy_t;
typedef std::vector<cv::Point> contour_t;


// Patch raduius
#define RADIUS 4
// The maximum number of pixels around a specified point on the target outline
#define BORDER_RADIUS 4

int mod(int a, int b);

void loadInpaintingImages(
                          const std::string& colorFilename,
                          const std::string& maskFilename,
                          cv::Mat& colorMat,
                          cv::Mat& maskMat,
                          cv::Mat& grayMat);

void showMat(const cv::String& winname, const cv::Mat& mat, int time=5);

void getContours(const cv::Mat& mask, contours_t& contours, hierarchy_t& hierarchy);

double computeConfidence(const cv::Mat& confidencePatch);

cv::Mat getPatch(const cv::Mat& image, const cv::Point& p);

void getDerivatives(const cv::Mat& grayMat, cv::Mat& dx, cv::Mat& dy);

cv::Point2f getNormal(const contour_t& contour, const cv::Point& point);

void computePriority(const contours_t& contours, const cv::Mat& grayMat, const cv::Mat& confidenceMat, cv::Mat& priorityMat);

void transferPatch(const cv::Point& psiHatQ, const cv::Point& psiHatP, cv::Mat& mat, const cv::Mat& maskMat);

cv::Mat computeSSD(const cv::Mat& tmplate, const cv::Mat& source, const cv::Mat& tmplateMask);

#endif