#include "utils.hpp"

// utility functions needed for inpainting

// TODO: use perror and other C functions pls

// TODO: fix utility algorihtms for the case where patch isn't square!


// load the color image and the mask image
void loadInpaintingImages(
                          const std::string& colorFilename,
                          const std::string& maskFilename,
                          cv::Mat& colorMat,
                          cv::Mat& maskMat,
                          cv::Mat& grayMat,
                          cv::Mat& cieMat)
{
    CV_Assert(colorFilename.length() && maskFilename.length());
    colorMat = cv::imread(colorFilename, 1);
    maskMat = cv::imread(maskFilename, 0);
    
    // TODO: convert to C assertions
    CV_Assert(colorMat.size() == maskMat.size());
    CV_Assert(!colorMat.empty() && !maskMat.empty());
    
    grayMat = cv::imread(colorFilename, 0);
    cv::cvtColor(colorMat, cieMat, CV_BGR2Lab);
    
    // convert colorMat to depth CV_32F;
    colorMat.convertTo(colorMat, CV_32F);
    colorMat = colorMat / 255.0f;
}


// show a mat object quickly
void showMat(const cv::String &winname, const cv::Mat& mat)
{
    CV_Assert(!mat.empty());
    cv::namedWindow(winname);
    cv::imshow(winname, mat);
    cv::waitKey(0);
    cv::destroyWindow(winname);
}


// function to extract closed boundary of a shape given the mask
void getContours(const cv::Mat& mask,
                 contours_t& contours,
                 hierarchy_t& hierarchy
                 )
{
    CV_Assert(mask.type() == CV_8UC1);
    // get the boundary from the mask using findCountours
    cv::findContours(mask.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
}


// get a patch of size radius around patchCenter in Mat
cv::Mat getPatch(const cv::Mat& mat, const cv::Point& p)
{
    CV_Assert(0 <= p.x && p.x < mat.cols && 0 <= p.y && p.y < mat.rows);
    
    return mat(
                 cv::Range(std::max(0, p.y-radius), std::min(mat.rows, p.y+radius)),
                 cv::Range(std::max(0, p.x-radius), std::min(mat.cols, p.x+radius))
                 );
}


// get the x and y derivatives of a patch centered at patchCenter in image
// computed using a 3x3 Scharr filter
void getDerivatives(const cv::Mat& grayMat, cv::Mat& dx, cv::Mat& dy)
{
    CV_Assert(grayMat.type() == CV_32FC1);
    /*
    cv::Mat patch = grayMat(
        cv::Range(
                  std::max(0, p.y-radius),
                  std::min(grayMat.rows, p.y+radius)
                  ),
        cv::Range(
                  std::max(0, p.x-radius-1),
                  std::min(grayMat.cols, p.x+radius)
                  )
    );
     */
    
    cv::Sobel(grayMat, dx, -1, 1, 0, -1);
    cv::Sobel(grayMat, dy, -1, 0, 1, -1);
}

// get the normal of a dense list of boundary point centered around point p
cv::Vec2d getNormal(const contour_t& contour, const cv::Point& point)
{
    CV_Assert(!contour.empty());
    long pointIndex = std::find(contour.begin(), contour.end(), point) - contour.begin();
    
    CV_Assert(pointIndex != contour.size());
    
    // create X and Y mat to SVD
    cv::Mat X(cv::Size(2, 2*border_radius+1), CV_32S);
    cv::Mat Y(cv::Size(1, 2*border_radius+1), CV_32S);
    
    // TODO: delete
    CV_Assert(X.rows == 2*border_radius+1 && X.cols == 2);
    CV_Assert(Y.rows == 2*border_radius+1 && Y.cols == 1);
    
    long i = (pointIndex - border_radius) % contour.size();
    
    int *Xrow;
    int *Yrow;
    
    int count = 0;
    while (count < 2*border_radius+1)
    {
        Xrow = X.ptr<int>(count);
        Xrow[0] = contour[i].x;
        Xrow[1] = 1;
        
        Yrow = Y.ptr<int>(count);
        Yrow[0] = contour[i].y;
        
        i = ((i+1) % contour.size());
    }
    
    // you have the needed points in contourWindow, now you perform least Squares
    // to find the line of best fit
    cv::Mat sol;
    cv::solve(X, Y, sol, cv::DECOMP_SVD);
    
    double slope = sol.ptr(0)[0];
    cv::Vec2d normal(-slope, 1);
    cv::normalize(normal, normal);
    
    return normal;
}

// get the position of the maximum in a Mat
template <typename T> cv::Point getMaxPosition(cv::Mat& mat) {
    int x = 0, y = 0;
    T max = 0;
    
    for (int r = 0; r < mat.rows; ++r)
    {
        T *row = mat.ptr<T>(r);
        T *rowMax = std::max_element(row, row+mat.cols);
        if (max < *rowMax)
        {
            max = *rowMax;
            y = r;
            x = rowMax - row;
        }
    }
    return cv::Point(x, y);
}

// get the confidence
double computeConfidence(const cv::Mat& confidencePatch)
{
    return cv::sum(confidencePatch)[0] / (double) confidencePatch.total();
}

// go over contours and compute the priority of each patch
void computePriority(const contours_t& contours, const cv::Mat& grayMat, const cv::Mat& confidenceMat, cv::Mat priorityMat)
{
    CV_Assert(grayMat.type() == CV_8UC1 &&
              priorityMat.type() == CV_32FC1 &&
              confidenceMat.type() == CV_32FC1
              );
    
    // for each point in contour
    cv::Point point;
    cv::Mat confidencePatch;
    double confidence;
    cv::Mat magnitudePatch;
    cv::Mat dxPatch;
    cv::Mat dyPatch;
    cv::Vec2d normal;
    cv::Point maxPoint;
    double priority;
    
    // get the derivatives and magnitude
    cv::Mat dx, dy, magnitude;
    getDerivatives(grayMat, dx, dy);
    cv::magnitude(dx, dy, magnitude);
    // mask the magnitude
    CV_Assert((confidenceMat != 0).type() == CV_8UC1);
    double min, max;
    cv::minMaxLoc((confidenceMat !=0), &min, &max);
    CV_Assert(min == 0 && max == 255);
    cv::bitwise_and((confidenceMat != 0), magnitude, magnitude);
    
    CV_Assert(dx.type() == CV_32FC1);
    CV_Assert(magnitude.type() == CV_32FC1);
    
    for (int i = 0; i < contours.size(); ++i)
    {
        contour_t contour = contours[i];
        
        for (int j = 0; j < contour.size(); ++j)
        {
            point = contour[j];
            confidencePatch = getPatch(confidenceMat, point);
            // get confidence of patch
            confidence = cv::sum(confidencePatch)[0] / (double) confidencePatch.total();
            // get the normal to the border around point
            normal = getNormal(contour, point);
            // get the maximum gradient in source around patch
            magnitudePatch = getPatch(magnitudePatch, point);
            maxPoint = getMaxPosition<float>(magnitudePatch);
            // compute the priority
            dxPatch = getPatch(dx, point);
            dyPatch = getPatch(dy, point);
            priority = confidence * std::abs(normal[0] * dyPatch.ptr<float>(maxPoint.y)[maxPoint.x] - normal[1] * dxPatch.ptr<double>(maxPoint.y)[maxPoint.x])/ 255.0;
            
            // set the priority in priorityMat
            priorityMat.ptr<double>(point.y)[point.x] = priority;
        }
    }
}

// get poin of patch with minimum euclidean distance in source to a given contour centered patch
template <typename T> cv::Point getClosestPatchPoint(
                                                     const cv::Mat& imageMat,
                                                     const cv::Mat& psiHatP,
                                                     const cv::Mat& mask
                                              ) {
    
    cv::Mat patch;      // temporary patch from imageMat
    cv::Mat localMask;  // local mask used for norm calculaiton
    double localNorm;   // temporary norm in loop
    double minNorm;     // globally minNorm
    int patchX = 0, patchY = 0;     // result variables
    
    for (int y = 0; y < imageMat.rows; ++y)
    {
        T* row = imageMat.ptr<T>(y);
        for (int x = 0; x < imageMat.cols; ++x)
        {
            patch = getPatch(imageMat, cv::Point(x, y));
            localMask = getPatch(mask, cv::Point(x, y));
            localNorm = cv::norm(psiHatP, patch, cv::NORM_L2, localMask);
            if (x == 0 && y == 0)
            {
                minNorm = localNorm;
            }
            else
            {
                if (localNorm < minNorm)
                {
                    minNorm = localNorm;
                    patchX = x;
                    patchY = y;
                }
            }
        }
    }
    
    return cv::Point(patchX, patchY);
}