#include "utils.hpp"

// utility functions needed for inpainting

// TODO: use perror and other C functions pls

// TODO: fix utility algorihtms for the case where patch isn't square!

// return a % b correctly for negative numbers
int mod(int a, int b) {
    return ((a % b) + b) % b;
}


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
    
    // convert colorMat to depth CV_32F;
    colorMat.convertTo(colorMat, CV_32F);
    colorMat = colorMat / 255.0f;
    
    cv::cvtColor(colorMat, grayMat, CV_BGR2GRAY);
    cv::cvtColor(colorMat, cieMat, CV_BGR2Lab);
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
    
    cv::Sobel(grayMat, dx, -1, 1, 0, -1);
    cv::Sobel(grayMat, dy, -1, 0, 1, -1);
}


// get the normal of a dense list of boundary point centered around point p
cv::Vec2f getNormal(const contour_t& contour, const cv::Point& point)
{
    int sz = (int) contour.size();
    
    CV_Assert(sz != 0);
    int pointIndex = (int) (std::find(contour.begin(), contour.end(), point) - contour.begin());
    
    CV_Assert(pointIndex != contour.size());
    
    // create X and Y mat to SVD
    cv::Mat X(cv::Size(2, 2*border_radius+1), CV_32F);
    cv::Mat Y(cv::Size(1, 2*border_radius+1), CV_32F);
    
    CV_Assert(X.rows == Y.rows && X.cols == 2 && Y.cols == 1 && X.type() == Y.type()
              && Y.type() == CV_32F);
    
    int i = mod((pointIndex - border_radius), sz);
    
    float *Xrow;
    float *Yrow;
    
    int count = 0;
    while (count < 2*border_radius+1)
    {
        Xrow = X.ptr<float>(count);
        Xrow[0] = contour[i].x;
        Xrow[1] = 1.0f;
        
        Yrow = Y.ptr<float>(count);
        Yrow[0] = contour[i].y;
        
        i = mod(i+1, sz);
        ++count;
    }
    
    // you have the needed points in contourWindow, now you perform least Squares
    // to find the line of best fit
    cv::Mat sol;
    cv::solve(X, Y, sol, cv::DECOMP_SVD);
    
    CV_Assert(sol.type() == CV_32F);
    
    float slope = sol.ptr<float>(0)[0];
    cv::Vec2f normal(-slope, 1);
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
            x = (int) (rowMax - row);
        }
    }
    
    double testmin, testmax;
    cv::minMaxLoc(mat, &testmin, &testmax);
    CV_Assert(max == testmax);
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
    CV_Assert(grayMat.type() == CV_32FC1 &&
              priorityMat.type() == CV_32FC1 &&
              confidenceMat.type() == CV_32FC1
              );
    
    // define some patches
    cv::Mat confidencePatch;
    cv::Mat magnitudePatch;
    
    cv::Vec2f normal;
    cv::Point maxPoint;
    
    double confidence;
    double priority;
    
    // get the derivatives and magnitude of the greyscale image
    cv::Mat dx, dy, magnitude;
    getDerivatives(grayMat, dx, dy);
    magnitude = cv::abs(dx) + cv::abs(dy);
    
    // mask the magnitude
    cv::Mat maskedMagnitude(magnitude.size(), magnitude.type(), cv::Scalar(0));
    magnitude.copyTo(maskedMagnitude, (confidenceMat != 0.0f));
    
    CV_Assert(maskedMagnitude.type() == CV_32FC1);
    
    // for each point in contour
    cv::Point point;
    
    // TODO: delete
    cv::namedWindow("contour (green = normal) (red = gradient) (blue = bounding box)");
    cv::Mat contourMat(grayMat.size(), CV_8UC1, cv::Scalar(0));
    cv::drawContours(contourMat, contours, -1, cv::Scalar(255, 255, 255));
    // end delete
    
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
            magnitudePatch = getPatch(maskedMagnitude, point);
            maxPoint = getMaxPosition<float>(magnitudePatch);
            
            // TODO: delete
            // print information
            std::cerr << "point: " << point << std::endl;
            std::cerr << "confidence of patch is: " << confidence << std::endl;
            std::cerr << "normal is: " << normal << std::endl;
            std::cerr << "maximum gradient at point: " << maxPoint << std::endl;
            std::cerr << "gradient is: [" << getPatch(dy, point).ptr<float>(maxPoint.y)[maxPoint.x] << ", "
            << -getPatch(dx, point).ptr<float>(maxPoint.y)[maxPoint.x] << ")" << std::endl;
            
            // draw the contour with debugging information
            cv::Mat drawMat = contourMat.clone();
            cv::rectangle(drawMat, cv::Point(std::max(0, point.x-radius), std::max(0, point.y-radius)),
                          cv::Point(std::min(drawMat.cols, point.x+radius), std::min(drawMat.rows, point.y+radius)), cv::Scalar(255, 0, 0));
            cv::arrowedLine(drawMat, point, point + cv::Point(normal[0], normal[1]), cv::Scalar(0, 255, 0));
            // normalize gradient
            cv::Vec2f normalized_gradient;
            cv::normalize(cv::Vec2f(getPatch(dy, point).ptr<float>(maxPoint.y)[maxPoint.x], -getPatch(dx, point).ptr<float>(maxPoint.y)[maxPoint.x]), normalized_gradient);
            cv::arrowedLine(drawMat, point, point + cv::Point(normalized_gradient[0], normalized_gradient[1]), cv::Scalar(0, 0, 255));
            // show drawMat
            cv::imshow("contour (green = normal) (red = gradient) (blue = bounding box)", drawMat);
            cv::waitKey(0);
            // end delete
            
            priority = confidence * std::abs(normal[0] * getPatch(dy, point).ptr<float>(maxPoint.y)[maxPoint.x] - normal[1] * getPatch(dx, point).ptr<float>(maxPoint.y)[maxPoint.x])/ 255.0;
            
            // set the priority in priorityMat
            priorityMat.ptr<float>(point.y)[point.x] = (float) priority;
        }
    }
}


// get point of patch with minimum euclidean distance in source to a given contour centered patch
 cv::Point getClosestPatchPoint(const cv::Mat& imageMat,
                                const cv::Mat& psiHatP,
                                const cv::Mat& mask)
{
 
    cv::Mat patch;                  // temporary patch from imageMat
    cv::Mat localMask;              // local mask used for norm calculaiton
    double localNorm;               // temporary norm in loop
    double minNorm = 0;             // globally minNorm
    int patchX = 0, patchY = 0;     // result variables
 
    for (int y = 0; y < imageMat.rows; ++y)
    {
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

