#include "utils.hpp"

// utility functions needed for inpainting

// TODO: use perror and other C functions pls

// load the color image and the mask image
void loadInpaintingImages(
                         const std::string& colorFilename,
                         const std::string& maskFilename,
                         cv::Mat& colorMat, cv::Mat& maskMat)
{
    colorMat = cv::imread(colorFilename, 1);
    maskMat = cv::imread(maskFilename, 0);
    
    // TODO: convert to C assertions
    CV_Assert(colorMat.size() == maskMat.size());
    CV_Assert(!colorMat.empty() && !maskMat.empty());
    
    // convert colorMat to depth CV_F32;
    colorMat.convertTo(colorMat, CV_32F);
    colorMat = colorMat / 255.0f;
}


// show a mat object quickly
void showMat(const cv::String &winname, const cv::Mat& mat)
{
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
    // get the boundary from the mask using findCountours
    cv::findContours(mask.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
}


// get a patch of size radius around patchCenter in Mat
cv::Mat getPatch(const cv::Mat& mat, const cv::Point& p)
{
    CV_Assert(0 <= p.x && p.x <= mat.cols && 0 <= p.y && p.y <= mat.rows);
    
    return mat(
                 cv::Range(std::max(0, p.y-radius), std::min(mat.rows, p.y+radius)),
                 cv::Range(std::max(0, p.x-radius), std::min(mat.cols, p.x+radius))
                 );
}


// get the x and y derivatives of a patch centered at patchCenter in image
// computed using a 3x3 Scharr filter
// TODO better to update than compute at runtime
struct gradients getDerivatives(const cv::Mat& grayMat, const cv::Point& p)
{
    // TODO: is this the correct way? is this the most efficient?
    
    cv::Mat patch = grayMat(
        cv::Range(
                  std::max(0, p.y-radius-1),
                  std::min(grayMat.rows, p.y+radius+1)
                  ),
        cv::Range(
                  std::max(0, p.x-radius-1),
                  std::min(grayMat.cols, p.x+radius+1)
                  )
    );
    
    struct gradients g;
    
    cv::Mat dx, dy;
    cv::Sobel(patch, dx, -1, 1, 0, -1);
    cv::Sobel(patch, dy, -1, 0, 1, -1);
    
    g.dx = dx(cv::Rect(1, 1, dx.rows-2, dx.cols-2));
    g.dy = dy(cv::Rect(1, 1, dy.rows-2, dy.cols-2));
    
    return g;
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
        
        i = (++i % contour.size());
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
template <typename T>
cv::Point getMaxPosition(cv::Mat& mat) {
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

// go over contours and compute the priority of each patch
void computePriority(const contours_t& contours, const cv::Mat grayMat, const cv::Mat confidenceMat, cv::Mat priorityMat)
{
    // for each point in contour
    cv::Point point;
    cv::Mat confidencePatch;
    double confidence;
    cv::Vec2d normal;
    cv::Mat magnitude;
    cv::Point maxPoint;
    double priority;
    
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
            gradients g = getDerivatives(grayMat, point);
            cv::magnitude(g.dx, g.dy, magnitude);
            cv::bitwise_and(magnitude, confidencePatch, magnitude);
            std::cerr << "magnitude depth is " << magnitude.depth() << std::endl;
            std::cerr << "magnitude type is " << magnitude.type() << std::endl;
            // TODO: double check size and type of magnitude, as this could lead to issues
            maxPoint = getMaxPosition<double>(magnitude);
            
            // compute the priority
            priority = confidence * std::abs(normal[0] * g.dy.ptr<double>(maxPoint.y)[maxPoint.x] - normal[1] * g.dx.ptr<double>(maxPoint.y)[maxPoint.x] )/ 255.0;
            
            // set the priority in priorityMat
            priorityMat.ptr<double>(point.y)[point.x] = priority;
        }
    }
}