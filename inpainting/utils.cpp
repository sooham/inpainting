// utility functions needed for inpainting
#include "utils.hpp"


// TODO: use perror and other C functions pls

// load the color image and the mask image
void loadInpaintingImage(
                         const std::string& colorFilename,
                         const std::string& maskFilename,
                         cv::Mat& colorMat, cv::Mat& maskMat)
{
    colorMat = cv::imread(colorFilename);
    maskMat = cv::imread(maskFilename);
    
    // TODO: convert to assertions
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
void getContours(const cv::Mat& mask, std::vector<std::vector<cv::Point>>& contours, std::vector<cv::Vec4i>& hierarchy)
{
    // get the boundary from the mask using findCountours
    cv::findContours(mask.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
}


// get a patch of size radius around patchCenter in Mat
cv::Mat getPatch(const cv::Mat image, const cv::Point& p)
{
    CV_Assert(0 <= p.x && p.x <= image.cols && 0 <= p.y && p.y <= image.rows);
    
    return image(
                 cv::Range(std::max(0, p.y-radius), std::min(image.rows, p.y+radius)),
                 cv::Range(std::max(0, p.x-radius), std::min(image.cols, p.x+radius))
                 );
}


// get the x and y derivatives of a patch centered at patchCenter in image
// computed using a 3x3 Scharr filter
struct gradients getDerivatives(cv::Mat& image, cv::Point& p)
{
    // TODO: is this the correct way? is this the most efficient?
    
    cv::Mat patch = image(
        cv::Range(
                  std::max(0, p.y-radius-1),
                  std::min(image.rows, p.y+radius+1)
                  ),
        cv::Range(
                  std::max(0, p.x-radius-1),
                  std::min(image.cols, p.x+radius+1)
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
cv::Vec2d getNormal(const std::vector<cv::Point>& contour, cv::Point& point)
{
    CV_Assert(!contour.empty());
    long pointIndex = std::find(contour.begin(), contour.end(), point) - contour.begin();
    
    CV_Assert(pointIndex != contour.size());
    
    // create X and Y mat to SVD
    cv::Mat X(cv::Size(2, 2*border_radius+1), CV_16U);
    cv::Mat Y(cv::Size(1, 2*border_radius+1), CV_16U);
    
    long i = (pointIndex - border_radius) % contour.size();
    int count = 0;
    
    uchar *Xrow;
    uchar *Yrow;
    while (count < 2*border_radius+1)
    {
        Xrow = X.ptr(count);
        Xrow[0] = contour[i].x;
        Xrow[1] = 1;
        
        Yrow = Y.ptr(count);
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

