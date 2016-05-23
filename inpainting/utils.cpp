  #include "utils.hpp"

// utility functions needed for inpainting
// TODO: algebraic notation instead

// TODO: use perror and other C functions pls

// TODO: fix utility algorihtms for the case where patch isn't square!

// return a % b correctly for negative numbers
int mod(int a, int b) {
    return ((a % b) + b) % b;
}


// load the color, mask, grayscale and CIE lab image with a border of size
// radius around every image to prevent boundary collisions when taking patches
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
    
    // add border around colorMat and maskMat
    cv::copyMakeBorder(colorMat, colorMat, radius, radius, radius, radius, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
    
    cv::cvtColor(colorMat, grayMat, CV_BGR2GRAY);
    cv::cvtColor(colorMat, cieMat, CV_BGR2Lab);
    cieMat /= 255.0f;
}


// show a mat object quickly
void showMat(const cv::String& winname, const cv::Mat& mat, int time /*= 5*/)
{
    CV_Assert(!mat.empty());
    cv::namedWindow(winname);
    cv::imshow(winname, mat);
    cv::waitKey(time);
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
// always returns a patch of size radius * 2 + 1
cv::Mat getPatch(const cv::Mat& mat, const cv::Point& p)
{
    CV_Assert(radius <= p.x && p.x < mat.cols-radius && radius <= p.y && p.y < mat.rows-radius);
    return  mat(
                 cv::Range(p.y-radius, p.y+radius+1),
                 cv::Range(p.x-radius, p.x+radius+1)
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


// get the unit normal of a dense list of boundary point centered around point p
cv::Point2f getNormal(const contour_t& contour, const cv::Point& point)
{
    int sz = (int) contour.size();
    
    CV_Assert(sz != 0);
    
    int pointIndex = (int) (std::find(contour.begin(), contour.end(), point) - contour.begin());
    
    CV_Assert(pointIndex != contour.size());
    
    if (sz == 1)
    {
        return cv::Point2f(1.0f, 0.0f);
    } else if (sz < 2 * border_radius + 1)
    {
        // return the normal with respect to adjacent neigbourhood
        cv::Point adj = contour[(pointIndex + 1) % sz] - contour[pointIndex];
        return cv::Point2f(adj.y, -adj.x) / cv::norm(adj);
    }
    
    // create X and Y mat to SVD
    cv::Mat X(cv::Size(2, 2*border_radius+1), CV_32F);
    cv::Mat Y(cv::Size(1, 2*border_radius+1), CV_32F);
    
    CV_Assert(X.rows == Y.rows && X.cols == 2 && Y.cols == 1 && X.type() == Y.type()
              && Y.type() == CV_32F);
    
    int i = mod((pointIndex - border_radius), sz);
    
    float* Xrow;
    float* Yrow;
    
    int count = 0;
    int countXequal = 0;
    while (count < 2*border_radius+1)
    {
        Xrow = X.ptr<float>(count);
        Xrow[0] = contour[i].x;
        Xrow[1] = 1.0f;
        
        Yrow = Y.ptr<float>(count);
        Yrow[0] = contour[i].y;
        
        if (Xrow[0] == contour[pointIndex].x)
        {
            ++countXequal;
        }
        
        i = mod(i+1, sz);
        ++count;
    }
    
    if (countXequal == count)
    {
        return cv::Point2f(1.0f, 0.0f);
    }
    // TODO: weighted least squares
    
    // you have the needed points in contourWindow, now you perform weighted least Squares
    // to find the line of best fit
    cv::Mat sol;
    cv::solve(X, Y, sol, cv::DECOMP_SVD);
    
    CV_Assert(sol.type() == CV_32F);
    
    float slope = sol.ptr<float>(0)[0];
    cv::Point2f normal(-slope, 1);
    
    return normal / cv::norm(normal);
}


// get the confidence
double computeConfidence(const cv::Mat& confidencePatch)
{
    return cv::sum(confidencePatch)[0] / (double) confidencePatch.total();
}


// go over contours and compute the priority of each patch
void computePriority(const contours_t& contours, const cv::Mat& grayMat, const cv::Mat& confidenceMat, cv::Mat& priorityMat)
{
    CV_Assert(grayMat.type() == CV_32FC1 &&
              priorityMat.type() == CV_32FC1 &&
              confidenceMat.type() == CV_32FC1
              );
    
    // define some patches
    cv::Mat confidencePatch;
    cv::Mat magnitudePatch;
    
    cv::Point2f normal;
    cv::Point maxPoint;
    cv::Point2f gradient;
    
    double confidence;
    
    // get the derivatives and magnitude of the greyscale image
    cv::Mat dx, dy, magnitude;
    getDerivatives(grayMat, dx, dy);
    cv::magnitude(dx, dy, magnitude);
    
    // mask the magnitude
    cv::Mat maskedMagnitude(magnitude.size(), magnitude.type(), cv::Scalar(0));
    magnitude.copyTo(maskedMagnitude, (confidenceMat != 0.0f));
    cv::erode(maskedMagnitude, maskedMagnitude, cv::Mat());
    
    CV_Assert(maskedMagnitude.type() == CV_32FC1);
    
    // for each point in contour
    cv::Point point;
    
    for (int i = 0; i < contours.size(); ++i)
    {
        contour_t contour = contours[i];
        
        for (int j = 0; j < contour.size(); ++j)
        {
            
            point = contour[j];
            
            confidencePatch = getPatch(confidenceMat, point);
            
            // get confidence of patch
            confidence = cv::sum(confidencePatch)[0] / (double) confidencePatch.total();
            CV_Assert(0 <= confidence && confidence <= 1.0f);
            
            // get the normal to the border around point
            normal = getNormal(contour, point);
            
            // get the maximum gradient in source around patch
            magnitudePatch = getPatch(maskedMagnitude, point);
            cv::minMaxLoc(magnitudePatch, NULL, NULL, NULL, &maxPoint);
            gradient = cv::Point2f(
                                   -getPatch(dy, point).ptr<float>(maxPoint.y)[maxPoint.x],
                                   getPatch(dx, point).ptr<float>(maxPoint.y)[maxPoint.x]
                                 );
            
            // set the priority in priorityMat
            priorityMat.ptr<float>(point.y)[point.x] = std::abs((float) confidence * gradient.dot(normal));
            CV_Assert(priorityMat.ptr<float>(point.y)[point.x] >= 0);
        }
    }
}


// transfer the values from patch centered at psiHatQ to patch centered at psiHatP in
// mat according to maskMat
void transferPatch(const cv::Point& psiHatQ, const cv::Point& psiHatP, cv::Mat& mat, const cv::Mat& maskMat)
{
    CV_Assert(maskMat.type() == CV_8U);
    CV_Assert(mat.size() == maskMat.size());
    CV_Assert(radius <= psiHatQ.x && psiHatQ.x < mat.cols-radius && radius <= psiHatQ.y && psiHatQ.y < mat.rows-radius);
    CV_Assert(radius <= psiHatP.x && psiHatP.x < mat.cols-radius && radius <= psiHatP.y && psiHatP.y < mat.rows-radius);
    
    // copy contents of psiHatQ to psiHatP with mask
    getPatch(mat, psiHatQ).copyTo(getPatch(mat, psiHatP), getPatch(maskMat, psiHatP));
}

cv::Mat computeSSD(const cv::Mat& tmplate, const cv::Mat& source, const cv::Mat& tmplateMask)
{
    CV_Assert(tmplateMask.type() == CV_8UC1);
    CV_Assert(tmplateMask.size() == tmplate.size());
    CV_Assert(tmplate.rows <= source.rows && tmplate.cols <= source.cols);
    
    cv::Mat result(source.size() - cv::Size(2*radius, 2*radius), CV_32FC1);
    
    cv::Mat sourcePatch;
    cv::Mat maskedTmplate(tmplate.size(), tmplate.type(), cv::Scalar(0.0f));
    tmplate.copyTo(maskedTmplate, tmplateMask);
    
    cv::Mat squaredDiff;
    
    float* row;
    for (int y = 0; y < result.rows; ++y)
    {
        row = result.ptr<float>(y);
        for (int x = 0; x < result.cols; ++x)
        {
            sourcePatch = getPatch(source, cv::Point(x,y) + cv::Point(radius, radius)).clone();
            sourcePatch.setTo(0.0f, tmplateMask == 0);
            
            // now get the norm between maskedTmplate and sourcePatch
            cv::pow((sourcePatch - maskedTmplate), 2, squaredDiff);
            row[x] = (float) cv::sum(squaredDiff)[0];
        }
    }
    
    return result;
}

