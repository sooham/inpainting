//
//  inpainting.cpp
//  TODO: brief descripton here
//
//  Created by Sooham Rafiz on 2016-05-16.
//  Copyright © 2016 Sooham Rafiz. All rights reserved.


// TODO: Read up on Compiler Return Value Optimization
// you don't need to return pointers to internal data
// TODO: also read on this:  https://www.wikiwand.com/en/Resource_Acquisition_Is_Initialization
// TODO: openGL / GPU support on android
/*
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

#include <iostream>
#include <string>
#include <vector>

#define WINDOW_RADIUS 4
*/

/*
int main_stupid( int argc, const char** argv )
{
    
    
    // ---------------- getting image and mask string names -----------------------
    
    // ------------- image reading -------------------------
 
    // --------------------- showing the loaded images ---------------------------
 
    // ---- get a full matrix of the x and y derivatives
    cv::Mat dx, dy, magnitude;
    cv::Sobel(img_in_grayscale, dx, -1, 1, 0, -1);
    cv::Sobel(img_in_grayscale, dy, -1, 0, 1, -1);
    cv::magnitude(dx, dy, magnitude);
    
    std::cerr << std::endl << "mag depth is: " << magnitude.depth() << std::endl;
    std::cerr << "min is: " << min << std::endl;
    std::cerr << "max is: " << max << std::endl;
    std::cerr << std::endl;
    
    // show the magnitude
    cv::namedWindow("magnitude");
    cv::imshow("magnitude", magnitude);
    cv::waitKey(0);
    cv::destroyWindow("magnitude");
    
    // ----------- run the algorithm ------------
    // get source - all non target Points
    cv::Mat source;
    cv::findNonZero(mask, source);

    // get boundary
    //std::vector<std::vector<cv::Point>> *contours = get_boundary((mask == 0));
    
    // assign each pixel on boundary a  priority
    // done in through the mask 255 0 values
    
    // show the image as we compute the normal and gradient of each contour patch
    //cv::namedWindow("Algorithm progress");
    
    // iterate over the contours points one by one
    //for (int i = 0; i < contours->size(); ++i) {
        //std::vector<cv::Point> contour = (*contours)[i];
        //for (int j = 0; j < contour.size(); ++j) {
            cv::Mat img_in_clone = img_in.clone();
            
            // get the patch on point in img_in (used to find gradient in source regions)
            //cv::Mat patch_img_in = get_radius(img_in, contour[j], WINDOW_RADIUS);
            
            // get the patch on point in mask (used to indicate which area is target vs source)
            //cv::Mat patch_mask = get_radius(mask, contour[j], WINDOW_RADIUS);
            
            // get the gradient patch in the img_in
            //cv::Mat patch_magnitude = get_radius(magnitude, contour[j], WINDOW_RADIUS);
            // set all target areas in magnitude to 0
            cv::Mat occluded_magnitude;
            //cv::bitwise_and(patch_mask, patch_magnitude, occluded_magnitude);
            
            // go over the occluded_magnitude and get the Point of maximum
            int nRows = occluded_magnitude.rows;
            int nCols = occluded_magnitude.cols;
            
            if (occluded_magnitude.isContinuous()) {
                nCols *= nRows;
                nRows = 1;
            }
            
            //double max = 0;
            //int x = 0, y = 0;
            //uchar *input = occluded_magnitude.data;
            //for (int j = 0; j < nRows; ++j) {
            //    for (int i = 0; i < nCols; ++i) {
            //        if (max < input[occluded_magnitude.step * j + i]) {
            //            x = i;
            //            y = j;
            //        }
            //    }
            //}
            
            // get the dx and dy tuple
            //float gradient_x = dx.at<float>(x, y);
            //float gradient_y = dy.at<float>(x, y);
            
            //cv::Vec2f gradient(gradient_x, gradient_y);
            
            // now what you have the gradient, you need to get the normal from the mask patch
            
            
            // compute the data value
            
            
            // compute the confidence of the patch
            
            // compute the priority of the patch
            
            
            // draw the patch and point on img_in_clone for view debugging
            //cv::rectangle(img_in_clone, cv::Rect(contour[j].x-WINDOW_RADIUS, contour[j].y-WINDOW_RADIUS, 2 * WINDOW_RADIUS + 1, 2 * WINDOW_RADIUS + 1), cv::Scalar(0,255,0));
            // draw point
            //cv::circle(img_in_clone, contour[j], 0, cv::Scalar(0,0,255));
            // now find the patch with the least distance in the source of CIE Lab colorspace
            //cv::imshow("Algorithm progress", img_in_clone);
            //cv::waitKey(1);
            //cv::destroyWindow("Algorithm progress");
        //}
    //}
    //cv::waitKey(0);
    
    
    return 0;
}
*/