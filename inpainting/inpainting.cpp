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

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

#include <iostream>
#include <string>
#include <vector>

#define WINDOW_RADIUS 4

std::vector<std::vector<cv::Point>> *get_boundary(const cv::Mat &mask) {
    // the vector holds the boundary elements for the mask
    // dynamically allocate memory for this vector
    // TODO: delete memory later outside of function
    // TODO: also change memory allocation from C style to C++ exception style
    std::vector<std::vector<cv::Point>> *contours = new (std::nothrow) std::vector<std::vector<cv::Point>>;
    
    if (!contours) {
        std::cerr << "Memory allocation failed (malloc)" << std::endl;
        std::exit(-1);
    }
    std::vector<cv::Vec4i> hierarchy;
    
    // get the boundary from the mask using findCountours
    cv::findContours(mask.clone(), *contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
    
    // TODO: delete draw contours
    // draw the contours to verify
    cv::Mat drawing = cv::Mat::zeros(mask.size(), CV_8UC1);
    
    for (int i = 0; i < contours->size(); i++) {
        cv::Scalar color(rand() & 255, rand() & 255, rand() & 255);
        // TODO: better drawings please
        cv::drawContours(drawing, *contours, i, color, 1, 8, hierarchy, 1, cv::Point());
    }
    
    // TODO: contour includes images edge pixels, remove these unneeded pictures
    cv::namedWindow("Contours");
    cv::imshow("Contours", drawing);
    cv::waitKey(0);
    cv::destroyWindow("Contours");
    return contours;
}





// get a patch of size radius around Point p in Mat
cv::Mat get_radius(cv::Mat image, cv::Point &p, int r=0) {
    // uses return value optimization
    // make sure roi is in the boundary of image
    
    // if (0 <= p.x && p.x <= image.cols && 0 <= p.y && p.y <= image.rows) TODO
    return image(cv::Range(std::max(0, p.y-r), std::min(image.rows, p.y+r)), cv::Range(std::max(0, p.x-r), std::min(image.cols, p.x+r)));
}








int main( int argc, const char** argv )
{
    // TODO: set Xcode command line arguments to point to images
    
    
    // ---------------- getting image and mask string names -----------------------
    std::string in_image_file;
    std::string mask_file;
    
    if (argc >= 3) {
        in_image_file = argv[1];
        mask_file = argv[2];
    } else {
        std::cerr << "input images not provided" << std::endl;
        return -1;
    }
    
    // ------------- image reading -------------------------
    cv::Mat img_in = cv::imread(in_image_file, 1);
    cv::Mat mask = cv::imread(mask_file, 0);
    
    if (img_in.empty() || mask_file.empty()) {
        std::cerr << "Image has not be loaded properly (imread)" << std::endl;
        return -1;
    }
    
    if (img_in.rows != mask.rows || img_in.cols != mask.cols) {
        std::cerr << "Images do not have same dimensions" << std::endl;
        return -1;
    }
    
    // add an alpha channel to img_in
    img_in.convertTo(img_in, CV_32F);
    img_in = img_in / 255.0f;
    
    std::cerr << "img_in aspects" << std::endl << std::endl;
    std::cerr << "\tchannels: " << img_in.channels() << std::endl;
    std::cerr << "\tsize: " << img_in.size << std::endl;
    std::cerr << "\tdepth: " << img_in.depth() << std::endl << std::endl;
    
    
    std::cerr << "mask aspects" << std::endl << std::endl;
    std::cerr << "\tchannels: " << mask.channels() << std::endl;
    std::cerr << "\tsize: " << mask.size << std::endl;
    std::cerr << "\tdepth: " << mask.depth() << std::endl << std::endl;
    
    
    // --------------------- showing the loaded images ---------------------------
    
    cv::namedWindow("img_in");
    cv::imshow("img_in", img_in);
    cv::waitKey(0);
    cv::destroyWindow("img_in");
    
    cv::namedWindow("mask");
    cv::imshow("mask", mask);
    cv::waitKey(0);
    cv::destroyWindow("mask");
    
    // convert img_in to grayscale
    cv::Mat img_in_grayscale;
    cv::cvtColor(img_in, img_in_grayscale, cv::COLOR_BGR2GRAY);
    
    double min, max;
    cv::minMaxLoc(img_in_grayscale, &min, &max);
    std::cerr << std::endl << "depth is: " << img_in_grayscale.depth() << std::endl;
    std::cerr << "min is: " << min << std::endl;
    std::cerr << "max is: " << max << std::endl;
    std::cerr << std::endl;
    
    // convert img_in to CIELab color space
    // TODO: should I divide by 255?
    cv::Mat img_in_cie;
    cv::cvtColor(img_in, img_in_cie, cv::COLOR_BGR2Lab);
    
    // show grayscale
    cv::namedWindow("gray");
    cv::imshow("gray", img_in_grayscale);
    cv::waitKey(0);
    cv::destroyWindow("gray");
    
    // ---- assign confidence value to img_in pixels ---------
    // done through mask (invert when getting boundary)
    
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
    std::vector<std::vector<cv::Point>> *contours = get_boundary((mask == 0));
    
    // assign each pixel on boundary a  priority
    // done in through the mask 255 0 values
    
    // show the image as we compute the normal and gradient of each contour patch
    cv::namedWindow("Algorithm progress");
    
    // iterate over the contours points one by one
    for (int i = 0; i < contours->size(); ++i) {
        std::vector<cv::Point> contour = (*contours)[i];
        for (int j = 0; j < contour.size(); ++j) {
            cv::Mat img_in_clone = img_in.clone();
            
            // get the patch on point in img_in (used to find gradient in source regions)
            cv::Mat patch_img_in = get_radius(img_in, contour[j], WINDOW_RADIUS);
            
            // get the patch on point in mask (used to indicate which area is target vs source)
            cv::Mat patch_mask = get_radius(mask, contour[j], WINDOW_RADIUS);
            
            // get the gradient patch in the img_in
            cv::Mat patch_magnitude = get_radius(magnitude, contour[j], WINDOW_RADIUS);
            // set all target areas in magnitude to 0
            cv::Mat occluded_magnitude;
            cv::bitwise_and(patch_mask, patch_magnitude, occluded_magnitude);
            
            // go over the occluded_magnitude and get the Point of maximum
            int nRows = occluded_magnitude.rows;
            int nCols = occluded_magnitude.cols;
            
            if (occluded_magnitude.isContinuous()) {
                nCols *= nRows;
                nRows = 1;
            }
            
            float max = 0;
            int x = 0, y = 0;
            uchar *input = occluded_magnitude.data;
            for (int j = 0; j < nRows; ++j) {
                for (int i = 0; i < nCols; ++i) {
                    if (max < input[occluded_magnitude.step * j + i]) {
                        x = i;
                        y = j;
                    }
                }
            }
            
            // get the dx and dy tuple
            float gradient_x = dx.at<float>(x, y);
            float gradient_y = dy.at<float>(x, y);
            
            cv::Vec2f gradient(gradient_x, gradient_y);
            
            // now what you have the gradient, you need to get the normal from the mask patch
            
            
            // compute the data value
            
            
            // compute the confidence of the patch
            
            // compute the priority of the patch
            
            
            // draw the patch and point on img_in_clone for view debugging
            cv::rectangle(img_in_clone, cv::Rect(contour[j].x-WINDOW_RADIUS, contour[j].y-WINDOW_RADIUS, 2 * WINDOW_RADIUS + 1, 2 * WINDOW_RADIUS + 1), cv::Scalar(0,255,0));
            // draw point
            cv::circle(img_in_clone, contour[j], 0, cv::Scalar(0,0,255));
            // now find the patch with the least distance in the source of CIE Lab colorspace
            cv::imshow("Algorithm progress", img_in_clone);
            cv::waitKey(1);
            cv::destroyWindow("Algorithm progress");
        }
    }
    cv::waitKey(0);
    
    
    return 0;
}