//
//  inpainting.cpp
//  TODO: brief descripton here
//
//  Created by Sooham Rafiz on 2016-05-16.
//  Copyright © 2016 Sooham Rafiz. All rights reserved.


// TODO: Read up on Compiler Return Value Optimization
// you don't need to return pointers to internal data
// TODO: also read on this:  https://www.wikiwand.com/en/Resource_Acquisition_Is_Initialization

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <sstream>

// takes in an image and the binary mask, adds the appropriate alpha channel
// and returns the confidence integer map
// mask is of type CV_8UC1 255 if target else 0

// function to get the confidence of a patch

// function to set the confidence of a patch

// function to set the data

// function to get the priority of a patch

// get the center of a patch at point p

// get the boundary of a CV_8UC1 matrix featuring a closed shape

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
        cv::drawContours(drawing, *contours, i, color, 2, 8, hierarchy, 0, cv::Point());
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
    return image(cv::Rect(p.x, p.y, 1+2*r, 1+2*r));
}


int main( int argc, const char** argv )
{
    // TODO: set Xcode command line arguments to point to images
    
    std::string in_image_file;
    std::string mask_file;
    
    if (argc >= 3) {
        in_image_file = argv[1];
        mask_file = argv[2];
    } else {
        std::cerr << "input images not provided" << std::endl;
        return -1;
    }
    
    std::cerr << "input filename: " << in_image_file << std::endl;
    std::cerr << "input mask: " << mask_file << std::endl;
    
    cv::Mat img_in = cv::imread(in_image_file, -1);
    cv::Mat mask = cv::imread(mask_file, 0);
    
    if (img_in.empty() || mask_file.empty()) {
        std::cerr << "Image has not be loaded properly (imread)" << std::endl;
        return -1;
    }
    
    if (img_in.rows != mask.rows || img_in.cols != mask.cols) {
        std::cerr << "Images do not have same dimensions" << std::endl;
        return -1;
    }
    
    // convert img_in to 32F
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
    
    
    // showing the loaded files
    
    cv::namedWindow("img_in");
    cv::imshow("img_in", img_in);
    cv::waitKey(0);
    cv::destroyWindow("img_in");
    
    cv::namedWindow("mask");
    cv::imshow("mask", mask);
    cv::waitKey(0);
    cv::destroyWindow("mask");
    
    // convert img_in to CIELab color space
    cv::Mat img_in_cie;
    cv::cvtColor(img_in, img_in_cie, cv::COLOR_BGR2Lab);
    img_in_cie = img_in_cie / 255.0;
    
    double min, max;
    cv::minMaxLoc(img_in_cie, &min, &max);
    // show the img_in_cie
    std::cerr << "img_in_cie aspects" << std::endl << std::endl;
    std::cerr << "\tchannels: " << img_in_cie.channels() << std::endl;
    std::cerr << "\tsize: " << img_in_cie.size << std::endl;
    std::cerr << "\tdepth: " << img_in_cie.depth() << std::endl << std::endl;
    std::cerr << "\tmin: " << min << std::endl;
    std::cerr << "\tmax: " << max << std::endl;
    
    cv::namedWindow("img_in_cie");
    cv::imshow("img_in_cie", img_in_cie);
    cv::waitKey(0);
    cv::destroyWindow("img_in_cie");
    
    // TODO: I'm using single precision floating point, so alls good
    
    std::vector<std::vector<cv::Point>> *contours = get_boundary(mask);

    
    return 0;
}