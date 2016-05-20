//
//  main.cpp
//  inpainting
//
//  Created by Sooham Rafiz on 2016-05-16.
//  Copyright © 2016 Sooham Rafiz. All rights reserved.
//

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

#include <vector>
#include <iostream>
#include <string>

#include "utils.hpp"

int main (int argc, char** argv) {
    
    // --------------- read filename strings ------------------
    std::string inputFilename, maskFilename;
    
    if (argc >= 3) {
        inputFilename = argv[1];
        maskFilename = argv[2];
    } else {
        std::cerr << "input images not provided" << std::endl;
        return -1;
    }
    
    // ---------------- read the images ------------------------
    cv::Mat inputImageColor, inputImageMask, inputImageCie;
    loadInpaintingImages(inputFilename, maskFilename, inputImageColor, inputImageMask);
    
    // show the images for testing
    showMat("color", inputImageColor);
    showMat("mask", inputImageMask);
    
    // ---------------- convert inputImageColor to inputImageCIE ---------------
    // CIE will be used when comparing euclidean distance between patches
    cv::cvtColor(inputImageColor, inputImageCie, cv::COLOR_BGR2Lab);
    showMat("CIE", inputImageCie);
    
    // ---------------- start the algorithm -----------------
    // get the contour
    contours_t contours;
    hierarchy_t hierarchy;
    getContours((inputImageMask == 0), contours, hierarchy);
    
    // draw the contour
    cv::Mat contourMat(inputImageMask.size(), inputImageMask.type(), cv::Scalar(0));
    for (int i = 0; i < contours.size(); ++i)
    {
        contour_t contour = contours[i];
        for (int j = 0; j < contour.size(); ++j)
        {
            contourMat.ptr(contour[j].y)[contour[j].x] = 255;
        }
    }
    
    showMat("contour", contourMat);
    
    // initial computations
    
    // main loop
    const size_t area = inputImageMask.total();
    while (cv::countNonZero(inputImageMask) != area)
    {
        // get the patch with the greatest priority
        // you need to compute the data for all patches on contour
        
        // get the patch in source with least distance to above patch
        
        // fillin cinputImageColor patch area
        
        // update inputImage mask confidences

    }
    return 0;
}