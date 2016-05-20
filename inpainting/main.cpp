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
    std::string colorFilename, maskFilename;
    
    if (argc == 3) {
        colorFilename = argv[1];
        maskFilename = argv[2];
    } else {
        std::cerr << "Usage ./inpainting colorImageFile maskImageFile" << std::endl;
        return -1;
    }
    
    // ---------------- read the images ------------------------
    cv::Mat colorMat, maskMat, grayMat, cieMat;
    loadInpaintingImages(
                         colorFilename,
                         maskFilename,
                         colorMat,
                         maskMat,
                         grayMat,
                         cieMat);
    
    // show the images for testing
    showMat("color", colorMat);
    showMat("mask", maskMat);
    showMat("CIE", cieMat);
    
    cv::Mat confidenceMat;     // holds the confidence values
    maskMat.convertTo(confidenceMat, CV_32F);
    confidenceMat = confidenceMat / 255.0f;
    
    // ---------------- start the algorithm -----------------
    
    contours_t contours;            // mask contours
    hierarchy_t hierarchy;          // contour hierarchy
    cv::Mat priorityMat(
                        maskMat.size(),
                        CV_32FC1,
                        cv::Scalar_<float>(0.0f)
                        );          // priority value matrix for each contour point
    cv::Point psiHatPCenter;        // the center Point of psiHatP patch
    cv::Mat psiHatPCie;             // psiHatP patch in Lab colorspace
    cv::Mat psiHatPGray;            // psiHatP patch in Gray colorspace
    cv::Mat psiHatPColor;           // psiHatP patch in BGR colorspace
    cv::Mat psiHatPConfidenceMat;   // psiHatP patch confidence mat
    double psiHatPConfidence;       // psiHatP patch confidence C(psiHatPMask)
    cv::Point psiHatQCenter;        // the center Point of psiHatQ patch
    
    // main loop
    const size_t area = maskMat.total();
    
    while (cv::countNonZero(maskMat) != area) // end when target is filled
    {
        // get the contours of mask
        getContours((maskMat == 0), contours, hierarchy);
        // compute the priority for all contour points
        computePriority(contours, grayMat, confidenceMat, priorityMat);
        // get the patch with the greatest priority
        psiHatPCenter = getMaxPosition<float>(priorityMat);
        psiHatPConfidenceMat = getPatch(confidenceMat, psiHatPCenter);
        psiHatPCie = getPatch(cieMat, psiHatPCenter);
        psiHatPGray = getPatch(grayMat, psiHatPCenter);
        psiHatPColor = getPatch(colorMat, psiHatPCenter);
        
        // get the patch in source with least distance to above patch
        psiHatQCenter = getClosestPatchPoint(cieMat, psiHatPCie, maskMat);

        
        // updates
        // fill in cieMat
        psiHatPCie += ((psiHatPConfidenceMat == 0.0f) & getPatch(cieMat, psiHatQCenter));
        // fill in grayMat
        psiHatPGray += ((psiHatPConfidenceMat == 0.0f) & getPatch(grayMat, psiHatQCenter));
        // fill in colorMat
        psiHatPColor += ((psiHatPConfidenceMat == 0.0f) & getPatch(colorMat, psiHatQCenter));
        
        // fill in confidenceMat with confidences C(pixel) = C(psiHatP)
        psiHatPConfidence = computeConfidence(psiHatPConfidenceMat);
        // set psiHatPMask(x,y) = psiHatPConfidence if psiHatPMask(x, y) == 0
        CV_Assert((psiHatPConfidenceMat == 0.0f).type() == CV_8UC1);
        psiHatPConfidenceMat += ((psiHatPConfidenceMat == 0.0f) & psiHatPConfidence);
        
        // update maskMat
        // maskMat is all non-zero elements in confidenceMat
        maskMat = (psiHatPConfidence != 0.0f);
    }
    return 0;
}