//
//  main.cpp
//  inpainting
//
//  Created by Sooham Rafiz on 2016-05-16.

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
    
    cv::Mat confidenceMat;     // holds the confidence values
    maskMat.convertTo(confidenceMat, CV_32F);
    confidenceMat = confidenceMat / 255.0f;
    
    // add borders around maskMat and confidenceMat
    cv::copyMakeBorder(maskMat, maskMat, radius, radius, radius, radius, cv::BORDER_CONSTANT, 255);
    cv::copyMakeBorder(confidenceMat, confidenceMat, radius, radius, radius, radius, cv::BORDER_CONSTANT, 0.001);
    
    // ---------------- start the algorithm -----------------
    
    contours_t contours;            // mask contours
    hierarchy_t hierarchy;          // contour hierarchy
    
    cv::Mat priorityMat(
                        confidenceMat.size(),
                        CV_32FC1);  // priority value matrix for each contour point
    
    CV_Assert(colorMat.size() == grayMat.size() && grayMat.size() == confidenceMat.size() && colorMat.size() == maskMat.size());
    
    cv::Point psiHatPCenter;        // the center Point of psiHatP patch
    
    cv::Mat psiHatPCie;             // psiHatP patch in Lab colorspace
    cv::Mat psiHatPConfidenceMat;   // psiHatP patch confidence mat
    
    double psiHatPConfidence;       // psiHatP patch confidence C(psiHatPMask)
    
    cv::Point psiHatQCenter;        // the center Point of psiHatQ patch
    
    
    // main loop
    const size_t area = maskMat.total();
    
    while (cv::countNonZero(maskMat) != area)
        // end when target is filled
    {
        // set priority matrix to 0
        priorityMat.setTo(0.0f);
        
        // get the contours of mask
        getContours((maskMat == 0), contours, hierarchy);
        
        // compute the priority for all contour points
        computePriority(contours, grayMat, confidenceMat, priorityMat);
        
        // get the patch with the greatest priority
        psiHatPCenter = getMaxPosition<float>(priorityMat);
        
        psiHatPConfidenceMat = getPatch(confidenceMat, psiHatPCenter);
        psiHatPCie = getPatch(cieMat, psiHatPCenter);
        
        // get the patch in source with least distance to psiHatPCie
        // TODO: move patch extraction to inside function getClosestPatchPoint
        psiHatQCenter = getClosestPatchPoint(cieMat, psiHatPCie, maskMat);

        // updates
        // copy from psiHatQ to psiHatP for each
        transferPatch(psiHatQCenter, psiHatPCenter, cieMat, (maskMat == 0));
        transferPatch(psiHatQCenter, psiHatPCenter, grayMat, (maskMat == 0));
        transferPatch(psiHatQCenter, psiHatPCenter, colorMat, (maskMat == 0));
        
        // fill in confidenceMat with confidences C(pixel) = C(psiHatP)
        psiHatPConfidence = computeConfidence(psiHatPConfidenceMat);
        CV_Assert(0 <= psiHatPConfidence && psiHatPConfidence <= 1.0f);
        // update confidence
        psiHatPConfidenceMat.setTo((float) psiHatPConfidence, (psiHatPConfidenceMat == 0.0f));
        // update maskMat
        maskMat = (confidenceMat != 0.0f);
    }
    
    showMat("final result", colorMat, 0);
    return 0;
}