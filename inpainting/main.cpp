//
//  main.cpp
//  An example main function showcasing how to use the inpainting function
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
        std::cerr << "Usage: ./inpainting colorImageFile maskImageFile" << std::endl;
        return -1;
    }
    
    // ---------------- read the images ------------------------
    // colorMat     - color picture + border
    // maskMat      - mask picture + border
    // grayMat      - gray picture + border
    // cieMat       - CIE Lab picture + border
    cv::Mat colorMat, maskMat, grayMat, cieMat;
    loadInpaintingImages(
                         colorFilename,
                         maskFilename,
                         colorMat,
                         maskMat,
                         grayMat,
                         cieMat);
    
    // confidenceMat - confidence picture + border
    cv::Mat confidenceMat;
    maskMat.convertTo(confidenceMat, CV_32F);
    confidenceMat /= 255.0f;
    
    // add borders around maskMat and confidenceMat
    cv::copyMakeBorder(maskMat, maskMat,
                       RADIUS, RADIUS, RADIUS, RADIUS,
                       cv::BORDER_CONSTANT, 255);
    cv::copyMakeBorder(confidenceMat, confidenceMat,
                       RADIUS, RADIUS, RADIUS, RADIUS,
                       cv::BORDER_CONSTANT, 0.0001f);
    
    // ---------------- start the algorithm -----------------
    
    contours_t contours;            // mask contours
    hierarchy_t hierarchy;          // contours hierarchy
    
    
    // priorityMat - priority values for all contour points + border
    cv::Mat priorityMat(
                        confidenceMat.size(),
                        CV_32FC1
                        );  // priority value matrix for each contour point
    
    assert(
           colorMat.size() == grayMat.size() &&
           colorMat.size() == cieMat.size() &&
           colorMat.size() == confidenceMat.size() &&
           colorMat.size() == maskMat.size()
           );
    
    cv::Point psiHatP;          // psiHatP - point of highest confidence
    
    cv::Mat psiHatPCie;         // cie patch around psiHatP
    
    cv::Mat psiHatPConfidence;  // confidence patch around psiHatP
    double confidence;          // confidence of psiHatPConfidence
    
    cv::Point psiHatQ;          // psiHatQ - point of closest patch
    
    cv::Mat result;             // holds result from template matching
    cv::Mat erodedMask;         // eroded mask
    
    // eroded mask is used to ensure that psiHatQ is not overlapping with target
    cv::erode(maskMat, erodedMask, cv::Mat(), cv::Point(-1, -1), RADIUS);
    
    // TODO delete
    cv::Mat drawMat;
    // end delete
    
    // main loop
    const size_t area = maskMat.total();
    
    while (cv::countNonZero(maskMat) != area)   // end when target is filled
    {
        // set priority matrix to 0
        priorityMat.setTo(0.0f);
        
        // get the contours of mask
        getContours((maskMat == 0), contours, hierarchy);

        // TODO: delete
        drawMat = colorMat.clone();
        // end delete
        
        // compute the priority for all contour points
        // TODO: multiply by confidence using elementwise multiplication
        computePriority(contours, grayMat, confidenceMat, priorityMat);
        
        // get the patch with the greatest priority
        // TODO: psiHat P should be a vector or a sparse matrix
        // not a dense matrix
        cv::minMaxLoc(priorityMat, NULL, NULL, NULL, &psiHatP);
        psiHatPCie = getPatch(cieMat, psiHatP);
        // mask the psiHatPCie with source to prevent target pixels from playing a role
        psiHatPConfidence = getPatch(confidenceMat, psiHatP);
        
        // get the patch in source with least distance to psiHatPCie wrt source of psiHatP
        result = computeSSD(psiHatPCie, cieMat, (psiHatPConfidence != 0.0f));
        cv::normalize(result, result, 0, 1, cv::NORM_MINMAX);
        cv::copyMakeBorder(result, result, RADIUS, RADIUS, RADIUS, RADIUS, cv::BORDER_CONSTANT, 1.0f);
        
        result.setTo(1.0f, erodedMask == 0);
        
        // get minimum point of SSD between psiHatPCie and cieMat
        cv::minMaxLoc(result, NULL, NULL, &psiHatQ);
        
        assert(psiHatQ != psiHatP);
        
        // TODO: delete
        cv::rectangle(drawMat, psiHatP - cv::Point(RADIUS, RADIUS), psiHatP + cv::Point(RADIUS+1, RADIUS+1), cv::Scalar(255, 0, 0));
        cv::rectangle(drawMat, psiHatQ - cv::Point(RADIUS, RADIUS), psiHatQ + cv::Point(RADIUS+1, RADIUS+1), cv::Scalar(0, 0, 255));
        showMat("red - psiHatQ", drawMat);
        // end delete

        // updates
        // copy from psiHatQ to psiHatP for each colorspace
        transferPatch(psiHatQ, psiHatP, cieMat, (maskMat == 0));
        transferPatch(psiHatQ, psiHatP, grayMat, (maskMat == 0));
        transferPatch(psiHatQ, psiHatP, colorMat, (maskMat == 0));
        
        // fill in confidenceMat with confidences C(pixel) = C(psiHatP)
        confidence = computeConfidence(psiHatPConfidence);
        assert(0 <= confidence && confidence <= 1.0f);
        // update confidence
        psiHatPConfidence.setTo((float) confidence, (psiHatPConfidence == 0.0f));
        // update maskMat
        maskMat = (confidenceMat != 0.0f);
    }
    
    showMat("final result", colorMat, 0);
    return 0;
}