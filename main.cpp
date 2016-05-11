#include <iostream>
#include <stdio.h>
#include <algorithm>
#include "vigra_sift.h"
#include <opencv2/opencv.hpp>

#define _SHOWIMAGE 1
//#define _INPUT_FILE "/home/neuron/SIFT/Lenna.png"
#define _INPUT_FILE "/home/neuron/SIFT/normal_window.png"
//#define _INPUT_FILE "/home/neuron/SIFT/rotated_window.png"
//#define _INPUT_FILE "/home/neuron/SIFT/scaled_window.png"
//#define _INPUT_FILE "/home/neuron/SIFT/map.png"
#define _NFEATURES 10
#define _NOCTAVELAYERS 3
#define _CONTRASTTHRES 0.04
#define _EDGETHRES 10
#define _SIGMA 1.6

using namespace std;

/**
 * Conversion methods for keypoint. Helpful for plotting.
 */
std::vector<vigra::KeyPoint> convertKeyPointsCV2Vigra(
        std::vector<cv::KeyPoint> cvkps
){
    std::vector<vigra::KeyPoint> vigrakps;
    for(auto const& value: cvkps) {
        vigra::KeyPoint kp;
        kp.ptx = value.pt.x;
        kp.pty = value.pt.y;
        kp.angle = value.angle;
        kp.octave = value.octave;
        kp.size = value.size;
        kp.response = value.response;
        vigrakps.push_back(kp);
    }
    return vigrakps;
}

/**
 * Conversion methods for keypoint. Helpful for plotting.
 */
std::vector<cv::KeyPoint> convertKeyPointsVigra2CV(
        std::vector<vigra::KeyPoint> vigkps
){
    std::vector<cv::KeyPoint> cvkps;
    for(auto const& value: vigkps) {
        cv::KeyPoint kp;
        kp.pt.x = value.ptx;
        kp.pt.y = value.pty;
        kp.angle = value.angle;
        kp.octave = value.octave;
        kp.size = value.size;
        kp.response = value.response;
        cvkps.push_back(kp);
    }
    return cvkps;
}

/**
 * This is important method which calls both keypoint detection and extraction.
 */
int call_sift(){

    // set default parameters
    //int _nfeatures = _NFEATURES;
    int _nOctaveLayers = _NOCTAVELAYERS;
    float _contrastThreshold = _CONTRASTTHRES;
    int _edgeThreshold = _EDGETHRES;
    float _sigma = _SIGMA;

    // set descriptor parameters
    cout << "Setting parameters ..." << endl;
    vigra::VigraSiftDetector vigraSiftDetector;
    vigraSiftDetector.setParameters(
            _nOctaveLayers,
            _sigma,
            _contrastThreshold,
            _edgeThreshold);

    // load image
    cout << "Loading image ..." << endl;
    vigraSiftDetector.allocateAndInitializeImage(_INPUT_FILE);

    // get the keypoints
    cout << "Detecting Keypoints ..." << endl;
    std::vector<vigra::KeyPoint> vigkps =  vigraSiftDetector.detect_keypoints();


#if _SHOWIMAGE
    // print output with opencv
    std::vector<cv::KeyPoint> cvkps = convertKeyPointsVigra2CV(vigkps);

    //
    cv::Mat img_src=cv::imread(_INPUT_FILE);

    cv::Mat img_keypoints;
    cv::drawKeypoints(
            img_src,
            cvkps,
            img_keypoints,
            cv::Scalar( 255, 255, 0 ),
            cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS
    );
    //
    cv::imshow("Keypoints", img_keypoints);
    cv::imwrite("kp.png", img_keypoints);
    cv::waitKey(0);

# endif
    
    // extract the descriptors
    cout << "Setting Keypoint descriptors parameters..." << endl;
    vigra::VigraSiftDescriptor vigraSiftDescriptor;
    vigraSiftDescriptor.setValues(
            (int) vigkps.size(),
            _nOctaveLayers,
            _contrastThreshold,
            _edgeThreshold,
            _sigma
    );
    vigraSiftDescriptor.allocateAndInitializeImage(_INPUT_FILE);
    cout << "Allocate memory for descriptor array ..." << endl;
    vigraSiftDescriptor.allocateDescriptorArray();
    vigraSiftDescriptor.setKeypoints(vigkps);
    vigraSiftDescriptor.build_gauss_pyr();


    cout << "Print Results ..." << endl;
    cout << "--------------------------------------------------------" << endl;
    float* ret;
    std::cout << "\n\tDescriptor for some vigra keypoints ... " << std::endl;
    std::cout << "\n\tKeypoint 0" << std::endl;
    ret = vigraSiftDescriptor.calculate_descriptors(0);
    for(int ii=0; ii<vigraSiftDescriptor.getDescriptorSize(); ii++){
        std::cout << (int)ret[ii] << " ";
    }
    std::cout << std::endl;
    std::cout << "\n\tKeypoint 2" << std::endl;
    ret = vigraSiftDescriptor.calculate_descriptors(2);
    for(int ii=0; ii<vigraSiftDescriptor.getDescriptorSize(); ii++){
        std::cout << (int)ret[ii] << " ";
    }
    std::cout << std::endl;
    std::cout << "\n\tKeypoint 3" << std::endl;
    ret = vigraSiftDescriptor.calculate_descriptors(3);
    for(int ii=0; ii<vigraSiftDescriptor.getDescriptorSize(); ii++){
        std::cout << (int)ret[ii] << " ";
    }
    std::cout << std::endl;
    std::cout << "\n\tKeypoint 4" << std::endl;
    ret = vigraSiftDescriptor.calculate_descriptors(4);
    for(int ii=0; ii<vigraSiftDescriptor.getDescriptorSize(); ii++){
        std::cout << (int)ret[ii] << " ";
    }
    std::cout << std::endl;
    std::cout << "\n\tKeypoint 5" << std::endl;
    ret = vigraSiftDescriptor.calculate_descriptors(5);
    for(int ii=0; ii<vigraSiftDescriptor.getDescriptorSize(); ii++){
        std::cout << (int)ret[ii] << " ";
    }
    std::cout << std::endl;
    std::cout << "\n\tKeypoint 8" << std::endl;
    ret = vigraSiftDescriptor.calculate_descriptors(8);
    for(int ii=0; ii<vigraSiftDescriptor.getDescriptorSize(); ii++){
        std::cout << (int)ret[ii] << " ";
    }
    std::cout << std::endl;
    std::cout << "\n\tKeypoint 9" << std::endl;
    ret = vigraSiftDescriptor.calculate_descriptors(9);
    for(int ii=0; ii<vigraSiftDescriptor.getDescriptorSize(); ii++){
        std::cout << (int)ret[ii] << " ";
    }
    std::cout << std::endl;
    
}

/**
 * Main method to call the SIFT
 */
int main() {

    // call both keypoint detector and descriptor code
    call_sift();

    return 0;
}


