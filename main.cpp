#include <iostream>
#include <stdio.h>
#include <algorithm>

#define _VIGRA 1
#define _OPENCV 1
#define _SHOWIMAGE 0
#define _INPUT_FILE "/home/neuron/SIFT/Lenna.png"
#define _INPUT_FILE_SMALL "/home/neuron/SIFT/window.png"
#define _INPUT_FILE_MAP "/home/neuron/SIFT/map.png"
#define _NFEATURES 10
#define _NOCTAVELAYERS 3
#define _CONTRASTTHRES 0.04
#define _EDGETHRES 10
#define _SIGMA 1.6


#if _VIGRA
#include "vigra_sift.h"
#endif

#if _OPENCV
#include<opencv2/highgui.hpp>
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/xfeatures2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#endif

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

#if _OPENCV
int sift_opencv(){

    cout << "_________________________________________" << endl;
    cout << "_________________________________________" << endl;
    cout << "_________________________________________" << endl;
    cout << "Hello, World! ... Using OpenCV" << endl;

    cout << "read the image" << endl;

    cv::Mat img;
    img = cv::imread(_INPUT_FILE, CV_LOAD_IMAGE_GRAYSCALE);
    if(img.empty() )
    {
        cout << "Can't read one of the images..."<< endl;
        return -1;
    }

#if _SHOWIMAGE
    //
    cv::imshow("Main Image", img);
    cv::waitKey(0);
#endif

    //
    cout << "Configure SIFT detector " << endl;
    vector<cv::KeyPoint> keypoints;
    int _nfeatures = _NFEATURES;
    int _nOctaveLayers = _NOCTAVELAYERS;
    double _contrastThreshold = _CONTRASTTHRES;
    double _edgeThreshold = _EDGETHRES;
    double _sigma = _SIGMA;
    cv::Ptr<cv::xfeatures2d::SiftFeatureDetector> detector =
            cv::xfeatures2d::SiftFeatureDetector::create(
                    _nfeatures,
                    _nOctaveLayers,
                    _contrastThreshold,
                    _edgeThreshold,
                    _sigma
            );

    cout << "Detect Keypoints " << endl;
    detector->detect(img, keypoints);

#if _SHOWIMAGE

    cv::Mat img_keypoints;
    cv::drawKeypoints(
            img,
            keypoints,
            img_keypoints,
            cv::Scalar( 255, 255, 0 ),
            cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS
    );
    //
    cv::imshow("Keypoints", img_keypoints);
    cv::waitKey(0);
#endif
    //
    cout << "Configure SIFT extractor " << endl;

    cv::Mat descriptor;
    cv::Ptr<cv::xfeatures2d::SiftDescriptorExtractor> extractor =
            cv::xfeatures2d::SiftDescriptorExtractor::create(
                    _nfeatures,
                    _nOctaveLayers,
                    _contrastThreshold,
                    _edgeThreshold,
                    _sigma
            );

    //
    cout << "Extract descriptors" << endl;
    extractor->compute(img, keypoints,descriptor);

    //
    cout << "Results ....." << endl;
    cout<<"\tNumber of keypoints detected: "<<keypoints.size()<<endl;
    cout<<"\tKeypoint 1 coordinates:"<<keypoints[1].pt<<" Keypoint 1 angle:"<<keypoints[1].angle<<" Keypoint 1 scale/octave:"<<keypoints[1].octave<<endl;
    cout<<"\tDescriptor size:"<<descriptor.rows<<" x "<<descriptor.cols<<endl;
    cout<<"\tDescriptor: "<<descriptor.row(1)<<endl;

    //release memory
    detector.release();
    extractor.release();
    descriptor.release();

}
#endif


#if _VIGRA
int sift_vigra(){

    cout << "_________________________________________" << endl;
    cout << "_________________________________________" << endl;
    cout << "_________________________________________" << endl;

    cout << "Hello, World! ... Using VIGRA" << endl;

    cout << "read the image" << endl;

    cv::Mat img;
    img = cv::imread(_INPUT_FILE, CV_LOAD_IMAGE_GRAYSCALE);
    if(img.empty() )
    {
        cout << "Can't read one of the images..."<< endl;
        return -1;
    }



#if _SHOWIMAGE
    //
    cv::imshow("Main Image", img);
    cv::waitKey(0);
#endif


    //
    cout << "Configure SIFT detector " << endl;
    vector<cv::KeyPoint> keypoints;
    int _nfeatures = _NFEATURES;
    int _nOctaveLayers = _NOCTAVELAYERS;
    double _contrastThreshold = _CONTRASTTHRES;
    double _edgeThreshold = _EDGETHRES;
    double _sigma = _SIGMA;
    cv::Ptr<cv::xfeatures2d::SiftFeatureDetector> detector =
            cv::xfeatures2d::SiftFeatureDetector::create(
                    _nfeatures,
                    _nOctaveLayers,
                    _contrastThreshold,
                    _edgeThreshold,
                    _sigma
            );

    cout << "Detect Keypoints " << endl;
    detector->detect(img, keypoints);

#if _SHOWIMAGE

    cv::Mat img_keypoints;
    cv::drawKeypoints(
            img,
            keypoints,
            img_keypoints,
            cv::Scalar( 255, 255, 0 ),
            cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS
    );
    //
    cv::imshow("Keypoints", img_keypoints);
    cv::waitKey(0);
#endif
    //
    cout << "Configure SIFT extractor " << endl;


    cv::Mat descriptor;
    cv::Ptr<cv::xfeatures2d::SiftDescriptorExtractor> extractor =
            cv::xfeatures2d::SiftDescriptorExtractor::create(
                    _nfeatures,
                    _nOctaveLayers,
                    _contrastThreshold,
                    _edgeThreshold,
                    _sigma
            );

    //
    cout << "Extract descriptors" << endl;
    extractor->compute(img, keypoints, descriptor);


    //
    cout << "Results ....." << endl;
    cout << "\tNumber of keypoints detected: "<<keypoints.size()<<endl;
    cout << "\tKeypoint 1 coordinates:"<<keypoints[1].pt<<" Keypoint 1 angle:"<<keypoints[1].angle<<" Keypoint 1 scale/octave:"<<keypoints[1].octave<<endl;
    cout << "\tDescriptor size:"<<descriptor.rows<<" x "<<descriptor.cols<<endl;
    cout << "\tDescriptor: "<<descriptor.row(0)<<endl;
    cout << "\tDescriptor: "<<descriptor.row(1)<<endl;
    cout << "\tDescriptor: "<<descriptor.row(2)<<endl;
    cout << "\tDescriptor: "<<descriptor.row(3)<<endl;

    //////////////////////////////////////////////////////////////////// VIGRA
    // read also with vigra
    vigra::VigraSiftDescriptor vigraSiftDescriptor;
    vigraSiftDescriptor.setValues(
            _nfeatures,
            _nOctaveLayers,
            _contrastThreshold,
            _edgeThreshold,
            _sigma
    );
    vigraSiftDescriptor.allocateAndInitializeImage(_INPUT_FILE);
    vigraSiftDescriptor.allocateDescriptorArray();
    vigraSiftDescriptor.setKeypoints(convertKeyPointsCV2Vigra(keypoints));
    vigraSiftDescriptor.build_gauss_pyr();


    float* ret;
    std::cout << "\n\tDescriptor from vigra keypoint " << std::endl;
    std::cout << "\n\tDescriptor 0" << std::endl;
    ret = vigraSiftDescriptor.calculate_descriptors(0);
    for(int ii=0; ii<vigraSiftDescriptor.getDescriptorSize(); ii++){
        std::cout << (int)ret[ii] << " ";
    }
    std::cout << std::endl;
    std::cout << "\n\tDescriptor 2" << std::endl;
    ret = vigraSiftDescriptor.calculate_descriptors(2);
    for(int ii=0; ii<vigraSiftDescriptor.getDescriptorSize(); ii++){
        std::cout << (int)ret[ii] << " ";
    }
    std::cout << std::endl;
    std::cout << "\n\tDescriptor 3" << std::endl;
    ret = vigraSiftDescriptor.calculate_descriptors(3);
    for(int ii=0; ii<vigraSiftDescriptor.getDescriptorSize(); ii++){
        std::cout << (int)ret[ii] << " ";
    }
    std::cout << std::endl;
    std::cout << "\n\tDescriptor 4" << std::endl;
    ret = vigraSiftDescriptor.calculate_descriptors(4);
    for(int ii=0; ii<vigraSiftDescriptor.getDescriptorSize(); ii++){
        std::cout << (int)ret[ii] << " ";
    }
    std::cout << std::endl;
    std::cout << "\n\tDescriptor 5" << std::endl;
    ret = vigraSiftDescriptor.calculate_descriptors(5);
    for(int ii=0; ii<vigraSiftDescriptor.getDescriptorSize(); ii++){
        std::cout << (int)ret[ii] << " ";
    }
    std::cout << std::endl;
    std::cout << "\n\tDescriptor 8" << std::endl;
    ret = vigraSiftDescriptor.calculate_descriptors(8);
    for(int ii=0; ii<vigraSiftDescriptor.getDescriptorSize(); ii++){
        std::cout << (int)ret[ii] << " ";
    }
    std::cout << std::endl;
    std::cout << "\n\tDescriptor 9" << std::endl;
    ret = vigraSiftDescriptor.calculate_descriptors(9);
    for(int ii=0; ii<vigraSiftDescriptor.getDescriptorSize(); ii++){
        std::cout << (int)ret[ii] << " ";
    }
    std::cout << std::endl;



    //release memory
    detector.release();
    extractor.release();
    descriptor.release();


}
#endif

void sift_sb(){

    // set defaults
    int _nOctaveLayers = 3;
    float _sigma = 1.6f;
    float _contrastThreshold = 0.04f;
    int _edgeThreshold = 10;
    vigra::VigraSiftDetector vigraSiftDetector;

    vigraSiftDetector.setParameters(
            _nOctaveLayers,
            _sigma,
            _contrastThreshold,
            _edgeThreshold);

    // load image
    vigraSiftDetector.allocateAndInitializeImage(_INPUT_FILE_MAP);

    //
    std::vector<vigra::KeyPoint> vigkps =  vigraSiftDetector.detect_keypoints();

    // print output
    std::vector<cv::KeyPoint> cvkps = convertKeyPointsVigra2CV(vigkps);

    //
    cv::Mat img_src=cv::imread(_INPUT_FILE_MAP);

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
    cv::waitKey(0);

    cout << "sdfsdfsdfsd";
}

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
    vigraSiftDetector.allocateAndInitializeImage(_INPUT_FILE_MAP);

    // get the keypoints
    cout << "Detecting Keypoints ..." << endl;
    std::vector<vigra::KeyPoint> vigkps =  vigraSiftDetector.detect_keypoints();


#if _OPENCV
    // print output with opencv
    std::vector<cv::KeyPoint> cvkps = convertKeyPointsVigra2CV(vigkps);

    //
    cv::Mat img_src=cv::imread(_INPUT_FILE_MAP);

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
    vigraSiftDescriptor.allocateAndInitializeImage(_INPUT_FILE_MAP);
    cout << "Allocate memory for descriptor array ..." << endl;
    vigraSiftDescriptor.allocateDescriptorArray();
    vigraSiftDescriptor.setKeypoints(vigkps);

    cout << "Build Gaussian pyramid ..." << endl;
    vigraSiftDescriptor.build_gauss_pyr();


    cout << "Print Results ..." << endl;
    cout << "--------------------------------------------------------" << endl;
    float* ret;
    std::cout << "\n\tDescriptor for some vigra keypoints ... " << std::endl;
    std::cout << "\n\tDescriptor 0" << std::endl;
    ret = vigraSiftDescriptor.calculate_descriptors(0);
    for(int ii=0; ii<vigraSiftDescriptor.getDescriptorSize(); ii++){
        std::cout << (int)ret[ii] << " ";
    }
    std::cout << std::endl;
    std::cout << "\n\tDescriptor 2" << std::endl;
    ret = vigraSiftDescriptor.calculate_descriptors(2);
    for(int ii=0; ii<vigraSiftDescriptor.getDescriptorSize(); ii++){
        std::cout << (int)ret[ii] << " ";
    }
    std::cout << std::endl;
    std::cout << "\n\tDescriptor 3" << std::endl;
    ret = vigraSiftDescriptor.calculate_descriptors(3);
    for(int ii=0; ii<vigraSiftDescriptor.getDescriptorSize(); ii++){
        std::cout << (int)ret[ii] << " ";
    }
    std::cout << std::endl;
    std::cout << "\n\tDescriptor 4" << std::endl;
    ret = vigraSiftDescriptor.calculate_descriptors(4);
    for(int ii=0; ii<vigraSiftDescriptor.getDescriptorSize(); ii++){
        std::cout << (int)ret[ii] << " ";
    }
    std::cout << std::endl;
    std::cout << "\n\tDescriptor 5" << std::endl;
    ret = vigraSiftDescriptor.calculate_descriptors(5);
    for(int ii=0; ii<vigraSiftDescriptor.getDescriptorSize(); ii++){
        std::cout << (int)ret[ii] << " ";
    }
    std::cout << std::endl;
    std::cout << "\n\tDescriptor 8" << std::endl;
    ret = vigraSiftDescriptor.calculate_descriptors(8);
    for(int ii=0; ii<vigraSiftDescriptor.getDescriptorSize(); ii++){
        std::cout << (int)ret[ii] << " ";
    }
    std::cout << std::endl;
    std::cout << "\n\tDescriptor 9" << std::endl;
    ret = vigraSiftDescriptor.calculate_descriptors(9);
    for(int ii=0; ii<vigraSiftDescriptor.getDescriptorSize(); ii++){
        std::cout << (int)ret[ii] << " ";
    }
    std::cout << std::endl;
    
}

int main() {


#if _OPENCV
    //sift_opencv();
#endif
#if _VIGRA
    //sift_vigra();
#endif

    call_sift();



    return 0;
}


