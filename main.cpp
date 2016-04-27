#include <iostream>
#include <stdio.h>
#include <algorithm>

#define _VIGRA 1
#define _OPENCV 1
#define _SHOWIMAGE 0

#if _VIGRA
#include "helper_vigra.h"
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
#include <opencv2/xfeatures2d/nonfree.hpp>

#endif

using namespace std;


#if _OPENCV
int sift_opencv(){

    cout << "_________________________________________" << endl;
    cout << "_________________________________________" << endl;
    cout << "_________________________________________" << endl;
    cout << "Hello, World! ... Using OpenCV" << endl;

    cout << "read the image" << endl;

    cv::Mat img;
    img = cv::imread("/home/neuron/SIFT/img.pgm", CV_LOAD_IMAGE_GRAYSCALE);
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
    int _nfeatures = 1000;
    int _nOctaveLayers = 3;
    double _contrastThreshold = 0.04;
    double _edgeThreshold = 10;
    double _sigma = 1.6;
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

#if _OPENCV
    cv::Mat img;
    img = cv::imread("/home/neuron/SIFT/img.pgm", CV_LOAD_IMAGE_GRAYSCALE);
    if(img.empty() )
    {
        cout << "Can't read one of the images..."<< endl;
        return -1;
    }
#endif

    // read also with vigra
    vigra::ImageImportInfo vigra_img_info("/home/neuron/SIFT/img.pgm");
    vigra::MultiArray<2, vigra::UInt8> vigra_img_array(vigra_img_info.shape());
    vigra::importImage(vigra_img_info, vigra_img_array);


#if _SHOWIMAGE
    //
    cv::imshow("Main Image", img);
    cv::waitKey(0);
#endif


#if _OPENCV
    //
    cout << "Configure SIFT detector " << endl;
    vector<cv::KeyPoint> keypoints;
    int _nfeatures = 1000;
    int _nOctaveLayers = 3;
    double _contrastThreshold = 0.04;
    double _edgeThreshold = 10;
    double _sigma = 1.6;
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
    vigra::VigraSiftDescriptor vigraSiftDescriptor;
    vigraSiftDescriptor.setValues(
            _nfeatures,
            _nOctaveLayers,
            _contrastThreshold,
            _edgeThreshold,
            _sigma
    );

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
    cout<<"\tNumber of keypoints detected: "<<keypoints.size()<<endl;
    cout<<"\tKeypoint 1 coordinates:"<<keypoints[1].pt<<" Keypoint 1 angle:"<<keypoints[1].angle<<" Keypoint 1 scale/octave:"<<keypoints[1].octave<<endl;
    cout<<"\tDescriptor size:"<<descriptor.rows<<" x "<<descriptor.cols<<endl;
    cout<<"\tDescriptor: "<<descriptor.row(1)<<endl;

    //release memory
    detector.release();
    extractor.release();
    descriptor.release();
#endif
}
#endif


int main() {

#if _OPENCV
    sift_opencv();
#endif
#if _VIGRA
    sift_vigra();
#endif



    return 0;
}


