#include <iostream>
#include <algorithm>

#define _VIGRA 0
#define _OPENCV 1

#if _VIGRA
#include <vigra/impex.hxx>
#include <vigra/multi_fft.hxx>
#include <vigra/convolution.hxx>
#include <vigra/gaborfilter.hxx>
#include <vigra/edgedetection.hxx>
#include <vigra/hdf5impex.hxx>
#endif

#if _OPENCV
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#endif

using namespace std;

int main() {

#if _VIGRA
    cout << "Hello, World! ... Using VIGRA" << endl;
#endif

#if _OPENCV
    cout << "Hello, World! ... Using OpenCV" << endl;
    string img_name = "img1.png";
    cv::Mat img = cv::imread(img_name, CV_LOAD_IMAGE_COLOR);
#endif


    return 0;
}