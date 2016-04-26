#include <iostream>
#include <stdio.h>
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
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#endif

using namespace std;

int main() {

#if _VIGRA
    cout << "Hello, World! ... Using VIGRA" << endl;
#endif

#if _OPENCV
    cout << "Hello, World! ... Using OpenCV" << endl;
    cv::Mat img;
    img = cv::imread("/home/neuron/SIFT/img.pgm", CV_LOAD_IMAGE_COLOR);
    if(img.empty() )
    {
        cout << "Can't read one of the images..."<< endl;
        return -1;
    }
    //cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    cv::imshow("Display Image", img);

    cv::waitKey(0);

#endif


    return 0;
}