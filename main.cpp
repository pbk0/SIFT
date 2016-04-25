#include <iostream>
#include <algorithm>

#define _VIGRA 0

#if _VIGRA
#include <vigra/impex.hxx>
#include <vigra/multi_fft.hxx>
#include <vigra/convolution.hxx>
#include <vigra/gaborfilter.hxx>
#include <vigra/edgedetection.hxx>
#include <vigra/hdf5impex.hxx>
#else
#include <opencv2/xfeatures2d/nonfree.hpp>
#endif

using namespace std;

int main() {

#if _VIGRA
    cout << "Hello, World! ... Using VIGRA" << endl;
#else
    cout << "Hello, World! ... Using OpenCV" << endl;
#endif
    return 0;
}