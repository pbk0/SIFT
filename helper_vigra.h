//
// Created by neuron on 25.04.16.
//

#ifndef SIFT_HELPER_VIGRA_H
#define SIFT_HELPER_VIGRA_H

#include <vigra/impex.hxx>
#include <vigra/multi_fft.hxx>
#include <vigra/multi_array.hxx>
#include <vigra/convolution.hxx>
#include <vigra/gaborfilter.hxx>
#include <vigra/edgedetection.hxx>
#include <vigra/hdf5impex.hxx>
#include <vigra/imageinfo.hxx>

namespace vigra
{

class VigraSiftDescriptor
{
    int features;
    int octaveLayers;
    double contrastThreshold;
    double edgeThreshold;
    double sigma;
    MultiArray<2, vigra::UInt8> image_array;
    MultiArray<2, vigra::UInt8> descriptor_array;


    // default width of descriptor histogram array
    static const int DESCR_WIDTH = 4;

    // default number of bins per histogram in descriptor array
    static const int DESCR_HIST_BINS = 8;

    public:

    /**
     * Method to set the values to be used by VigraSiftDescriptor
     */
    void setValues(
            int features,
            int octaveLayers,
            double contrastThreshold,
            double edgeThreshold,
            double sigma
    );

    /**
     * Get the size of descriptor.
     */
    int getDescriptorSize();

    /**
     * Allocate and initialize image array.
     */
    void allocateAndInitializeImage(const char*  file_name);

    /**
     * Allocate and intialize descriptor array.
     */
    void allocateDescriptorArray();


};


}



#endif //SIFT_HELPER_VIGRA_H
