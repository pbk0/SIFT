//
// Created by neuron on 25.04.16.
//

#ifndef SIFT_HELPER_VIGRA_H
#define SIFT_HELPER_VIGRA_H

#include <vigra/impex.hxx>
#include <vigra/multi_fft.hxx>
#include <vigra/convolution.hxx>
#include <vigra/gaborfilter.hxx>
#include <vigra/edgedetection.hxx>
#include <vigra/hdf5impex.hxx>

namespace vigra
{

class VigraSiftDescriptor
{
    int features;
    int octaveLayers;
    double contrastThreshold;
    double edgeThreshold;
    double sigma;

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


};


}



#endif //SIFT_HELPER_VIGRA_H
