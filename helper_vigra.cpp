//
// Created by neuron on 25.04.16.
//
/**
 *
 */

#include "helper_vigra.h"

namespace vigra{

    void VigraSiftDescriptor::setValues(
            int features,
            int octaveLayers,
            double contrastThreshold,
            double edgeThreshold,
            double sigma
    ){
        this->features = features;
        this->octaveLayers = octaveLayers;
        this->contrastThreshold = contrastThreshold;
        this->edgeThreshold = edgeThreshold;
        this->sigma = sigma;
    }

    int VigraSiftDescriptor::getDescriptorSize() {
        return DESCR_WIDTH*DESCR_WIDTH*DESCR_HIST_BINS;
    }
}