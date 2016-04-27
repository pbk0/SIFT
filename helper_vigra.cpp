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

    void VigraSiftDescriptor::allocateAndInitializeImage(const char* file_name){

        ImageImportInfo vigra_img_info(file_name);
        MultiArray<2, vigra::UInt8> vigra_img_array(vigra_img_info.shape());
        importImage(vigra_img_info, vigra_img_array);
        this->image_array = vigra_img_array;
    }

    void VigraSiftDescriptor::allocateDescriptorArray() {

        MultiArray<2, vigra::UInt8> vigra_array(Shape2(
                this->features,
                this->getDescriptorSize()
        ));
        this->descriptor_array = vigra_array;
    }
}