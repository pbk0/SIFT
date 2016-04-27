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

    void VigraSiftDescriptor::setKeypoints(std::vector<KeyPoint> key_points){
        this->key_points = key_points;
    }

    void VigraSiftDescriptor::build_gauss_pyr()
    {
        ////// modded
        MultiArray<2, UInt8> const & baseImg =  this->image_array;
        int firstOctave = -1;
        int octaves =
                (int)(std::log(
                        (double)std::min(baseImg.shape(0), baseImg.shape(1))
                ) / (std::log(2.) - 2) - firstOctave);
        int intervals = this->octaveLayers;
        double init_sigma = this->sigma;

        /////
        std::vector<MultiArray<2, UInt8> > gauss_pyr;

        //Initializing scales for all intervals
        double k = pow( 2.0, 1.0 / intervals );
        double sig[intervals+3];

        sig[0] = init_sigma;
        sig[1] = init_sigma * sqrt( k*k- 1 );

        for (int i = 2; i < intervals + 3; i++)
            sig[i] = sig[i-1] * k;


        //Building scale space image pyramid
        for(int o = 0; o < octaves; o++ )
        {
            for(int i = 0; i < intervals + 3 ; i++ )
            {
                if( o == 0  &&  i == 0 ){
                    gauss_pyr.push_back(baseImg);
                }
                else if( i == 0 )
                {
                    //Downsample
                    double factor=0.5;
                    MultiArray<2, UInt8> prev_oct_last =
                            gauss_pyr[(o-1)*(intervals+3)+(intervals-1)];
                    MultiArray<2, UInt8> nxt_octv_base(
                            (int)(factor*prev_oct_last.shape(0)),
                            (int)(factor*prev_oct_last.shape(1))
                    );
                    vigra::resizeImageNoInterpolation(
                            prev_oct_last,
                            nxt_octv_base
                    );
                    gauss_pyr.push_back(nxt_octv_base);
                }
                else
                {
                    //Smooth base image
                    vigra::Kernel2D<double> filter;
                    filter.initGaussian(sig[i]);
                    MultiArray<2, UInt8> prev =
                            gauss_pyr[o*(intervals+3)+(i-1)];
                    MultiArray<2, UInt8> next(prev.shape());
                    vigra::convolveImage(prev, next, filter);
                    gauss_pyr.push_back(next);

                }

            }//endfor_i

        }//endfor_o

        //return gauss_pyr;
        this->gaussian_pyramid = gauss_pyr;
    }
}