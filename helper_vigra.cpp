//
// Created by neuron on 25.04.16.
//
/**
 *
 */

#include "helper_vigra.h"


#define _OPENCV 1
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
#include <opencv2/core/hal/hal.hpp>

#endif

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
        std::cout << "build gaussian pyramid ..." <<std::endl;
        ////// modded
        MultiArray<2, UInt8> const & baseImg =  this->image_array;
        //int firstOctave = -1;
        int firstOctave = 0;
        int octaves =
                (int)(std::log((double)std::min(baseImg.shape(0), baseImg.shape(1)) / std::log(2.) - 2)) - firstOctave;
        int intervals = this->octaveLayers;
        double init_sigma = this->sigma;

        /////
        //std::vector<MultiArray<2, UInt8> > gauss_pyr;

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
                //std::cout << "octave " << o << " layer " << i << std::endl;
                if( o == 0  &&  i == 0 ){
                    this->gaussian_pyramid.push_back(baseImg);
                }
                else if( i == 0 )
                {
                    //Downsample
                    double factor=0.5;
                    MultiArray<2, UInt8> prev_oct_last =
                            this->gaussian_pyramid[(o-1)*(intervals+3)+(intervals-1)];
                    MultiArray<2, UInt8> nxt_octv_base(
                            (int)(factor*prev_oct_last.shape(0)),
                            (int)(factor*prev_oct_last.shape(1))
                    );
                    vigra::resizeImageNoInterpolation(
                            prev_oct_last,
                            nxt_octv_base
                    );
                    this->gaussian_pyramid.push_back(nxt_octv_base);
                }
                else
                {
                    //Smooth base image
                    vigra::Kernel2D<double> filter;
                    filter.initGaussian(sig[i]);
                    MultiArray<2, UInt8> prev =
                            this->gaussian_pyramid[o*(intervals+3)+(i-1)];
                    MultiArray<2, UInt8> next(prev.shape());
                    vigra::convolveImage(prev, next, filter);
                    this->gaussian_pyramid.push_back(next);

                }

            }//endfor_i

        }//endfor_o

        //return gauss_pyr;
        //this->gaussian_pyramid = gauss_pyr;
        int i = 0;
        for(auto const& img: this->gaussian_pyramid) {
            //std::cout << "image " << i << std::endl;
            //std::cout << (int)this->gaussian_pyramid[i][Shape2(1,2)] << std::endl;
            i++;
        }
        std::cout << "finished building gaussian pyramid ..." <<std::endl;
    }

    float* VigraSiftDescriptor::calculate_descriptors_helper(
            const MultiArray<2, vigra::UInt8> img,
            float ptx,
            float pty,
            float orientation,
            float size
    )
    {
        int pt_x = (int)std::roundf(ptx);
        int pt_y = (int)std::roundf(pty);

        float cos_t = cosf(orientation*(float)(PI/180));
        float sin_t = sinf(orientation*(float)(PI/180));
        float bins_per_rad = DESCR_HIST_BINS / 360.f;
        float exp_scale = -1.f/(DESCR_WIDTH * DESCR_WIDTH * 0.5f);
        float hist_width = DESCR_SCL_FCTR * size;
        int radius = cvRound(hist_width * 1.4142135623730951f * (DESCR_WIDTH + 1) * 0.5f);
        // Clip the radius to the diagonal of the image to avoid autobuffer too large exception
        radius = std::min(radius, (int) sqrt((double) img.shape(0)*img.shape(0) + img.shape(1)*img.shape(1)));
        cos_t /= hist_width;
        sin_t /= hist_width;

        int i, j, k, len = (radius*2+1)*(radius*2+1), histlen = (DESCR_WIDTH+2)*(DESCR_WIDTH+2)*(DESCR_HIST_BINS+2);
        int rows = (int)img.shape(1), cols = (int)img.shape(0);

        cv::AutoBuffer<float> buf(size_t(len*6 + histlen));
        float *X = buf, *Y = X + len, *Mag = Y, *Ori = Mag + len, *W = Ori + len;
        float *RBin = W + len, *CBin = RBin + len, *hist = CBin + len;

        for( i = 0; i < DESCR_WIDTH+2; i++ )
        {
            for( j = 0; j < DESCR_WIDTH+2; j++ )
                for( k = 0; k < DESCR_HIST_BINS+2; k++ )
                    hist[(i*(DESCR_WIDTH+2) + j)*(DESCR_HIST_BINS+2) + k] = 0.f;
        }

        for( i = -radius, k = 0; i <= radius; i++ )
            for( j = -radius; j <= radius; j++ )
            {
                // Calculate sample's histogram array coords rotated relative to orientation.
                // Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
                // r_rot = 1.5) have full weight placed in row 1 after interpolation.
                float c_rot = j * cos_t - i * sin_t;
                float r_rot = j * sin_t + i * cos_t;
                float rbin = r_rot + DESCR_WIDTH/2 - 0.5f;
                float cbin = c_rot + DESCR_WIDTH/2 - 0.5f;
                int r = pt_y + i, c = pt_x + j;

                if( rbin > -1 && rbin < DESCR_WIDTH && cbin > -1 && cbin < DESCR_WIDTH &&
                    r > 0 && r < rows - 1 && c > 0 && c < cols - 1 )
                {
                    float dx = (float)(img[Shape2(r, c+1)] - img[Shape2(r, c-1)]);
                    float dy = (float)(img[Shape2(r-1, c)] - img[Shape2(r+1, c)]);
                    X[k] = dx; Y[k] = dy; RBin[k] = rbin; CBin[k] = cbin;
                    W[k] = (c_rot * c_rot + r_rot * r_rot)*exp_scale;
                    k++;
                }
            }

        std::cout<<"pkpkpkpk vigra hack \t "<< k << " \t"<<cos_t<<" \t" <<sin_t <<" \t" << radius << " \t" << rows << " \t" << cols <<std::endl;


        len = k;
        cv::hal::fastAtan2(Y, X, Ori, len, true);
        cv::hal::magnitude32f(X, Y, Mag, len);
        cv::hal::exp32f(W, W, len);

        for( k = 0; k < len; k++ )
        {
            float rbin = RBin[k], cbin = CBin[k];
            float obin = (Ori[k] - orientation)*bins_per_rad;
            float mag = Mag[k]*W[k];

            int r0 = cvFloor( rbin );
            int c0 = cvFloor( cbin );
            int o0 = cvFloor( obin );
            rbin -= r0;
            cbin -= c0;
            obin -= o0;

            if( o0 < 0 )
                o0 += DESCR_HIST_BINS;
            if( o0 >= DESCR_HIST_BINS )
                o0 -= DESCR_HIST_BINS;

            // histogram update using tri-linear interpolation
            float v_r1 = mag*rbin, v_r0 = mag - v_r1;
            float v_rc11 = v_r1*cbin, v_rc10 = v_r1 - v_rc11;
            float v_rc01 = v_r0*cbin, v_rc00 = v_r0 - v_rc01;
            float v_rco111 = v_rc11*obin, v_rco110 = v_rc11 - v_rco111;
            float v_rco101 = v_rc10*obin, v_rco100 = v_rc10 - v_rco101;
            float v_rco011 = v_rc01*obin, v_rco010 = v_rc01 - v_rco011;
            float v_rco001 = v_rc00*obin, v_rco000 = v_rc00 - v_rco001;

            int idx = ((r0+1)*(DESCR_WIDTH+2) + c0+1)*(DESCR_HIST_BINS+2) + o0;
            hist[idx] += v_rco000;
            hist[idx+1] += v_rco001;
            hist[idx+(DESCR_HIST_BINS+2)] += v_rco010;
            hist[idx+(DESCR_HIST_BINS+3)] += v_rco011;
            hist[idx+(DESCR_WIDTH+2)*(DESCR_HIST_BINS+2)] += v_rco100;
            hist[idx+(DESCR_WIDTH+2)*(DESCR_HIST_BINS+2)+1] += v_rco101;
            hist[idx+(DESCR_WIDTH+3)*(DESCR_HIST_BINS+2)] += v_rco110;
            hist[idx+(DESCR_WIDTH+3)*(DESCR_HIST_BINS+2)+1] += v_rco111;
        }

        // finalize histogram, since the orientationentation histograms are circular
        float* dst = new float[this->getDescriptorSize()];
        for( i = 0; i < DESCR_WIDTH; i++ )
            for( j = 0; j < DESCR_WIDTH; j++ )
            {
                int idx = ((i+1)*(DESCR_WIDTH+2) + (j+1))*(DESCR_HIST_BINS+2);
                hist[idx] += hist[idx+DESCR_HIST_BINS];
                hist[idx+1] += hist[idx+DESCR_HIST_BINS+1];
                for( k = 0; k < DESCR_HIST_BINS; k++ )
                    dst[(i*DESCR_WIDTH + j)*DESCR_HIST_BINS + k] = hist[idx+k];
            }
        // copy histogram to the descriptor,
        // apply hysteresis thresholding
        // and scale the result, so that it can be easily converted
        // to byte array
        float nrm2 = 0;
        len = DESCR_WIDTH*DESCR_WIDTH*DESCR_HIST_BINS;
        for( k = 0; k < len; k++ )
            nrm2 += dst[k]*dst[k];
        float thr = std::sqrt(nrm2)*DESCR_MAG_THR;
        for( i = 0, nrm2 = 0; i < k; i++ )
        {
            float val = std::min(dst[i], thr);
            dst[i] = val;
            nrm2 += val*val;
        }
        nrm2 = INT_DESCR_FCTR/std::max(std::sqrt(nrm2), FLT_EPSILON);

        for( k = 0; k < len; k++ )
        {
            //dst[k] = cv::saturate_cast<uchar>(dst[k]*nrm2);
            dst[k] = dst[k]*nrm2;
        }


        return dst;
    }

    float*  VigraSiftDescriptor::calculate_descriptors(int keypoint_id) {

        std::cout << "calculation descriptors ..." <<std::endl;

        vigra::KeyPoint kp = this->key_points[keypoint_id];

        // get the octave details
        int octave = kp.octave & 255;
        int layer = (kp.octave >> 8) & 255;
        octave = octave < 128 ? octave : (-128 | octave);
        float scale =
                octave >= 0 ? 1.f/(1 << octave) : (float)(1 << -octave);

        // get the image scale correctly
        float size = kp.size*scale;
        float ptx = kp.ptx*scale;
        float pty = kp.pty*scale;
        int gau_pyr_index = (octave)*(this->octaveLayers+3)+layer;
        MultiArray<2, vigra::UInt8> &curr_img =
                this->gaussian_pyramid[gau_pyr_index];

        //std::cout << curr_img[Shape2(1,2)] << std::endl;

        // get the angle
        float angle = 360.f - kp.angle;
        if(std::abs(angle - 360.f) < FLT_EPSILON)
            angle = 0.f;

        /////


        float* ret_float = this->calculate_descriptors_helper(curr_img, ptx, pty, angle, size*0.5f);

        return ret_float;



        //calcSIFTDescriptor(img, ptf, angle, size*0.5f, d, n, descriptors.ptr<float>((int)i));

    }




}