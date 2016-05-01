//
// Created by someone on 25.04.16.
//
/**
 *
 */

#include "vigra_sift.h"
//#include <iostream>
//#include <string.h>
//#include <algorithm>
//#include <math.h>
//#include <vector>
//#include <stdlib.h>
//#include <fstream>

#define _OPENCV 1
#if _OPENCV
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/core/hal/hal.hpp>

#endif

using namespace vigra::multi_math;
using Eigen::Matrix2cd;
using namespace std;

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
        int radius = (int)std::roundf(hist_width * 1.4142135623730951f * (DESCR_WIDTH + 1) * 0.5f);
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


    /*Constructor -initialising SIFT parameters*/
    void VigraSiftDetector::setParameters(
            int intervals=SIFT_INTVLS, float sigma=SIFT_SIGMA,
            float contr_thr=SIFT_CONTR_THR, int curv_thr=SIFT_CURV_THR){
        this->intervals=intervals;
        this->sigma=sigma;
        this->contr_thr=contr_thr;
        this->curv_thr=curv_thr;

        this->hist_orientation.resize(SIFT_ORI_HIST_BINS);
    }

    /*sets Octaves*/
    void VigraSiftDetector::setOctaves(int octaves){
        this->octaves=octaves;
    }

    /**
     *
     */
    void VigraSiftDetector::allocateAndInitializeImage(const char* file_name){

        ImageImportInfo vigra_img_info(file_name);
        MultiArray<2, vigra::UInt8> vigra_img_array(vigra_img_info.shape());
        importImage(vigra_img_info, vigra_img_array);
        this->src_img = vigra_img_array;
    }

    void VigraSiftDetector::build_gauss_pyr()
    {

        //Initializing scales for all intervals
        float k = pow( 2.0f, 1.0f / this->intervals );
        float sig[intervals+3];

        sig[0] = this->sigma;
        for( int i = 1; i < intervals + 3; i++ )
        {
            float sig_prev = std::pow(k, (float)(i-1))*this->sigma;
            float sig_total = sig_prev*k;
            sig[i] = std::sqrt(sig_total*sig_total - sig_prev*sig_prev);
        }

        gauss_pyr.clear();

        //Building scale space image pyramid
        for(int o = 0; o < octaves; o++ )
        {
            for(int i = 0; i < intervals + 3 ; i++ )
            {
                if( o == 0  &&  i == 0 ){
                    //Smooth base image
                    MultiArray<2, UInt8> next(this->src_img.shape());
                    gaussianSmoothing(this->src_img, next, sig[i]);
                    gauss_pyr.push_back(next);

                }
                else if( i == 0 )
                {
                    //Beginning of next octave - Downsample
                    double factor=0.5;
                    MultiArray<2, UInt8> prev_oct_last=gauss_pyr[(o-1)*(intervals+3)+(intervals-1)];
                    MultiArray<2, UInt8> nxt_octv_base((int)(factor*prev_oct_last.shape(0)), (int)(factor*prev_oct_last.shape(1)));
                    vigra::resizeImageNoInterpolation(prev_oct_last, nxt_octv_base);
                    gauss_pyr.push_back(nxt_octv_base);
                }
                else
                {
                    //Smooth base image
                    MultiArray<2, UInt8> prev = gauss_pyr[o*(intervals+3)+(i-1)];
                    MultiArray<2, UInt8> next(prev.shape());
                    gaussianSmoothing(prev, next, sig[i]);
                    gauss_pyr.push_back(next);

                }

            }//endfor_i

        }//endfor_o

    }

    /**
     *
     */
    void VigraSiftDetector::build_dog_pyr()
    {
        for(int o = 0; o < octaves; o++){
            for(int i = 1; i < intervals + 3; i++){
                MultiArray<2, UInt8> curr = gauss_pyr[ o*(intervals+3)+i ];
                MultiArray<2, UInt8> prev = gauss_pyr[ o*(intervals+3)+(i -1) ];
                MultiArray<2, vigra::UInt8> diff(curr.shape());
                for(int row=0;row<curr.shape(0);row++) {
                    for (int col = SIFT_IMG_BORDER;
                         col < curr.shape(1); col++) {
                        diff[Shape2(row, col)] =
                                curr[Shape2(row, col)] - prev[Shape2(row, col)];
                    }
                }
                dog_pyr.push_back(diff);
            }
        }

    }

    /**
     *
     */
    bool VigraSiftDetector::is_extremum( int oc, int intv, int rIdx, int cIdx )
    {

        MultiArray<2, UInt8> curr = dog_pyr[ oc*(intervals+2)+intv ];
        UInt8 val=curr[Shape2(rIdx,cIdx)];

        int i, j, k;

        /* check for maximum */
        if( val > 0 )
        {
            for( i = -1; i <= 1; i++ )
                for( j = -1; j <= 1; j++ )
                    for( k = -1; k <= 1; k++ ){
                        if( val < dog_pyr[ oc*(intervals+2)+intv+i ]
                                  [Shape2(rIdx+j,cIdx+k)] )
                            return false;
                    }
        }

            /* check for minimum */
        else
        {
            for( i = -1; i <= 1; i++ )
                for( j = -1; j <= 1; j++ )
                    for( k = -1; k <= 1; k++ ){
                        if( val >
                                dog_pyr[ oc*(intervals+2)+intv+i ]
                                  [Shape2(rIdx+j,cIdx+k)] )
                            return false;
                    }
        }

        return true;
    }

    Eigen::Vector3d VigraSiftDetector::compute_pderivative(
            int oc, int intv, int rIdx, int cIdx  )
    {
        const float img_scale = 1.0f/(255*SIFT_FIXPT_SCALE);
        const float deriv_scale = img_scale*0.5f;

        MultiArray<2, UInt8> curr = dog_pyr[ oc*(intervals+2)+intv ];
        MultiArray<2, UInt8> next = dog_pyr[ oc*(intervals+2)+intv+1 ];
        MultiArray<2, UInt8> prev = dog_pyr[ oc*(intervals+2)+intv-1 ];
        UInt8 val=curr[Shape2(rIdx,cIdx)];

        double dx, dy, ds;

        dx = ( curr[Shape2(rIdx,cIdx+1)] - curr[Shape2(rIdx,cIdx-1)] )
             *deriv_scale;
        dy = ( curr[Shape2(rIdx+1,cIdx)] - curr[Shape2(rIdx-1,cIdx)] )
             *deriv_scale;
        ds = ( next[Shape2(rIdx,cIdx)] - prev[Shape2(rIdx,cIdx)])*deriv_scale;

        Eigen::Vector3d pderivative(dx,dy,ds);

        return pderivative;

    }

    Eigen::Matrix3d VigraSiftDetector::compute_hessian(
            int oc, int intv, int rIdx, int cIdx  ){

        float img_scale = 1.0f/(255*SIFT_FIXPT_SCALE);
        float deriv_scale = img_scale*0.5f;
        float second_deriv_scale = img_scale;
        float cross_deriv_scale = img_scale*0.25f;

        float val, dxx, dyy, dss, dxy, dxs, dys;
        MultiArray<2, UInt8> curr = dog_pyr[ oc*(intervals+2)+intv ];
        MultiArray<2, UInt8> next = dog_pyr[ oc*(intervals+2)+intv+1 ];
        MultiArray<2, UInt8> prev = dog_pyr[ oc*(intervals+2)+intv-1 ];

        val=curr[Shape2(rIdx,cIdx)];

        dxx = ( curr[Shape2(rIdx,cIdx+1)] + curr[Shape2(rIdx,cIdx-1)] - 2 * val )*second_deriv_scale;
        dyy = ( curr[Shape2(rIdx+1,cIdx)] + curr[Shape2(rIdx-1,cIdx)] - 2 * val )*second_deriv_scale;
        dss = ( next[Shape2(rIdx,cIdx)] + prev[Shape2(rIdx,cIdx)] - 2 * val )*second_deriv_scale;
        dxy = ( curr[Shape2(rIdx+1,cIdx+1)] -
                curr[Shape2(rIdx+1,cIdx-1)] -
                curr[Shape2(rIdx-1,cIdx+1)] +
                curr[Shape2(rIdx-1,cIdx-1)] ) *cross_deriv_scale;
        dxs = ( next[Shape2(rIdx,cIdx+1)] -
                next[Shape2(rIdx,cIdx-1)] -
                prev[Shape2(rIdx,cIdx+1)] +
                prev[Shape2(rIdx,cIdx-1)] ) *cross_deriv_scale;
        dys = ( next[Shape2(rIdx+1,cIdx)] -
                next[Shape2(rIdx-1,cIdx)] -
                prev[Shape2(rIdx+1,cIdx)] +
                prev[Shape2(rIdx-1,cIdx)] ) *cross_deriv_scale;

        Eigen::Matrix3d hess(3,3);
        hess(0,0)=dxx;
        hess(0,1)=dxy;
        hess(0,2)=dxs;
        hess(1,0)=dxy;
        hess(1,1)=dyy;
        hess(1,2)=dys;
        hess(2,0)=dxs;
        hess(2,1)=dys;
        hess(2,2)=dss;

        return hess;

    }

    bool VigraSiftDetector::interpolate_step(
            int oc, int intv, int rIdx, int cIdx, float & xi, float & xr,
            float & xc )
    {
        Eigen::Vector3d derv=compute_pderivative(oc,intv,rIdx,cIdx);
        Eigen::Matrix3d hess=compute_hessian(oc,intv,rIdx,cIdx);

        Eigen::Matrix3d hess_inv;
        Eigen::Vector3d res;

        bool invertible;
        hess.computeInverseWithCheck(hess_inv,invertible);
        if(invertible){
            res= hess_inv * derv;

            xi = -res(2);
            xr = -res(1);
            xc = -res(0);

            return true;
        }
        else{
            return false;
        }

    }

    float VigraSiftDetector::interpolate_contr(
            int oc, int intv, int rIdx, int cIdx, float xi,
            float xr, float xc)
    {
        const float img_scale = 1.0f/(255*SIFT_FIXPT_SCALE);
        Eigen::Vector3d derv=compute_pderivative(oc,intv,rIdx,cIdx);
        Eigen::Vector3d X(xc,xr,xi);
        double res = X.transpose()*derv;

        MultiArray<2, UInt8> curr=dog_pyr[ oc*(intervals+2)+intv ];

        return (float)(curr[Shape2(rIdx,cIdx)]*img_scale + res * 0.5f);

    }

    int VigraSiftDetector::is_too_edge_like(
            MultiArray<2, UInt8> const & dog_img, int rIdx, int cIdx )
    {
        float img_scale = 1.0f/(255*SIFT_FIXPT_SCALE);
        float second_deriv_scale = img_scale;
        float cross_deriv_scale = img_scale*0.25f;

        float d, dxx, dyy, dxy, tr, det;

        /* principal curvatures are computed using the trace
         * and det of Hessian */
        d = dog_img[Shape2(rIdx, cIdx)];
        dxx = (dog_img[Shape2(rIdx, cIdx+1)] +
                dog_img[Shape2(rIdx, cIdx-1)] - 2 * d)*second_deriv_scale;
        dyy = (dog_img[Shape2(rIdx+1, cIdx)] +
                dog_img[Shape2(rIdx-1, cIdx)] - 2 * d)*second_deriv_scale;
        dxy = ( dog_img[Shape2(rIdx+1, cIdx+1)] -
                dog_img[Shape2(rIdx+1, cIdx-1)] -
                dog_img[Shape2(rIdx-1, cIdx+1)] +
                dog_img[Shape2(rIdx-1, cIdx-1)] ) *cross_deriv_scale;
        tr = dxx + dyy;
        det = dxx * dyy - dxy * dxy;

        /* negative determinant -> curvatures have different signs;
         * reject feature */
        if( det <= 0 )
            return 1;

        if( tr * tr / det < ( curv_thr + 1.0 )*( curv_thr + 1.0 ) / curv_thr )
            return 0;

        return 1;
    }

    bool VigraSiftDetector::interpolate_extremum(
            int oc, int intv, int rIdx, int cIdx, vigra::KeyPoint & kp)
    {
        float xi=0.f, xr=0.f, xc=0.f;
        int i = 0;

        while( i < SIFT_MAX_INTERP_STEPS )
        {
            if(!interpolate_step( oc, intv, rIdx, cIdx, xi, xr, xc )){
                return false;
            }
            if( abs( xi ) < 0.5  &&  abs( xr ) < 0.5  &&  abs( xc ) < 0.5 )
                break;

            if( std::abs(xi) > (float)(INT_MAX/3) ||
                std::abs(xr) > (float)(INT_MAX/3) ||
                std::abs(xc) > (float)(INT_MAX/3) ){
                return false;
            }

            cIdx += std::round( xc );
            rIdx += std::round( xr );
            intv += std::round( xi );

            if( intv < 1  ||
                intv > intervals  ||
                cIdx < SIFT_IMG_BORDER  ||
                rIdx < SIFT_IMG_BORDER  ||
                cIdx >= dog_pyr[oc].shape(1) - SIFT_IMG_BORDER  ||
                rIdx >= dog_pyr[oc].shape(0) - SIFT_IMG_BORDER ){
                return false;
            }

            i++;
        }

        /* ensure convergence of interpolation */
        if( i >= SIFT_MAX_INTERP_STEPS ){
            return false;
        }

        // interpolate contrast
        float contr = interpolate_contr( oc, intv, rIdx, cIdx, xi, xr, xc );
        if( abs( contr ) < contr_thr / intervals ){
            return false;
        }

        // reject edge like features
        if(is_too_edge_like(
                dog_pyr[ oc*(intervals+2)+intv ],
                (int)std::round(rIdx+xr),
                (int)std::round(cIdx+xc))){
            return false;
        }
        else
        {

            //Assign keypoint
            kp.ptx = ( rIdx + xr ) * pow( 2.0f, oc );
            kp.pty = ( cIdx + xc ) * pow( 2.0f, oc );
            kp.r = (int)std::round(rIdx + xr);
            kp.c = (int)std::round(cIdx + xc);
            kp.octave = oc ;
            kp.intvl = (int)std::round(intv+xi);
            kp.size = 2 * sigma * pow( 2, (intv + xi)/intervals ) *
                    pow( 2.0f, oc);
            kp.response = std::abs(contr);
            kp.scale_octv = sigma * pow( 2, (intv + xi)/intervals );
        }
        return true;
    }

    float VigraSiftDetector::calculate_orientation_hist(
            MultiArray<2, UInt8> const & img, int rIdx, int cIdx, int nbins,
            int radius, float sigma )
    {
        double PI2 = M_PI * 2.0f;
        float mag, ori, w, exp_denom;
        int bin, i, j;
        exp_denom = 2.0f * sigma * sigma;

        // Loop through neighbouring radius*radius window
        // and build histogram
        for( i = -radius; i <= radius; i++ ){
            for( j = -radius; j <= radius; j++ ){

                int x = rIdx + i;
                int y = cIdx + j;
                if(  x <= 0 || x > img.shape(0) || y <= 0 || y > img.shape(1) )
                    continue;

                //calculate magnitude, orientation, weight
                float dx = img[Shape2(rIdx,cIdx+1)] - img[Shape2(rIdx,cIdx-1)];
                float dy = img[Shape2(rIdx-1,cIdx)] - img[Shape2(rIdx+1,cIdx)];
                mag = sqrt( dx*dx + dy*dy );
                ori = atan2( dy, dx );
                w = exp( -( i*i + j*j ) / exp_denom );

                //update histogram
                bin = (int)std::round( nbins * (ori+ M_PI) / PI2 );
                bin = ( bin < nbins )? bin : 0;
                hist_orientation[bin] += w * mag;
            }
        }

        // Smooth histogram
        for( j = 0; j < SIFT_ORI_SMOOTH_PASSES; j++ )
            smooth_ori_hist();

        //Determine dominant orientation
        //float maxval=*std::max_element(hist.begin(),hist.end());
        float max_val = -999999999999999;
        for (int ii = 0; ii < hist_orientation.size(); ii++){
            if (max_val < hist_orientation[(ii)]){
                max_val=hist_orientation[(ii)];
            }
        }

        return max_val;
    }

    void VigraSiftDetector::smooth_ori_hist()
    {
        float prev, tmp, h0 = this->hist_orientation[0];
        int i;
        int nbins=SIFT_ORI_HIST_BINS;

        prev = this->hist_orientation[(nbins-1)];
        for( i = 0; i < nbins; i++ )
        {
            tmp = hist_orientation[(i)];
            hist_orientation[(i)] =
                    0.25f * prev + 0.5f * hist_orientation[(i)] + 0.25f *
            ( ( i+1 == nbins )? h0 : hist_orientation[(i+1)]);
            prev = tmp;
        }
    }


    void VigraSiftDetector::detect_extrema()
    {
        int _cnt = 0;
        float prelim_contr_thr =  0.5f * contr_thr / intervals *
                255 * SIFT_FIXPT_SCALE;
        vigra::KeyPoint kpt;

        for(int o = 0; o < octaves; o++){
            for(int i = 1; i < intervals ; i++){

                MultiArray<2, UInt8> curr = dog_pyr[ o*(intervals+2)+i ];

                std::cout<<"Octave: "<<o<<" , Interval: "<<i<<std::endl;

                for(int row=SIFT_IMG_BORDER;
                    row<curr.shape(0)-SIFT_IMG_BORDER;
                    row++){
                    for(int col=SIFT_IMG_BORDER;
                        col<curr.shape(1)-SIFT_IMG_BORDER;
                        col++)
                    {

                        if(abs(curr[Shape2(row,col)]) > prelim_contr_thr &&
                                is_extremum(o,i,row,col))
                        {
                            //keypoint elimination
                            if(interpolate_extremum(o,i,row,col,kpt)){

                                //calculate orientation and add keypoint
                                int nbins=SIFT_ORI_HIST_BINS;

                                //calculate dominant orientation
                                float max_mag =
                                        calculate_orientation_hist(
                                                curr,kpt.r,kpt.c,nbins,
                                                (int)std::round(
                                                        SIFT_ORI_RADIUS *
                                                                kpt.scale_octv
                                                ),(SIFT_ORI_SIG_FCTR *
                                                        kpt.scale_octv));
                                float mag_thresh =
                                        (float)(max_mag * SIFT_ORI_PEAK_RATIO);

                                //Detect dominant and secondary orientations
                                // and add keypoints
                                int cccccc = 0;
                                for( int j = 0; j < nbins; j++ )
                                {
                                    int l = j > 0 ? j - 1 : nbins - 1;
                                    int r2 = j < nbins-1 ? j + 1 : 0;
                                    double PI2 = M_PI * 2.0;

                                    if( hist_orientation[j] >
                                                hist_orientation[l]  &&
                                            hist_orientation[j] >
                                                    hist_orientation[r2]  &&
                                            hist_orientation[j] >= mag_thresh )
                                    {
                                        float bin = j + 0.5f *
                                                        (hist_orientation[l]-
                                                         hist_orientation[r2]) /
                                                        (hist_orientation[l] -
                                                         2*hist_orientation[j] +
                                                         hist_orientation[r2]);
                                        bin = bin < 0 ? nbins + bin :
                                              bin >= nbins ? bin - nbins : bin;
                                        kpt.angle=(float)((( PI2 * bin )
                                                          / nbins ) -
                                                  M_PI);
                                        keypoints.push_back(kpt);
                                        cccccc++;
                                        if(cccccc>3){
                                            cout << "hhhhhh" << endl;
                                        }
                                    }
                                }//endfor_j
                            }//endif_interpolate_extremum
                            else{
                                _cnt++;
                            }
                        }//endif_extremum
                    }//endfor_col
                }//endfor_row
            }//endfor_i
        }//endfor_o
        int num_keypoints=(int)keypoints.size();
        std::cout<<std::endl<<"Number of keypoints detected: "
        <<num_keypoints<<std::endl;
        std::cout << "rejected " << _cnt << std::endl;
    }

    std::vector<vigra::KeyPoint> VigraSiftDetector::detect_keypoints(){


        int octv = log( std::min( src_img.shape(0), src_img.shape(1) ) )
                   / log(2) - 3;
        setOctaves(octv);
        std::cout<<"Octaves: "<<octv<<std::endl;

        build_gauss_pyr();
        std::cout<<"Gaussian Pyramid built"<<std::endl;

        build_dog_pyr();
        std::cout<<"DOG Pyramid built"<<std::endl;

        detect_extrema();
        std::cout<<"Keypoints computed"<<std::endl;

        return this->keypoints;
    }

}