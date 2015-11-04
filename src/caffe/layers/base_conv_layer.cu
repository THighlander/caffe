#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/fft/CuFFTWrapper.cuh"
#include "caffe/fft/FBFFTHost.h"
//#include "caffe/cuda/fbfft/FBFFTCommon.cuh"
//#include "caffe/cuda/fbfft/FBFFT.h"
//#include "caffe/cuda/fbfft/FBFFT2D-inl.cuh"
//#include "caffe/cuda/fbfft/FBIFFT2D-inl.cuh"
//#include "caffe/cuda/fbfft/FBFFTCommon.cuh"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>

namespace caffe {

__global__ void hadamard_forward_with_sum(int n, float* image, float* kernel, float* output, int fftSize, 
                                          int out_channels, int in_channels, int numDivisions)
{
  int index = 0;
  int which_out_channel = 0;
  int which_division = 0;
  int k_index, im_index;


  CUDA_KERNEL_LOOP(j, n)
  {
    //The 2* is due to the complex numbers being real, imaginary for each value
    output[2*j] = output[2*j+1] = 0;
    which_out_channel = j/(fftSize*fftSize*numDivisions*numDivisions);
    which_division = (j/(fftSize*fftSize))%(numDivisions*numDivisions);
    index = j%(fftSize*fftSize);


    for (int in = 0; in <= in_channels/5; ++in)
    {


      k_index = 2*(which_out_channel*in_channels*fftSize*fftSize + 
                   5*in*fftSize*fftSize +
                   index);

      im_index = 2*(5*in*fftSize*fftSize*numDivisions*numDivisions +
                    which_division*fftSize*fftSize +
                    index);

      //float test = image[im_index+im_complexOffset];
      //printf("Index: %d Value: %f\n", ( im_index+im_complexOffset), test);
      
      if(5*in < in_channels)
      {
        //printf("%din: \n", in);
        output[2*j] += ((image[im_index] * kernel[k_index]) - (image[im_index+1] * kernel[k_index+1]));
        output[2*j+1] += ((image[im_index] * kernel[k_index+1]) + (image[im_index+1] * kernel[k_index]));
      }
      else
       break;
      
      if(5*in+1 < in_channels)
      {
        output[2*j] += ((image[im_index+(2*(1)*fftSize*fftSize*numDivisions*numDivisions)] * kernel[k_index+(2*(1)*fftSize*fftSize)]) - (image[im_index+(2*(1)*fftSize*fftSize*numDivisions*numDivisions)+1] * kernel[k_index+(2*(1)*fftSize*fftSize)+1]));
        output[2*j+1] += ((image[im_index+(2*(1)*fftSize*fftSize*numDivisions*numDivisions)] * kernel[k_index+(2*(1)*fftSize*fftSize)+1]) + (image[im_index+(2*(1)*fftSize*fftSize*numDivisions*numDivisions)+1] * kernel[k_index+(2*(1)*fftSize*fftSize)]));
      }
      else
      {
        break;
      }
      if(5*in+2 < in_channels)
      {
        output[2*j] += ((image[im_index+(2*(2)*fftSize*fftSize*numDivisions*numDivisions)] * kernel[k_index+(2*(2)*fftSize*fftSize)]) - (image[im_index+(2*(2)*fftSize*fftSize*numDivisions*numDivisions)+1] * kernel[k_index+(2*(2)*fftSize*fftSize)+1]));
        output[2*j+1] += ((image[im_index+(2*(2)*fftSize*fftSize*numDivisions*numDivisions)] * kernel[k_index+(2*(2)*fftSize*fftSize)+1]) + (image[im_index+(2*(2)*fftSize*fftSize*numDivisions*numDivisions)+1] * kernel[k_index+(2*(2)*fftSize*fftSize)]));
      }
      else
        break; 
      if(5*in+3 < in_channels)
      {
        output[2*j] += ((image[im_index+(2*(3)*fftSize*fftSize*numDivisions*numDivisions)] * kernel[k_index+(2*(3)*fftSize*fftSize)]) - (image[im_index+(2*(3)*fftSize*fftSize*numDivisions*numDivisions)+1] * kernel[k_index+(2*(3)*fftSize*fftSize)+1]));
        output[2*j+1] += ((image[im_index+(2*(3)*fftSize*fftSize*numDivisions*numDivisions)] * kernel[k_index+(2*(3)*fftSize*fftSize)+1]) + (image[im_index+(2*(3)*fftSize*fftSize*numDivisions*numDivisions)+1] * kernel[k_index+(2*(3)*fftSize*fftSize)]));
      }
      else
          break;
      if(5*in+4 < in_channels)
      {
        output[2*j] += ((image[im_index+(2*(4)*fftSize*fftSize*numDivisions*numDivisions)] * kernel[k_index+(2*(4)*fftSize*fftSize)]) - (image[im_index+(2*(4)*fftSize*fftSize*numDivisions*numDivisions)+1] * kernel[k_index+(2*(4)*fftSize*fftSize)+1]));
        output[2*j+1] += ((image[im_index+(2*(4)*fftSize*fftSize*numDivisions*numDivisions)] * kernel[k_index+(2*(4)*fftSize*fftSize)+1]) + (image[im_index+(2*(4)*fftSize*fftSize*numDivisions*numDivisions)+1] * kernel[k_index+(2*(4)*fftSize*fftSize)]));
      }
      else
        break;
       
    }

  }
}

__global__ void hadamard_forward(int n, float* image, float* kernel, float* output, int fftSize, 
                                          int out_channels, int in_channels, int numDivisions)
{
  int index = 0;
  int which_out_channel = 0;
  int which_in_channel = 0;
  int which_division = 0;
  int k_index, im_index;

  CUDA_KERNEL_LOOP(j, n)
  {
    which_out_channel = (j/(fftSize*fftSize*numDivisions*numDivisions))%(out_channels);
    which_in_channel = (j/(fftSize*fftSize*numDivisions*numDivisions*out_channels));
    which_division = (j/(fftSize*fftSize))%(numDivisions*numDivisions);
    index = j%(fftSize*fftSize);

    im_index = 2*(which_in_channel*fftSize*fftSize*numDivisions*numDivisions +
                    which_division*fftSize*fftSize +
                    index);

    k_index = 2*(which_out_channel*in_channels*fftSize*fftSize + 
                   which_in_channel*fftSize*fftSize +
                   index);

    output[2*j] = ((image[im_index] * kernel[k_index]) - (image[im_index+1] * kernel[k_index+1]));
    output[(2*j)+1] = ((image[im_index] * kernel[k_index+1]) + (image[im_index+1] * kernel[k_index]));
  }
}

__global__ void lookSee (int n, float* array, float scaler)
{
  for (int i = 0; i < n; ++i)
  {
    printf("%f\n", array[i]/scaler);

  }
}


/* CuFFT Hadamard
__global__ void hadamard_forward_with_sum(int n, cufftComplex* image, cufftComplex* kernel, cufftComplex* output, int fftSize, int out_channels, int in_channels)
{
  int index = 0;
  int which_out_channel = 0;
  int k_index, im_index;
  //printf("Hadamard");

  CUDA_KERNEL_LOOP(j, n)
  {
    which_out_channel = j/(fftSize*fftSize);
    index = j%(fftSize*fftSize);

    for (int in = 0; in < in_channels; ++in)
    {
      k_index = index+(in*fftSize*fftSize) + which_out_channel*in_channels*fftSize*fftSize;
      im_index = index+(in*fftSize*fftSize);

      output[j].x += ((image[im_index].x * kernel[k_index].x) - (image[im_index].y * kernel[k_index].y));
      output[j].y += ((image[im_index].x * kernel[k_index].y) + (image[im_index].y * kernel[k_index].x));
    }
  }
}
*/

template <typename Dtype>
__global__ void overlap_and_crop(int n, float* input, Dtype* output, int fftSize, int overlapedSize, 
                                  int kernelSize, int numDivisions, int extraInfo, int outSize, float scaler)
{
  int i, j;
  //FFT size includes extra padding because of 'nice transforms'
  int checkFFTsize = 2 * kernelSize -1;
  int addedI, addedJ, currentIDivision, currentJDivision, whichChannel, startIDiv, startJDiv;
  bool east, south;

  startIDiv = startJDiv = currentIDivision = currentJDivision = addedI = addedJ = 0;

  //loop goes through the overlaped output size


  CUDA_KERNEL_LOOP(loop,n)
  {
    i = (loop%outSize)+extraInfo;
    j = ((loop/outSize)%outSize)+extraInfo;
    whichChannel = loop/(outSize*outSize);

    addedI = i%kernelSize;
    addedJ = j%kernelSize;
    currentIDivision = i/kernelSize;
    currentJDivision = j/kernelSize;

    if(i/checkFFTsize > 0)
    {
      startIDiv = (i+kernelSize-1)/checkFFTsize;
    }
    else
    {
      startIDiv = 0;
    }

    if(j/checkFFTsize > 0)
    {
      startJDiv = (j+kernelSize-1)/checkFFTsize;
    }
    else
    {
      startJDiv = 0;
    }

    //Set intial output value
    output[(whichChannel*outSize*outSize) + (j-extraInfo)*outSize +(i-extraInfo)] = 
                                input[(whichChannel*fftSize*fftSize*numDivisions*numDivisions) +
                                      (startJDiv*fftSize*fftSize*numDivisions) + 
                                      (startIDiv*fftSize*fftSize) +
                                      ((j-startJDiv*kernelSize)*fftSize) + 
                                      (i-startIDiv*kernelSize)]/scaler;

    east = south = false;

    //Check if there is an overlap in the i direction (x)
    if( !(i<kernelSize || overlapedSize - i <= kernelSize)  && i%kernelSize < kernelSize-1)
    {
      east = true;
    }
    
    //Check if there is an overlap in the j direction (x)
    if(!(j<kernelSize || overlapedSize - j <= kernelSize)  && j%kernelSize < kernelSize-1)
    {
      south = true;
    }

    if(east && south)
    {
      output[(whichChannel*outSize*outSize) + (j-extraInfo)*outSize +(i-extraInfo)] += 
                                input[(whichChannel*fftSize*fftSize*numDivisions*numDivisions) +
                                      (currentJDivision*fftSize*fftSize*numDivisions) + 
                                      (currentIDivision*fftSize*fftSize) +
                                      (addedJ* fftSize) + 
                                      addedI]/scaler;
      
      output[(whichChannel*outSize*outSize) + (j-extraInfo)*outSize +(i-extraInfo)] += 
                                input[(whichChannel*fftSize*fftSize*numDivisions*numDivisions) +
                                      ((currentJDivision-1)*fftSize*fftSize*numDivisions) + 
                                      (currentIDivision*fftSize*fftSize) +
                                      ((addedJ+kernelSize)* fftSize) + 
                                      addedI]/scaler;
      
      output[(whichChannel*outSize*outSize) + (j-extraInfo)*outSize +(i-extraInfo)] +=
                                input[(whichChannel*fftSize*fftSize*numDivisions*numDivisions) +
                                      (currentJDivision*fftSize*fftSize*numDivisions) + 
                                      ((currentIDivision-1)*fftSize*fftSize) +
                                      (addedJ* fftSize) + 
                                      (addedI+kernelSize)]/scaler;
                                      
    }
    else if(east || south)
    {
      output[(whichChannel*outSize*outSize) + (j-extraInfo)*outSize +(i-extraInfo)] += 
                                input[(whichChannel*fftSize*fftSize*numDivisions*numDivisions) +
                                      (currentJDivision*fftSize*fftSize*numDivisions) +
                                      (currentIDivision*fftSize*fftSize) +
                                      (addedJ* fftSize) + 
                                      addedI]/scaler;
    }
    
  }
}




template <typename Dtype>
__global__ void crop_ouput(int n, float* fft_output, Dtype* output, int length, int fftSize, int offset, int out_channels)
{
  //printf("crop");
  fft_output += offset*fftSize;
  int size = length * length;
  CUDA_KERNEL_LOOP(index, n)
  {
    output[index] = fft_output[((index/size)*fftSize*fftSize) + 
                                (((index/length)%length)*fftSize) + 
                                offset + 
                                (index%length)]/((float)(fftSize*fftSize));
  }
}


template <typename Dtype>
void BaseConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  // Configure the kernel size, padding, stride, and inputs.
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  CHECK(!conv_param.has_kernel_size() !=
      !(conv_param.has_kernel_h() && conv_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
  CHECK(conv_param.has_kernel_size() ||
      (conv_param.has_kernel_h() && conv_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  CHECK((!conv_param.has_pad() && conv_param.has_pad_h()
      && conv_param.has_pad_w())
      || (!conv_param.has_pad_h() && !conv_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!conv_param.has_stride() && conv_param.has_stride_h()
      && conv_param.has_stride_w())
      || (!conv_param.has_stride_h() && !conv_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  if (conv_param.has_kernel_size()) {
    kernel_h_ = kernel_w_ = conv_param.kernel_size();
  } else {
    kernel_h_ = conv_param.kernel_h();
    kernel_w_ = conv_param.kernel_w();
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!conv_param.has_pad_h()) {
    pad_h_ = pad_w_ = conv_param.pad();
  } else {
    pad_h_ = conv_param.pad_h();
    pad_w_ = conv_param.pad_w();
  }
  if (!conv_param.has_stride_h()) {
    stride_h_ = stride_w_ = conv_param.stride();
  } else {
    stride_h_ = conv_param.stride_h();
    stride_w_ = conv_param.stride_w();
  }
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  is_1x1_ = kernel_w_ == 1 && kernel_h_ == 1
      && stride_h_ == 1 && stride_w_ == 1 && pad_h_ == 0 && pad_w_ == 0;
  // Configure output channels and groups.
  channels_ = bottom[0]->channels();
  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.convolution_param().group();
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
  if (reverse_dimensions()) {
    conv_out_channels_ = channels_;
    conv_in_channels_ = num_output_;
  } else {
    conv_out_channels_ = num_output_;
    conv_in_channels_ = channels_;
  }
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(
        conv_out_channels_, conv_in_channels_ / group_, kernel_h_, kernel_w_));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      vector<int> bias_shape(1, num_output_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  num_ = bottom[0]->num();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  CHECK_EQ(bottom[0]->channels(), channels_) << "Input size incompatible with"
    " convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK_EQ(num_, bottom[bottom_id]->num()) << "Inputs must have same num.";
    CHECK_EQ(channels_, bottom[bottom_id]->channels())
        << "Inputs must have same channels.";
    CHECK_EQ(height_, bottom[bottom_id]->height())
        << "Inputs must have same height.";
    CHECK_EQ(width_, bottom[bottom_id]->width())
        << "Inputs must have same width.";
  }
  // Shape the tops.
  string currentType = this->type();
  compute_output_shape();
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
  }
  if (reverse_dimensions()) {
    conv_in_height_ = height_out_;
    conv_in_width_ = width_out_;
    conv_out_spatial_dim_ = height_ * width_;
  } else {
    conv_in_height_ = height_;
    conv_in_width_ = width_;
    conv_out_spatial_dim_ = height_out_ * width_out_;

    //Check the FFT size and make it a nice size, must handle proper padding and divisions still
    
    if(currentType.compare("ConvolutionOaA") == 0)
    {
      fftSize_ = kernel_h_+ kernel_h_ - 1;

      if(fftSize_ < 4)
      {
        fftSize_=4;
      }
      else if (fftSize_ < 8)
      {
        fftSize_=8;
      }
      else if (fftSize_ < 16)
      {
        fftSize_ = 16;
      }
      else if (fftSize_ < 32)
      {
        fftSize_ = 32;
      }
      else if (fftSize_ < 64)
      {
        fftSize_ = 64;
      }
    }

  }

  kernel_dim_ = conv_in_channels_ * kernel_h_ * kernel_w_;
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_ / group_;
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_ / group_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;

  int numDivisions = conv_in_height_/kernel_h_;

  if(conv_in_height_/kernel_h_ != 0)
    numDivisions++;

  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  if (reverse_dimensions()) {
    col_buffer_.Reshape(1, kernel_dim_, height_, width_);

  } else {
    col_buffer_.Reshape(1, kernel_dim_, height_out_, width_out_);
    if(currentType.compare("ConvolutionOaA") == 0)
    {
      w_pad_buffer_.Reshape(1, conv_out_channels_*conv_in_channels_*numDivisions*numDivisions, fftSize_, fftSize_);
      im_pad_buffer_.Reshape(1, conv_in_channels_*numDivisions*numDivisions, fftSize_, fftSize_);
      o_pad_buffer_.Reshape(1, conv_out_channels_*numDivisions*numDivisions, fftSize_, fftSize_);
    }
  }
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, height_out_ * width_out_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }
  if(currentType.compare("ConvolutionOaA") == 0)
  {
    im_Complex = thrust::device_vector<float>(fftSize_*fftSize_*conv_in_channels_*numDivisions*numDivisions*2);
    w_Complex = thrust::device_vector<float>(fftSize_*fftSize_*conv_in_channels_*conv_out_channels_*2);
    o_Complex = thrust::device_vector<float>(fftSize_*fftSize_*conv_out_channels_*numDivisions*numDivisions*2);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;

  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    }
    col_buff = col_buffer_.cpu_data();
  }
  
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_ / group_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
  }

}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_bias(Dtype* output,
    const Dtype* bias) {

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      height_out_ * width_out_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
      (Dtype)1., output);

  //for teting purposes only
  //cufftDestroy(planR2CInDepth_);
  //cufftDestroy(planR2CNumK_);
  //cufftDestroy(planC2Rout_);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_cpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_ / group_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_cpu(col_buff, input);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_cpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) 
{
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_ / group_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, height_out_ * width_out_, 1.,
      input, bias_multiplier_.cpu_data(), 1., bias);
}

#ifndef CPU_ONLY

//Traditional Caffe
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {

  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    }
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_ / group_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
  }

}

//Convolutional OaA Forward Prop.
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_gemm_oaa(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {

int numDivisions = conv_in_height_/kernel_h_;

if(conv_in_height_%kernel_h_ != 0)
  numDivisions++;


const int in_sizes[3] = {conv_in_channels_*numDivisions*numDivisions, fftSize_,fftSize_};
const int c_in_sizes[4] = {conv_in_channels_*numDivisions*numDivisions, fftSize_,fftSize_, 2};
const int out_sizes[3] = {conv_out_channels_*numDivisions*numDivisions, fftSize_,fftSize_};
const int c_out_sizes[4] = {conv_out_channels_*numDivisions*numDivisions, fftSize_,fftSize_,2};
const int w_sizes[3] = {conv_out_channels_*conv_in_channels_,fftSize_,fftSize_};
const int c_w_sizes[4] = {conv_out_channels_*conv_in_channels_,fftSize_,fftSize_,2};

//Im pad buffer must be made larger
//Pad the input image
float* im_pad_buff;
conv_pad_im_gpu(input, numDivisions, im_pad_buffer_.mutable_gpu_data());
im_pad_buff = im_pad_buffer_.mutable_gpu_data();

float* w_pad_buff;
conv_pad_w_gpu(weights, w_pad_buffer_.mutable_gpu_data());
w_pad_buff = w_pad_buffer_.mutable_gpu_data();


//device tensor(data, size)
facebook::cuda::DeviceTensor<float, 3> im_space(im_pad_buff, in_sizes);
facebook::cuda::DeviceTensor<float, 4> im_complex(thrust::raw_pointer_cast(&im_Complex[0]), c_in_sizes);
//It should not use the buffer yet. We will see if we ever need it. 
//If we do the buffer should be given a size and some memory
facebook::cuda::DeviceTensor<float, 4>* buffer;
facebook::cuda::DeviceTensor<float, 3> w_space(w_pad_buff, w_sizes);
facebook::cuda::DeviceTensor<float, 4> w_complex(thrust::raw_pointer_cast(&w_Complex[0]), c_w_sizes);

//lookSee<<<1,1>>>(fftSize_*fftSize_*numDivisions*numDivisions*conv_in_channels_, im_pad_buff, 1.0);
//lookSee<<<1,1>>>(fftSize_*fftSize_*conv_in_channels_*conv_out_channels_, w_pad_buff, 1.0);


facebook::deeplearning::torch::fbfft2dHost<1>(im_space, im_complex, buffer, 
                facebook::deeplearning::torch::FFTParameters().withFbfft().forward(), 
                0);

//lookSee<<<1,1>>>(fftSize_*fftSize_*conv_in_channels_*2*numDivisions*numDivisions, im_complex.data(), 1.0);

//Use the same stream for now
facebook::deeplearning::torch::fbfft2dHost<1>(w_space, w_complex, buffer, 
                facebook::deeplearning::torch::FFTParameters().withFbfft().forward(), 
                0);

//lookSee<<<1,1>>>(fftSize_*fftSize_*conv_out_channels_*conv_in_channels_*numDivisions*numDivisions*2, w_complex.data(), 1.0);


facebook::cuda::fbfft::fbfft2D<1>(w_space, w_complex, 0);

facebook::cuda::DeviceTensor<float, 4> out_complex(thrust::raw_pointer_cast(&o_Complex[0]), c_out_sizes);

//In this case output size
int num_kernels = fftSize_ * fftSize_ * conv_out_channels_*numDivisions*numDivisions;


hadamard_forward_with_sum<<<CAFFE_GET_BLOCKS(num_kernels),
                           CAFFE_CUDA_NUM_THREADS>>>(num_kernels, 
                            im_complex.data(), w_complex.data(), out_complex.data(),
                            fftSize_, conv_out_channels_, conv_in_channels_, numDivisions);

//lookSee<<<1,1>>>(fftSize_*fftSize_*conv_out_channels_*2*numDivisions*numDivisions, out_complex.data(), 1.0);


facebook::cuda::DeviceTensor<float, 3> out_space_dt(o_pad_buffer_.mutable_gpu_data(), out_sizes);


//ifft still uses real, complex (dest, source)
facebook::deeplearning::torch::fbfft2dHost<1>(out_space_dt, out_complex, buffer, 
                facebook::deeplearning::torch::FFTParameters().withFbfft().inverse().normalize(false), 
                0);

//lookSee<<<1,1>>>(fftSize_*fftSize_*conv_out_channels_*numDivisions*numDivisions, out_space_dt.data(), 16.0);

int overlap = kernel_h_-1;
int overlapedSize = ((2*kernel_h_-1)*numDivisions) - (overlap*(numDivisions-1));
int extraInfo = (overlapedSize-conv_in_height_)/2;

num_kernels = conv_out_channels_*height_out_*height_out_;

overlap_and_crop<<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>
                            (num_kernels, out_space_dt.data(), output, 
                              fftSize_, overlapedSize, kernel_h_, numDivisions, extraInfo, height_out_, (float)fftSize_*fftSize_);

//lookSee<<<1,1>>>(conv_out_channels_*height_out_*height_out_, thrust::raw_pointer_cast(&tempOutput[0]), 1.0);

}


template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      height_out_ * width_out_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_gpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_ / group_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_gpu(col_buff, input);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_gpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_ / group_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, height_out_ * width_out_, 1.,
      input, bias_multiplier_.gpu_data(), 1., bias);
}

#endif

INSTANTIATE_CLASS(BaseConvolutionLayer);

}  // namespace caffe
