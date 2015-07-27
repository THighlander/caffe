#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>

namespace caffe {

__global__ void hadamard_forward_with_sum(int n, cufftComplex* image, cufftComplex* kernel, cufftComplex* output, int fftSize, int out_channels, int in_channels)
{
  int index = 0;
  int which_out_channel = 0;
  int k_index, im_index;


  CUDA_KERNEL_LOOP(j, n)
  {
    which_out_channel = j/(fftSize*fftSize);
    index = j%(fftSize*fftSize);

    for (int in = 0; in < in_channels; ++in)
    {
      k_index = index+(in*fftSize*fftSize) + which_out_channel*in_channels*fftSize*fftSize;
      im_index = index+(in*fftSize*fftSize);

      output[j].x += (image[im_index].x * kernel[k_index].x - image[im_index].y * kernel[k_index].y);
      output[j].y += (image[im_index].x * kernel[k_index].y + image[im_index].y * kernel[k_index].x);
    }
  }
}

template <typename Dtype>
__global__ void crop_ouput(int n, float* fft_output, Dtype* output, int length, int fftSize, int offset, int out_channels)
{
  fft_output += offset*fftSize;
  int size = length * length;
  CUDA_KERNEL_LOOP(index, n)
  {
    output[index] = fft_output[((index/size)*fftSize*fftSize) + (((index/length)%length)*fftSize) + offset + (index%length)]/((float)(fftSize*fftSize));
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
    fftSize_ = conv_in_height_ + kernel_h_ - 1;
  }

  kernel_dim_ = conv_in_channels_ * kernel_h_ * kernel_w_;
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_ / group_;
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_ / group_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  if (reverse_dimensions()) {
    col_buffer_.Reshape(1, kernel_dim_, height_, width_);

  } else {
    col_buffer_.Reshape(1, kernel_dim_, height_out_, width_out_);
    w_pad_buffer_.Reshape(1, conv_out_channels_*conv_in_channels_, fftSize_, fftSize_);
    im_pad_buffer_.Reshape(1, conv_in_channels_, fftSize_, fftSize_);
    o_pad_buffer_.Reshape(1, conv_out_channels_, fftSize_, fftSize_);
  }
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, height_out_ * width_out_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }


  #ifndef CPU_ONLY

    fftSize_ = conv_in_height_ + kernel_h_ - 1;
    int dims[2] = {fftSize_, fftSize_};

    
    cufftPlanMany(&planR2CInDepth_, 2, dims,
        dims, 1, fftSize_*fftSize_, //in dims
        dims, 1, fftSize_*fftSize_, //out dims
        CUFFT_R2C, conv_in_channels_);

    cufftPlanMany(&planR2CNumK_, 2, dims,
        dims, 1, fftSize_*fftSize_, //in dims
        dims, 1, fftSize_*fftSize_, //out dims
        CUFFT_R2C, conv_out_channels_*conv_in_channels_);
    
    cufftPlanMany(&planC2Rout_, 2, dims,
        dims, 1, fftSize_*fftSize_, //in dims
        dims, 1, fftSize_*fftSize_, //out dims
        CUFFT_C2R, conv_out_channels_);
  

  //Create the device vectors for the 
  im_pad_buff_Complex = thrust::device_vector<cufftComplex>(fftSize_*fftSize_*conv_in_channels_);
  w_pad_buff_Complex = thrust::device_vector<cufftComplex>(fftSize_*fftSize_*conv_in_channels_*conv_out_channels_);

  cufftComplex complexZero;
  complexZero.x = 0;
  complexZero.y = 0;

  //Initializes the output vector with 0's so that maybe we can += in CUDA for loops
  out_pad_buff_Complex = thrust::device_vector<cufftComplex>(fftSize_*fftSize_*conv_out_channels_, complexZero);

  #endif
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

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {


//Pad the input image
cufftReal* im_pad_buff;
conv_pad_im_gpu(input, im_pad_buffer_.mutable_gpu_data());
im_pad_buff = im_pad_buffer_.mutable_gpu_data();

cufftReal* w_pad_buff;
conv_pad_w_gpu(weights, w_pad_buffer_.mutable_gpu_data());
w_pad_buff = w_pad_buffer_.mutable_gpu_data();

//Execute forward FFTs
cufftExecR2C(planR2CInDepth_, im_pad_buff, thrust::raw_pointer_cast(&im_pad_buff_Complex[0]));
cufftExecR2C(planR2CNumK_, w_pad_buff, thrust::raw_pointer_cast(&w_pad_buff_Complex[0]));
//cudaDeviceSynchronize();

//Time for the hadamard product with a sum over the conv_in_channels_ 
int num_kernels = fftSize_ * fftSize_ * conv_out_channels_;

hadamard_forward_with_sum<<<CAFFE_GET_BLOCKS(num_kernels),
                           CAFFE_CUDA_NUM_THREADS>>>(num_kernels, 
                            thrust::raw_pointer_cast(&im_pad_buff_Complex[0]), 
                            thrust::raw_pointer_cast(&w_pad_buff_Complex[0]),
                           thrust::raw_pointer_cast(&out_pad_buff_Complex[0]), fftSize_, conv_out_channels_, conv_in_channels_);

//Inverse FFT 
cufftExecC2R(planC2Rout_, thrust::raw_pointer_cast(&out_pad_buff_Complex[0]), o_pad_buffer_.mutable_gpu_data());

float* o_pad_buff = o_pad_buffer_.mutable_gpu_data();


float* test = o_pad_buffer_.mutable_cpu_data();
for (int k = 0; k < conv_in_channels_*conv_out_channels_; ++k)
{
  for (int i = 0; i < fftSize_; ++i)
  {
    for (int j = 0; j < fftSize_; ++j)
    {
      std::cout << test[(k*fftSize_ * fftSize_) + i*fftSize_ +j]/(fftSize_*fftSize_) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}


//Need crop the answer and put it at the start of the output blob
int extraInfo = (fftSize_- height_out_)/2;

num_kernels = conv_out_spatial_dim_*conv_out_channels_;
crop_ouput<<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>
                          (num_kernels, o_pad_buff, output, height_out_, fftSize_, extraInfo,
                            conv_out_channels_);

cufftDestroy(planR2CInDepth_);
cufftDestroy(planR2CNumK_);
cufftDestroy(planC2Rout_);

/*
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
  */
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

//May not be the optimal loction for this, however it works so for development purposes we will leave it here.
template <typename Dtype>
BaseConvolutionLayer<Dtype>::~BaseConvolutionLayer()
{
  //destroy the cufft plans created by the convoltuional layer in vision_layers.hpp
  cufftDestroy(planR2CInDepth_);
  cufftDestroy(planR2CNumK_);
  cufftDestroy(planC2Rout_);
}
#endif

INSTANTIATE_CLASS(BaseConvolutionLayer);

}  // namespace caffe
