#include <THC/THC.h>
#include <math.h>
#include "psroi_align_pooling_kernel.h"

extern THCState* state;

int psroi_align_pooling_forward_cuda(
 const int pooled_height, const int pooled_width,
 const int sample_height, const int sample_width,
 const float spatial_scale,
 const int group_size, const int output_dim,
 THCudaTensor* features,
 THCudaTensor* rois,
 THCudaTensor* output,
 THCudaIntTensor* mapping_channel,
 THCudaIntTensor* argmax_position)
{
    float* bottom_data = THCudaTensor_data(state, features);
    float* bottom_rois = THCudaTensor_data(state, rois);
    float* top_data = THCudaTensor_data(state, output);
    int* top_mapping_channel = THCudaIntTensor_data(state, mapping_channel);
    int* top_argmax_position = THCudaIntTensor_data(state, argmax_position);
	//Get # of Rois
    int num_rois = THCudaTensor_size(state, rois, 0);
	int size_rois = THCudaTensor_size(state, rois, 1);
	if(size_rois !=5){
	    return 0;
	}

	//Get # of batch_size
	int batch_size = THCudaTensor_size(state, features, 0);
	if (batch_size!=1)
	{
		return 0;
	}
	int num_channels = THCudaTensor_size(state, features, 1);
	int data_height = THCudaTensor_size(state, features, 2);
	int data_width = THCudaTensor_size(state, features, 3);

	cudaStream_t stream = THCState_getCurrentStream(state);


	//Forward Kernel
	return PSAlignPoolForwardLauncher(
                    bottom_data, spatial_scale, num_rois,
                    num_channels, data_height, data_width,
                    pooled_height, pooled_width,
                    sample_height, sample_width,
                    bottom_rois,
                    output_dim, group_size, top_data,
                    top_mapping_channel, top_argmax_position,
                    stream);
}


int psroi_align_pooling_backward_cuda(
 const int pooled_height, const int pooled_width,
 const int sample_height, const int sample_width,
 const float spatial_scale,
 const int group_size, const int output_dim,
 THCudaTensor* top_grad,
 THCudaTensor* rois,
 THCudaTensor* bottom_grad,
 THCudaIntTensor* mapping_channel,
 THCudaIntTensor* argmax_position)
{
    float* top_grad_flat = THCudaTensor_data(state, top_grad);
    float* rois_flat = THCudaTensor_data(state, rois);

    float* bottom_grad_flat = THCudaTensor_data(state, bottom_grad);
    int* mapping_channel_flat = THCudaIntTensor_data(state,mapping_channel);
    int* argmax_position_flat = THCudaIntTensor_data(state,argmax_position);

    int num_rois = THCudaTensor_size(state, rois, 0);
    int size_rois = THCudaTensor_size(state, rois, 1);
    if (size_rois != 5)
    {
        return 0;
    }

    int batch_size = THCudaTensor_size(state, bottom_grad, 0);
    if (batch_size != 1)
    {
        return 0;
    }
    int data_height = THCudaTensor_size(state, bottom_grad, 2);
    int data_width = THCudaTensor_size(state, bottom_grad, 3);
    int num_channels = THCudaTensor_size(state, bottom_grad, 1);

    cudaStream_t stream = THCState_getCurrentStream(state);

    //Backward Kernel
    return PSAlignPoolBackwardLauncher(
                    top_grad_flat,
                    mapping_channel_flat, argmax_position_flat,
                    batch_size,
                    num_rois,  spatial_scale,  num_channels,
                    data_height, data_width,
                    pooled_height, pooled_width,
                    sample_height, sample_width,
                    output_dim, bottom_grad_flat,
                    rois_flat, stream);
}

