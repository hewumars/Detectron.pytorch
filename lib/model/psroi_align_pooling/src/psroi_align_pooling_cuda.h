
int psroi_align_pooling_forward_cuda(
 const int pooled_height, const int pooled_width,
 const int sample_height, const int sample_width,
 const float spatial_scale,
 const int group_size, const int output_dim,
 THCudaTensor* features,
 THCudaTensor* rois,
 THCudaTensor* output,
 THCudaIntTensor* mapping_channel,
 THCudaIntTensor* argmax_position);

int psroi_align_pooling_backward_cuda(
 const int pooled_height, const int pooled_width,
 const int sample_height, const int sample_width,
 const float spatial_scale,
 const int group_size, const int output_dim,
 THCudaTensor* top_grad,
 THCudaTensor* rois,
 THCudaTensor* bottom_grad,
 THCudaIntTensor* mapping_channel,
 THCudaIntTensor* argmax_position);

