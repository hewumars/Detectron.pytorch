#ifndef _PSROI_ALIGN_KERNEL
#define _PSROI_ALIGN_KERNEL

#ifdef __cplusplus
extern "C" {
#endif


int PSAlignPoolForwardLauncher(
    const float* bottom_data, const float spatial_scale, const int num_rois,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int sample_height, const int sample_width,
    const float* bottom_rois, const int output_dim,
    const int group_size, float* top_data,
    int* mapping_channel, int* argmax_position,
    cudaStream_t stream);

int PSAlignPoolBackwardLauncher(
    const float* top_diff,
    const int* mapping_channel, const int* argmax_position,
    const int batch_size,
    const int num_rois, const float spatial_scale, const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int sample_height, const int sample_width,
    const int output_dim, float* bottom_diff,
    const float* bottom_rois, cudaStream_t stream);






#ifdef __cplusplus
}
#endif

#endif