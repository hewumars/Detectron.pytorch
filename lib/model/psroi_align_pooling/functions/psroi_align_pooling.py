import torch
from torch.autograd import Function
from .._ext import psroi_align_pooling

class PSRoiAlignPoolingFunction(Function):
    def __init__(self,pooled_height,pooled_width,sample_height,sample_width,spatial_scale,group_size):
        self.pooled_height = int(pooled_height)
        self.pooled_width = int(pooled_width)
        self.sample_height = int(sample_height)
        self.sample_width = int(sample_width)
        self.spatial_scale = float(spatial_scale)
        self.group_size = int(group_size)
        self.output = None
        self.mapping_channel = None
        self.argmax_position = None
        self.rois = None
        self.feature_size = None
        self.output_dim = None
    def forward(self, features, rois):
        batch_size, num_channels, data_height, data_width = features.size()
        self.output_dim = num_channels // self.pooled_height // self.pooled_width
        # self.output_dim = num_channels
        num_rois = rois.size()[0]
        output = torch.zeros(num_rois, self.output_dim, self.pooled_height, self.pooled_width)
        mapping_channel = torch.cuda.IntTensor(num_rois, self.output_dim, self.pooled_height, self.pooled_width).zero_()
        argmax_position = torch.cuda.IntTensor(num_rois, self.output_dim, self.pooled_height, self.pooled_width).zero_()
        output = output.cuda()
        psroi_align_pooling.psroi_align_pooling_forward_cuda(self.pooled_height,
                                                            self.pooled_width,
                                                            self.sample_height,
                                                            self.sample_width,
                                                            self.spatial_scale,
                                                            self.group_size,
                                                            self.output_dim,
                                                            features,
                                                            rois,
                                                            output,
                                                            mapping_channel,
                                                            argmax_position
                                                             )
        self.output = output
        self.mapping_channel = mapping_channel
        self.argmax_position = argmax_position
        self.rois = rois
        self.feature_size = features.size()

        return output

    def backward(self, grad_output):
        assert(self.feature_size is not None and grad_output.is_cuda)

        batch_size, num_channels, data_height, data_width = self.feature_size

        grad_input = torch.zeros(batch_size, num_channels, data_height, data_width).cuda()
        # import pdb
        # pdb.set_trace()
        psroi_align_pooling.psroi_align_pooling_backward_cuda(self.pooled_height,
                                                            self.pooled_width,
                                                            self.sample_height,
                                                            self.sample_width,
                                                            self.spatial_scale,
                                                            self.group_size,
                                                            self.output_dim,
                                                            grad_output,
                                                            self.rois,
                                                            grad_input,
                                                            self.mapping_channel,
                                                            self.argmax_position)
        return grad_input, None
