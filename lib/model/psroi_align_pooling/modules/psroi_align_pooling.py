from torch.nn.modules.module import Module
from ..functions.psroi_align_pooling import PSRoiAlignPoolingFunction

class PSRoIAlignPool(Module):
    def __init__(self, pooled_height, pooled_width,sample_height,sample_width, spatial_scale, group_size, output_dim):
        super(PSRoIAlignPool, self).__init__()

        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.sample_height = int(sample_height)
        self.sample_width = int(sample_width)
        self.spatial_scale = float(spatial_scale)
        self.group_size = int(group_size)
        self.output_dim = int(output_dim)

    def forward(self, features, rois):
        return PSRoiAlignPoolingFunction(self.pooled_height, self.pooled_width,self.sample_height,self.sample_width, self.spatial_scale, self.group_size, self.output_dim)(features, rois)
