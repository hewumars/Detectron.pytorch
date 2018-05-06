import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from core.config import cfg
import utils.net as net_utils

# # light head
# conv_new_1 = mx.sym.Convolution(data=relu1, kernel=(15, 1), pad=(7, 0), num_filter=256, name="conv_new_1", lr_mult=3.0)
# relu_new_1 = mx.sym.Activation(data=conv_new_1, act_type='relu', name='relu1')
# conv_new_2 = mx.sym.Convolution(data=relu_new_1, kernel=(1, 15), pad=(0, 7), num_filter=10 * 7 * 7, name="conv_new_2",
#                                 lr_mult=3.0)
# relu_new_2 = mx.sym.Activation(data=conv_new_2, act_type='relu', name='relu2')
# conv_new_3 = mx.sym.Convolution(data=relu1, kernel=(1, 15), pad=(0, 7), num_filter=256, name="conv_new_3", lr_mult=3.0)
# relu_new_3 = mx.sym.Activation(data=conv_new_3, act_type='relu', name='relu3')
# conv_new_4 = mx.sym.Convolution(data=relu_new_3, kernel=(15, 1), pad=(7, 0), num_filter=10 * 7 * 7, name="conv_new_4",
#                                 lr_mult=3.0)
# relu_new_4 = mx.sym.Activation(data=conv_new_4, act_type='relu', name='relu4')
# light_head = mx.symbol.broadcast_add(name='light_head', *[relu_new_2, relu_new_4])
# roi_pool = mx.contrib.sym.PSROIPooling(name='roi_pool', data=light_head, rois=rois, group_size=7, pooled_size=7,
#                                        output_dim=10, spatial_scale=0.0625)
# fc_new_1 = mx.symbol.FullyConnected(name='fc_new_1', data=roi_pool, num_hidden=2048)
# fc_new_1_relu = mx.sym.Activation(data=fc_new_1, act_type='relu', name='fc_new_1_relu')
# cls_score = mx.symbol.FullyConnected(name='cls_score', data=fc_new_1_relu, num_hidden=num_classes)
# bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=fc_new_1_relu, num_hidden=num_reg_classes * 4)

class light_head_rcnn_outputs(nn.Module):
    def __init__(self, dim_in,roi_xform_func,spatial_scale):
        super().__init__()
        self.ps_fc_1 = nn.Linear(dim_in, 2048)
        self.cls_score = nn.Linear(2048, cfg.MODEL.NUM_CLASSES)
        if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
            self.bbox_pred = nn.Linear(2048, 4)
        else:
            self.bbox_pred = nn.Linear(2048, 4 * cfg.MODEL.NUM_CLASSES)
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self._init_weights()

    def _init_weights(self):
        init.normal(self.ps_fc_1.weight, std=0.01)
        init.constant(self.ps_fc_1.bias, 0)
        init.normal(self.cls_score.weight, std=0.01)
        init.constant(self.cls_score.bias, 0)
        init.normal(self.bbox_pred.weight, std=0.001)
        init.constant(self.bbox_pred.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'ps_fc_1.weight': 'ps_fc_1_w',
            'ps_fc_1.bias': 'ps_fc_1_b',
            'cls_score.weight': 'cls_score_w',
            'cls_score.bias': 'cls_score_b',
            'bbox_pred.weight': 'bbox_pred_w',
            'bbox_pred.bias': 'bbox_pred_b'
        }
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='rois',
            method=cfg.LIGHT_HEAD_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.LIGHT_HEAD_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale
        )
        x = F.relu(self.ps_fc_1(x.view(x.size(0),-1)),inplace=True)
        cls_score = self.cls_score(x)
        if not self.training:
            cls_score = F.softmax(cls_score, dim=1)
        bbox_pred = self.bbox_pred(x)

        return cls_score, bbox_pred


def light_head_rcnn_losses(cls_score, bbox_pred, label_int32, bbox_targets,
                     bbox_inside_weights, bbox_outside_weights):
    device_id = cls_score.get_device()
    rois_label = Variable(torch.from_numpy(label_int32.astype('int64'))).cuda(device_id)
    # logits_rois_label = Variable(torch.cuda.IntTensor(cls_score.size()[0],cls_score.size()[1]).zero_())
    # for i, val in enumerate(rois_label.data):
    #     logits_rois_label.data[i, val] = 1 if val > 0 else 0
    loss_cls =F.cross_entropy(cls_score, rois_label)# F.binary_cross_entropy_with_logits(cls_score,logits_rois_label)
    bbox_targets = Variable(torch.from_numpy(bbox_targets)).cuda(device_id)
    bbox_inside_weights = Variable(torch.from_numpy(bbox_inside_weights)).cuda(device_id)
    bbox_outside_weights = Variable(torch.from_numpy(bbox_outside_weights)).cuda(device_id)
    loss_bbox = net_utils.smooth_l1_loss(
        bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)
    return loss_cls, loss_bbox*2

