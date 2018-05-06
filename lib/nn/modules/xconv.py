
from torch.nn import init
from torch import nn
import torch.nn.functional as F


class large_separable_conv(nn.Module):
    def __init__(self,chl_in,  ks=15, chl_mid=256, chl_out=1024):
        super().__init__()
        pad=(ks-1)//2
        self.col_max = nn.Conv2d(chl_in,chl_mid,(ks,1),padding=(pad,0))
        self.col     = nn.Conv2d(chl_mid,chl_out,(1,ks),padding=(0,pad))
        self.row_max = nn.Conv2d(chl_in,chl_mid,(1,ks),padding=(pad,0))
        self.row     = nn.Conv2d(chl_mid,chl_out,(ks,1),padding=(0,pad))

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'col_max.weight':'col_max_w',
            'col_max.bias': 'col_max_b',
            'col.weight': 'col_w',
            'col.bias': 'col_b',
            'row_max.weight': 'row_max_w',
            'row_max.bias': 'row_max_b',
            'row.weight': 'row_w',
            'row.bias': 'row_b',
        }
        return detectron_weight_mapping,[]

    def forward(self, x):
        y1 = self.col(self.col_max(x))
        y2 = self.row(self.row_max(x))
        return y1+y2
