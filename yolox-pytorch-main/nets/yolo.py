import torch
import torch.nn as nn
import sys
sys.argv.append('.')
from nets.darknet import BaseConv, CSPDarknet, CSPLayer, DWConv
from collections import OrderedDict

def conv2d(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels,scale):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=scale, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x

def make_three_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m

#---------------------------------------------------#
#   五次卷积块
#---------------------------------------------------#
def make_five_conv(filters_list, in_filters):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m


class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width=1.0, strides=[8, 16, 32], in_channels=[256, 512, 1024], act="silu", depthwise=False,):
        super().__init__()
        self.n_anchors = 1
        self.num_classes = num_classes
        
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(BaseConv(in_channels=int(in_channels[i] * width), out_channels=int(256 * width), ksize=1, stride=1, act=act))
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels=int(256 * width),out_channels=int(256 * width),ksize=3,stride=1,act=act),
                Conv(in_channels=int(256 * width),out_channels=int(256 * width),ksize=3,stride=1,act=act),
            ]))
            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels=int(256 * width),out_channels=int(256 * width),ksize=3,stride=1,act=act),
                Conv(in_channels=int(256 * width),out_channels=int(256 * width),ksize=3,stride=1,act=act)
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels=int(256 * width),out_channels=self.n_anchors * self.num_classes,kernel_size=1,stride=1,padding=0)
            )
            self.reg_preds.append(
                nn.Conv2d(in_channels=int(256 * width),out_channels=4,kernel_size=1,stride=1,padding=0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels=int(256 * width),out_channels=self.n_anchors * 1,kernel_size=1,stride=1,padding=0)
            )

    def forward(self, inputs):
        outputs = []
        for k, x in enumerate(inputs):
            x       = self.stems[k](x)

            cls_feat    = self.cls_convs[k](x)
            cls_output  = self.cls_preds[k](cls_feat)

            reg_feat    = self.reg_convs[k](x)
            reg_output  = self.reg_preds[k](reg_feat)
            obj_output  = self.obj_preds[k](reg_feat)

            output      = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs

class YOLOPAFPN(nn.Module):
    def __init__(self, depth=1.0, width=1.0, in_features=("dark3", "dark4", "dark5"), in_channels=[256, 512, 1024], depthwise=False, act="silu"):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)

        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        self.upsample_4 = Upsample(512, 128, 4)
        # 26,26,512 -> 52,52,256
        self.upsample_2 = Upsample(256, 128, 2)
        # 52,52,768 -> 52,52,256
        self.make_five_conv1 = make_five_conv([128, 128*2], 128*3)

    def forward(self, input):
        out_features = self.backbone.forward(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features
        # print(x2.shape,x1.shape,x0.shape)
        # print(x0.shape)    #[1, 512, 20, 20]
        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        # print(fpn_out0.shape)   #[1, 256, 20, 20]
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16
        # print(f_out0.shape)   #1, 256, 40, 40

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        # print(fpn_out1.shape)   #1, 128, 40, 40
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8
        # print(pan_out2.shape)   #1, 128, 80, 80

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16
        # print(pan_out1.shape)   #1, 256, 40, 40

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32
        # print(pan_out1.shape)   #1, 256, 40, 40

        # outputs = (pan_out2, pan_out1, pan_out0)
        #=========================================================================
        # print(*(pan_out2.shape, pan_out1.shape, pan_out0.shape))
        up_out5 = self.upsample_4(pan_out0)
        up_out4 = self.upsample_2(pan_out1)
        p3 = torch.cat([pan_out2, up_out4, up_out5], axis=1)  # ([1, 384, 80, 80])
        pan_out2 = self.make_five_conv1(p3)

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)

        return outputs

class YoloBody(nn.Module):
    def __init__(self, num_classes, phi):
        super().__init__()
        depth_dict = {'s' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
        width_dict = {'s' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        depth, width    = depth_dict[phi], width_dict[phi]

        self.backbone   = YOLOPAFPN(depth, width)
        self.head       = YOLOXHead(num_classes, width)

    def forward(self, x):
        fpn_outs    = self.backbone.forward(x)
        outputs     = self.head.forward(fpn_outs)
        return outputs


img = torch.FloatTensor(1,3,640,640)
print(img.shape)
yolo = YoloBody(1,'s')
import numpy as np
output = yolo(img)
#
# for i in output:
#     print(i.shape)

# print(output.shape)