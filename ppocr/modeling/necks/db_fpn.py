# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle import ParamAttr
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../../..')))

from ppocr.modeling.backbones.det_mobilenet_v3 import SEModule


class DSConv(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding,
                 stride=1,
                 groups=None,
                 if_act=True,
                 act="relu",
                 **kwargs):
        super(DSConv, self).__init__()
        if groups == None:
            groups = in_channels
        self.if_act = if_act
        self.act = act
        self.conv1 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias_attr=False)

        self.bn1 = nn.BatchNorm(num_channels=in_channels, act=None)

        self.conv2 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=int(in_channels * 4),
            kernel_size=1,
            stride=1,
            bias_attr=False)

        self.bn2 = nn.BatchNorm(num_channels=int(in_channels * 4), act=None)

        self.conv3 = nn.Conv2D(
            in_channels=int(in_channels * 4),
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias_attr=False)
        self._c = [in_channels, out_channels]
        if in_channels != out_channels:
            self.conv_end = nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                bias_attr=False)

    def forward(self, inputs):

        x = self.conv1(inputs)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.if_act:
            if self.act == "relu":
                x = F.relu(x)
            elif self.act == "hardswish":
                x = F.hardswish(x)
            else:
                print("The activation function({}) is selected incorrectly.".
                      format(self.act))
                exit()

        x = self.conv3(x)
        if self._c[0] != self._c[1]:
            x = x + self.conv_end(inputs)
        return x


class DBFPN(nn.Layer):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(DBFPN, self).__init__()
        self.out_channels = out_channels
        weight_attr = paddle.nn.initializer.KaimingUniform()

        self.in2_conv = nn.Conv2D(
            in_channels=in_channels[0],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.in3_conv = nn.Conv2D(
            in_channels=in_channels[1],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.in4_conv = nn.Conv2D(
            in_channels=in_channels[2],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.in5_conv = nn.Conv2D(
            in_channels=in_channels[3],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.p5_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.p4_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.p3_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.p2_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)

    def forward(self, x):
        c2, c3, c4, c5 = x

        in5 = self.in5_conv(c5)
        in4 = self.in4_conv(c4)
        in3 = self.in3_conv(c3)
        in2 = self.in2_conv(c2)

        out4 = in4 + F.upsample(
            in5, scale_factor=2, mode="nearest", align_mode=1)  # 1/16
        out3 = in3 + F.upsample(
            out4, scale_factor=2, mode="nearest", align_mode=1)  # 1/8
        out2 = in2 + F.upsample(
            out3, scale_factor=2, mode="nearest", align_mode=1)  # 1/4

        p5 = self.p5_conv(in5)
        p4 = self.p4_conv(out4)
        p3 = self.p3_conv(out3)
        p2 = self.p2_conv(out2)
        p5 = F.upsample(p5, scale_factor=8, mode="nearest", align_mode=1)
        p4 = F.upsample(p4, scale_factor=4, mode="nearest", align_mode=1)
        p3 = F.upsample(p3, scale_factor=2, mode="nearest", align_mode=1)

        fuse = paddle.concat([p5, p4, p3, p2], axis=1)
        return fuse


class RSELayer(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, shortcut=True):
        super(RSELayer, self).__init__()
        weight_attr = paddle.nn.initializer.KaimingUniform()
        self.out_channels = out_channels
        self.in_conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            padding=int(kernel_size // 2),
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.se_block = SEModule(self.out_channels)
        self.shortcut = shortcut

    def forward(self, ins):
        x = self.in_conv(ins)
        if self.shortcut:
            out = x + self.se_block(x)
        else:
            out = self.se_block(x)
        return out


class RSEFPN(nn.Layer):
    def __init__(self, in_channels, out_channels, shortcut=True, **kwargs):
        super(RSEFPN, self).__init__()
        self.out_channels = out_channels
        self.ins_conv = nn.LayerList()
        self.inp_conv = nn.LayerList()

        for i in range(len(in_channels)):
            self.ins_conv.append(
                RSELayer(
                    in_channels[i],
                    out_channels,
                    kernel_size=1,
                    shortcut=shortcut))
            self.inp_conv.append(
                RSELayer(
                    out_channels,
                    out_channels // 4,
                    kernel_size=3,
                    shortcut=shortcut))

    def forward(self, x):
        c2, c3, c4, c5 = x

        in5 = self.ins_conv[3](c5)
        in4 = self.ins_conv[2](c4)
        in3 = self.ins_conv[1](c3)
        in2 = self.ins_conv[0](c2)

        out4 = in4 + F.upsample(
            in5, scale_factor=2, mode="nearest", align_mode=1)  # 1/16
        out3 = in3 + F.upsample(
            out4, scale_factor=2, mode="nearest", align_mode=1)  # 1/8
        out2 = in2 + F.upsample(
            out3, scale_factor=2, mode="nearest", align_mode=1)  # 1/4

        p5 = self.inp_conv[3](in5)
        p4 = self.inp_conv[2](out4)
        p3 = self.inp_conv[1](out3)
        p2 = self.inp_conv[0](out2)

        p5 = F.upsample(p5, scale_factor=8, mode="nearest", align_mode=1)
        p4 = F.upsample(p4, scale_factor=4, mode="nearest", align_mode=1)
        p3 = F.upsample(p3, scale_factor=2, mode="nearest", align_mode=1)

        fuse = paddle.concat([p5, p4, p3, p2], axis=1)
        return fuse


class LKPAN(nn.Layer):
    def __init__(self, in_channels, out_channels, mode='large', **kwargs):
        super(LKPAN, self).__init__()
        self.out_channels = out_channels
        weight_attr = paddle.nn.initializer.KaimingUniform()

        self.ins_conv = nn.LayerList()
        self.inp_conv = nn.LayerList()
        # pan head
        self.pan_head_conv = nn.LayerList()
        self.pan_lat_conv = nn.LayerList()

        if mode.lower() == 'lite':
            p_layer = DSConv
        elif mode.lower() == 'large':
            p_layer = nn.Conv2D
        else:
            raise ValueError(
                "mode can only be one of ['lite', 'large'], but received {}".
                format(mode))

        for i in range(len(in_channels)):
            self.ins_conv.append(
                nn.Conv2D(
                    in_channels=in_channels[i],
                    out_channels=self.out_channels,
                    kernel_size=1,
                    weight_attr=ParamAttr(initializer=weight_attr),
                    bias_attr=False))

            self.inp_conv.append(
                p_layer(
                    in_channels=self.out_channels,
                    out_channels=self.out_channels // 4,
                    kernel_size=9,
                    padding=4,
                    weight_attr=ParamAttr(initializer=weight_attr),
                    bias_attr=False))

            if i > 0:
                self.pan_head_conv.append(
                    nn.Conv2D(
                        in_channels=self.out_channels // 4,
                        out_channels=self.out_channels // 4,
                        kernel_size=3,
                        padding=1,
                        stride=2,
                        weight_attr=ParamAttr(initializer=weight_attr),
                        bias_attr=False))
            self.pan_lat_conv.append(
                p_layer(
                    in_channels=self.out_channels // 4,
                    out_channels=self.out_channels // 4,
                    kernel_size=9,
                    padding=4,
                    weight_attr=ParamAttr(initializer=weight_attr),
                    bias_attr=False))

    def forward(self, x):
        c2, c3, c4, c5 = x

        in5 = self.ins_conv[3](c5)
        in4 = self.ins_conv[2](c4)
        in3 = self.ins_conv[1](c3)
        in2 = self.ins_conv[0](c2)

        out4 = in4 + F.upsample(
            in5, scale_factor=2, mode="nearest", align_mode=1)  # 1/16
        out3 = in3 + F.upsample(
            out4, scale_factor=2, mode="nearest", align_mode=1)  # 1/8
        out2 = in2 + F.upsample(
            out3, scale_factor=2, mode="nearest", align_mode=1)  # 1/4

        f5 = self.inp_conv[3](in5)
        f4 = self.inp_conv[2](out4)
        f3 = self.inp_conv[1](out3)
        f2 = self.inp_conv[0](out2)

        pan3 = f3 + self.pan_head_conv[0](f2)
        pan4 = f4 + self.pan_head_conv[1](pan3)
        pan5 = f5 + self.pan_head_conv[2](pan4)

        p2 = self.pan_lat_conv[0](f2)
        p3 = self.pan_lat_conv[1](pan3)
        p4 = self.pan_lat_conv[2](pan4)
        p5 = self.pan_lat_conv[3](pan5)

        p5 = F.upsample(p5, scale_factor=8, mode="nearest", align_mode=1)
        p4 = F.upsample(p4, scale_factor=4, mode="nearest", align_mode=1)
        p3 = F.upsample(p3, scale_factor=2, mode="nearest", align_mode=1)

        fuse = paddle.concat([p5, p4, p3, p2], axis=1)
        return fuse
    
    
 '''
change from mmocr
https://github.com/open-mmlab/mmocr/blob/b1ab4c7c33424e241d96cefa9c6c225cfa3bcd58/mmocr/models/textdet/necks/fpn_cat.py 

'''

class DBASFFPN(nn.Layer):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(DBASFFPN, self).__init__()
        self.out_channels = out_channels  # 256
        weight_attr = paddle.nn.initializer.KaimingUniform()
        self.num_ins = len(in_channels)  # 4
        self.num_outs = self.num_ins     # 4

        self.in2_conv = nn.Conv2D(
            in_channels=in_channels[0],   #  16 24 56 480
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.in3_conv = nn.Conv2D(
            in_channels=in_channels[1],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.in4_conv = nn.Conv2D(
            in_channels=in_channels[2],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.in5_conv = nn.Conv2D(
            in_channels=in_channels[3],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.p5_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.p4_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.p3_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.p2_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.asf_conv =nn.Conv2D(
            in_channels=out_channels * self.num_outs,  # 256*4
            out_channels=out_channels * self.num_outs, # 256*4
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.asf_attn = ScaleChannelSpatialAttention(
            self.out_channels * self.num_outs,
            (self.out_channels * self.num_outs) // 4, self.num_outs)
    def forward(self, x):
        c2, c3, c4, c5 = x   #  [1, 16, 160, 160] [1, 24, 80, 80] [1, 56, 40, 40] [1, 480, 20, 20]

        in5 = self.in5_conv(c5)  # [1, 256, 20, 20]
        in4 = self.in4_conv(c4)   # [1, 256, 40, 40]
        in3 = self.in3_conv(c3)   # [1, 256, 80, 80]
        in2 = self.in2_conv(c2)   # [1, 256, 160, 160]

        out4 = in4 + F.upsample(
            in5, scale_factor=2, mode="nearest", align_mode=1)  # 1/16   [1, 256, 40, 40]
        out3 = in3 + F.upsample(
            out4, scale_factor=2, mode="nearest", align_mode=1)  # 1/8   [1, 256, 80, 80]
        out2 = in2 + F.upsample(
            out3, scale_factor=2, mode="nearest", align_mode=1)  # 1/4   [1, 256, 160, 160]

        p5 = self.p5_conv(in5)    # [1, 64, 20, 20]
        p4 = self.p4_conv(out4)   # [1, 64, 40, 40]
        p3 = self.p3_conv(out3)   # [1, 64, 80, 80]
        p2 = self.p2_conv(out2)   # [1, 64, 160, 160]
        p5 = F.upsample(p5, scale_factor=8, mode="nearest", align_mode=1)   # [1, 64, 160, 160]
        p4 = F.upsample(p4, scale_factor=4, mode="nearest", align_mode=1)   # [1, 64, 160, 160]
        p3 = F.upsample(p3, scale_factor=2, mode="nearest", align_mode=1)   # [1, 64, 160, 160]

        fuse = paddle.concat([p5, p4, p3, p2], axis=0)  #  [1, 256, 160, 160]  [4, 64, 160, 160]
        asf_feature = self.asf_conv(fuse)   # [1, 256, 160, 160]  [4, 4, 160, 160]
        attention = self.asf_attn(asf_feature)
        enhanced_feature = []               # [1, 256, 160, 160]
        for i, out in enumerate(fuse):
            enhanced_feature.append(attention[:, i:i + 1] * fuse[i])
        out = paddle.concat(enhanced_feature, dim=1)

        return out

class ScaleChannelSpatialAttention(nn.Layer):
    def __init__(self, in_channels, c_wise_channels, out_channels, **kwargs):
        super(ScaleChannelSpatialAttention, self).__init__()

        self.in_channels = in_channels          # 256*4  1024
        self.out_channels = out_channels        # 4

        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        weight_attr = paddle.nn.initializer.KaimingUniform()
        # Channel Wise
        self.channel_wise_1 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=c_wise_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)    # 'ReLU'
        self.channel_wise_2 = nn.Conv2D(
            in_channels=c_wise_channels,
            out_channels=in_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)   # 'Sigmoid'
        # Spatial Wise
        self.spatial_wise_1 = nn.Conv2D(
            in_channels=1,
            out_channels=1,
            kernel_size=3, 
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)   # 'ReLU'
        self.spatial_wise_2 = nn.Conv2D(
            in_channels=1,
            out_channels=1,
            kernel_size=1, 
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)   # 'Sigmoid'
        # Attention Wise
        self.attention_wise = nn.Conv2D(
            in_channels=self.in_channels,        #256
            out_channels=self.out_channels,       #4
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)   # 'Sigmoid'
        self.ReLU= paddle.nn.ReLU()
        self.Sigmoid= paddle.nn.Sigmoid()

    def forward(self,inputs):
        # inputs  [1, 256, 160, 160]
        x = self.avg_pool(inputs)    # [1, 256, 1, 1]  [4, 4, 1, 1]
        x = self.channel_wise_1(x)   # [1, 64, 1, 1]
        x = self.ReLU(x)
        x = self.channel_wise_2(x)    # [1, 256, 1, 1]
        x = self.Sigmoid(x)
        x = inputs + x              # [1, 256, 160, 160]

        inputs = paddle.mean(x, axis=1, keepdim=True)  # [1, 1, 160, 160]
        x = self.spatial_wise_1(inputs)   # [1, 1, 160, 160]       
        x = self.ReLU(x)
        x = self.spatial_wise_2(x)        # [1, 1, 160, 160]
        x = self.Sigmoid(x)
        x = inputs + x                    # [1, 1, 160, 160]   [1, 256, 160, 160]

        out = self.attention_wise(x)       #  256 4  [1, 4, 160, 160]
        out = self.Sigmoid(out)            # [1, 256, 160, 160]   [1, 4, 160, 160]
 
        return out  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
