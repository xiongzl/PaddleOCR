import paddle 
from ppocr.modeling.backbones.det_resnet_vd import ResNet
from ppocr.modeling.necks.db_fpn import DBFPN,DBASFFPN


fake_inputs = paddle.randn([1, 3, 640, 640], dtype="float32")
model_backbone = ResNet()
in_channles = model_backbone.out_channels

# model_fpn = DBFPN(in_channels=in_channles, out_channels=256)
model_fpn = DBASFFPN(in_channels=in_channles, out_channels=256)
# print(model_fpn)
 
outs = model_backbone(fake_inputs)
# print(outs)
fpn_outs = model_fpn(outs)
 
# print(f"The shape of fpn outs {fpn_outs.shape}")
