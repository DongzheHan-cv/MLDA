
import torch, torchvision
from torchvision.models import resnet50
from thop import profile


# model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
print("Model:\n{}".format(model))
input = torch.randn(1, 3, 256, 256) #模型输入的形状,batch_size=1
flops, params = profile(model, inputs=(input, ))
print(flops/1e9,params/1e6) #flops单位G，para单位M
