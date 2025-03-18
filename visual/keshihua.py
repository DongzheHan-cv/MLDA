from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torchvision.models as models
from models.Models import Siam_NestedUNet_Conc, SNUNet_ECAM
import torch.nn as nn

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def draw_CAM(model,img1_path,img2_path, save_path,resize=227,isSave=False,isShow=False):
    # 图像加载&预处理
    img1=Image.open(img1_path).convert('RGB')
    img2=Image.open(img2_path).convert('RGB')
    loader = transforms.Compose([transforms.Resize(size=(resize,resize)),transforms.ToTensor()]) 
    loader2 = transforms.Compose([transforms.Resize(size=(resize,resize)),transforms.ToTensor()]) 
    img1 = loader(img1).unsqueeze(0).to(dev) # unsqueeze(0)在第0维增加一个维度
    img2 = loader2(img2).unsqueeze(0).to(dev)

    
    # 获取模型输出的feature/score
    model.eval().to(dev) # 测试模式，不启用BatchNormalization和Dropout
    # feature=model.features(img)
    feature1 = a(img1)
    feature2 = a(img2)
    # print(feature.shape)
    # print(feature.view(1,-1).shape)
    feature1 = feature1.view(1,-1)
    feature2 = feature2.view(1,-1)
    output=model(feature1,feature2)
    
    # 预测得分最高的那一类对应的输出score
    pred = torch.argmax(output).item()
    pred_class = output[:, pred]

    # 记录梯度值
    def hook_grad(grad):
        global feature_grad
        feature_grad=grad
    feature1.register_hook(hook_grad)
    # 计算梯度
    pred_class.backward()
    
    grads=feature_grad # 获取梯度
    
    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1)) # adaptive_avg_pool2d自适应平均池化函数,输出大小都为（1，1）

    # 此处batch size默认为1，所以去掉了第0维（batch size维）
    pooled_grads = pooled_grads[0] # shape为[batch,通道,size,size],此处batch为1，所以直接取[0]即取第一个batch的元素，就取到了每个batch内的所有元素
    features = feature1[0] # 取【0】原因同上
    
    ########################## 导数（权重）乘以相应元素
    # 512是最后一层feature的通道数
    for i in range(128):
        features[i, ...] *= pooled_grads[i, ...] # features[i, ...]与features[i]效果好像是一样的，都是第i个元素
    ##########################
    
    # 绘制热力图
    heatmap = features.detach().numpy()
    heatmap = np.mean(heatmap, axis=0) # axis=0,对各列求均值，返回1*n
 
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    # 可视化原始热力图
    if isShow:
        plt.matshow(heatmap)
        plt.show()
        
    img = cv2.imread(img1_path)  # 用cv2加载原始图像
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子
    # 将图像保存到硬盘
    if isSave:
        cv2.imwrite(save_path, superimposed_img)  
    # 展示图像
    if isShow:
        superimposed_img/=255
        plt.imshow(superimposed_img)

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # self.net = models.googlenet(pretrained=True)
        path = './tmp-20221024-xviewalldata-fiveclass-128-GMC-0.001/checkpoint_epoch_29.pt'   # the path of the model
        self.net = torch.load(path)
        # If you treat GooLeNet as a fixed feature extractor, disable the gradients and save some memory
        for p in self.net.parameters():
            p.requires_grad = False
        # Define which layers you are going to extract
        self.features = nn.Sequential(*list(self.net.children())[:4])
        print(self.features)

    def forward(self, x):
        return self.features(x)

a = FeatureExtractor()
    # model=models.vgg16()
path = './tmp-20221024-xviewalldata-fiveclass-128-GMC-0.001/checkpoint_epoch_29.pt'
model = torch.load(path).to(dev)
# draw_CAM(model,'/Users/liuyanzhe/Study/陶瓷研究相关/陶瓷数据/ceramic_data/训练/1牡丹/mudan15.png','/Users/liuyanzhe/Downloads/热力图2.png',isSave=True,isShow=True)
draw_CAM(model,'./xview-newalldata-fiveclass-128/test/A/test_10.png','./xview-newalldata-fiveclass-128/test/B/test_10.png','./output-keshihua/111.png',isSave=True,isShow=True)

