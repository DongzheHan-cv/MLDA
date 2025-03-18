import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, utils
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image
import json
from models.Models import *

if __name__ == "__main__":
    # 定义tranform
    transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0., std=1.)
    ])

    # 加载图片
    image1 = Image.open("./xview-newalldata-fiveclass-128/test/A/test_10.png")
    image2 = Image.open("./xview-newalldata-fiveclass-128/test/B/test_10.png")

    # 加载模型
    # model = models.resnet18(pretrained=True)
    path = './tmp-20221024-xviewalldata-fiveclass-128-GMC-0.001/checkpoint_epoch_29.pt'   # the path of the model
    model= torch.load(path)
    # print(model)

    model_weights = []
    conv_layers = []
    model_children = list(model.children())
    print(model_children)
    counter = 0
    for i in range(len(model_children)):
        print('---------------------------------------')
        print(type(model_children[i]))
        print('---------------------------------------')
        conv_layers.append(model_children[i])
        # if type(model_children[i]) == nn.Conv2d:
        #     counter += 1
        #     model_weights.append(model_children[i].weight)
        #     conv_layers.append(model_children[i])
        # elif type(model_children[i]) == nn.Sequential:
        #     for j in range(len(model_children[i])):
        #         for child in model_children[i][j].children():
        #             if type(child) == nn.Conv2d:
        #                 counter+=1
        #                 model_weights.append(child.weight)
        #                 conv_layers.append(child)
        
        # if type(model_children[i]) == conv_block_nested:
            # print('shit')
            # for child in model_children[i].children():
            #     # print('999999999999999999999',type(child))
            #     if type(child) == conv3x3:
            #         for item in child.children():
            #             for itemc in item.children():
            #                 for itemcc in itemc:
            #                     if type(itemcc) == nn.Conv2d:
            #                         print('4444444444444444', type(itemcc))
            #                         counter+=1
            #                         model_weights.append(itemcc.weight)
            #                         conv_layers.append(itemcc)
        # elif type(model_children[i]) == ChannelAttention:
        #     for j in range(len(model_children[i])):
        #         for child in model_children[i][j].children():
        #             if type(child) == nn.Conv2d:
        #                 counter+=1
        #                 model_weights.append(child.weight)
        #                 conv_layers.append(child)
    
    print(f"Total convolution layers: {counter}")
    # print(conv_layers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    image1 = transform(image1)
    image2 = transform(image2)
    # print(f"Image shape before: {image.shape}")
    image1 = image1.unsqueeze(0)
    image2 = image2.unsqueeze(0)
    # print(f"Image shape after: {image.shape}")
    image1 = image1.to(device)
    image2 = image2.to(device)
    

    outputs = []
    names = []
    print(image1.shape)
    for layer in model_children[0:]:
        image = layer(image1)
        outputs.append(image)
        names.append(str(layer))
    print(len(outputs))
    
    #print feature_maps
    for feature_map in outputs:
        print(feature_map.shape)

    processed = []
    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map,0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())
    for fm in processed:
        print(fm.shape)

    fig = plt.figure(figsize=(30, 50))
    for i in range(len(processed)):
        a = fig.add_subplot(5, 4, i+1)
        imgplot = plt.imshow(processed[i])
        a.axis("off")
        a.set_title(names[i].split('(')[0], fontsize=30)
    plt.savefig(str('feature_maps.jpg'), bbox_inches='tight')