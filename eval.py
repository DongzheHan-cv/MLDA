import torch.utils.data
from utils.parser import get_parser_with_args
from utils.helpers import get_test_loaders
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix

from utils.metric_tool import ConfuseMatrixMeter, cm2score, get_confuse_matrix
# The Evaluation Methods in our paper are slightly different from this file.
# In our paper, we use the evaluation methods in train.py. specifically, batch size is considered.
# And the evaluation methods in this file usually produce higher numerical indicators.

"""
加载参数配置: Initialize Parser and define arguments and set device
"""
parser, metadata = get_parser_with_args()
opt = parser.parse_args()
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# dev = torch.device('cpu')

"""
加载测试模型
"""
test_loader = get_test_loaders(opt)
#权参保存路径
path = './tmp/checkpoint_epoch_55.pt'   # the path of the model
#将权参赋值到模型中
model = torch.load(path)
#混淆矩阵计算# define some other vars to record the training states
bit_metric = ConfuseMatrixMeter(n_class=5)

count = 0
# c_matrix = {'acc': 0, 'miou': 0, 'mf1': 0, 'iou_0': 0, 'iou_1': 0, 'iou_2': 0, 'iou_3': 0, 'F1_0': 0, 'F1_1': 0, 'F1_2': 0, 'F1_3': 0, 'precision_0': 0, 'precision_1': 0, 'precision_2': 0, 'precision_3': 0, 'recall_0': 0, 'recall_1': 0, 'recall_2': 0, 'recall_3': 0}
# c_matrix = {'acc': 0, 'miou': 0, 'mf1': 0, 'iou_0': 0, 'iou_1': 0, 'iou_2': 0, 'F1_0': 0, 'F1_1': 0, 'F1_2': 0, 'precision_0': 0, 'precision_1': 0, 'precision_2': 0,  'recall_0': 0, 'recall_1': 0, 'recall_2': 0}
c_matrix = {'acc': 0, 'miou': 0, 'mf1': 0, 'iou_0': 0, 'iou_1': 0, 'iou_2': 0, 'iou_3': 0, 'iou_4': 0, 'F1_0': 0, 'F1_1': 0, 'F1_2': 0, 'F1_3': 0, 'F1_4': 0, 'precision_0': 0, 'precision_1': 0, 'precision_2': 0, 'precision_3': 0, 'precision_4': 0, 'recall_0': 0, 'recall_1': 0, 'recall_2': 0, 'recall_3': 0, 'recall_4': 0}
"""
Begin Testing
"""
model.eval()
#初始化评价指标
f1b = 0.0
#不进行梯度回传
with torch.no_grad():
    #使用tqdm库创建一个进度条来显示训练进度
    tbar = tqdm(test_loader)
    #输入图像：灾前、灾后、灾后灾损标签
    for batch_img1, batch_img2, labels in tbar:
        count += 1
        #Set variables for training这些输入数据转换为浮点张量并送入设备进行计算
        batch_img1 = batch_img1.float().to(dev)
        batch_img2 = batch_img2.float().to(dev)
        labels = labels.long().to(dev)

        #获得预测结果 Get model predictions
        cd_preds = model(batch_img1, batch_img2)
        cd_preds = cd_preds[-1]
        _, cd_preds = torch.max(cd_preds, 1)

        # aaa = confusion_matrix(labels.data.cpu().numpy().flatten(), cd_preds.data.cpu().numpy().flatten()).ravel()
        # print('=============================')
        # print(aaa)
        # tn, fp, fn, tp = multilabel_confusion_matrix(labels.data.cpu().numpy().flatten(),
        #                 cd_preds.data.cpu().numpy().flatten()).ravel()
        
        #计算混淆矩阵指标
        bit_metric.update_cm(cd_preds.data.cpu().numpy().flatten(), labels.data.cpu().numpy().flatten())
        scores, F1b = bit_metric.get_scores()
        f1b += F1b
        # break
        c_matrix['acc'] += scores['acc']
        c_matrix['miou'] += scores['miou']
        c_matrix['mf1'] += scores['mf1']

        c_matrix['iou_0'] += scores['iou_0']
        c_matrix['F1_0'] += scores['F1_0']
        c_matrix['precision_0'] += scores['precision_0']
        c_matrix['recall_0'] += scores['recall_0']

        c_matrix['iou_1'] += scores['iou_1']
        c_matrix['F1_1'] += scores['F1_1']
        c_matrix['precision_1'] += scores['precision_1']
        c_matrix['recall_1'] += scores['recall_1']

        c_matrix['iou_2'] += scores['iou_2']
        c_matrix['F1_2'] += scores['F1_2']
        c_matrix['precision_2'] += scores['precision_2']
        c_matrix['recall_2'] += scores['recall_2']

        c_matrix['iou_3'] += scores['iou_3']
        c_matrix['F1_3'] += scores['F1_3']
        c_matrix['precision_3'] += scores['precision_3']
        c_matrix['recall_3'] += scores['recall_3']
     
        c_matrix['iou_4'] += scores['iou_4']
        c_matrix['F1_4'] += scores['F1_4']
        c_matrix['precision_4'] += scores['precision_4']
        c_matrix['recall_4'] += scores['recall_4']
     
print('===================================== 测试得分 ============================================')
print("F1b: ", f1b / count)
print('acc: ', c_matrix['acc'] / count )

# print('miou: ', c_matrix['miou'] / count )
print('mf1: ', c_matrix['mf1'] / count )
print()
# print('iou_0: ', c_matrix['iou_0'] / count )
print('F1_0: ', c_matrix['F1_0'] / count )
print('precision_0: ', c_matrix['precision_0'] / count )
print('recall_0: ', c_matrix['recall_0'] / count )
print()
# print('iou_1: ', c_matrix['iou_1'] / count )
print('F1_1: ', c_matrix['F1_1'] / count )
print('precision_1: ', c_matrix['precision_1'] / count )
print('recall_1: ', c_matrix['recall_1'] / count )
print()
# print('iou_2: ', c_matrix['iou_2'] / count )
print('F1_2: ', c_matrix['F1_2'] / count )
print('precision_2: ', c_matrix['precision_2'] / count )
print('recall_2: ', c_matrix['recall_2'] / count )
print()

print('F1_3: ', c_matrix['F1_3'] / count )
print('precision_3: ', c_matrix['precision_3'] / count )
print('recall_3: ', c_matrix['recall_3'] / count )
print()

#c_matrix该类别的总值，/count = 均值
print('F1_4: ', c_matrix['F1_4'] / count )
print('precision_4: ', c_matrix['precision_4'] / count )
print('recall_4: ', c_matrix['recall_4'] / count )
print('============================================================================================')