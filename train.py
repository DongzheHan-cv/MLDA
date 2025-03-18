import datetime
import torch
from sklearn.metrics import precision_recall_fscore_support as prfs
from utils.parser import get_parser_with_args
from utils.helpers import (get_loaders, get_criterion,
                           load_model, initialize_metrics, get_mean_metrics,
                           set_metrics)
import os
import logging
import json
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random
import numpy as np


"""
加载参数配置: Initialize Parser and define arguments
"""
parser, metadata = get_parser_with_args()
opt = parser.parse_args()

"""
加载日志配置:Initialize experiments log
"""
logging.basicConfig(level=logging.INFO)
writer = SummaryWriter(opt.log_dir + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

"""
设置数据与训练硬件:Set up environment: define paths, download data, and set device
"""
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# dev = torch.device('cpu')
logging.info('GPU AVAILABLE? ' + str(torch.cuda.is_available()))

#传入一个特定的种子值，可以确保在不同的运行中产生相同的随机数序列，从而使得实验结果可重现
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(seed=777)

"""
加载模型和训练参数
"""
train_loader, val_loader = get_loaders(opt)

logging.info('LOADING Model')
#加载模型
model = load_model(opt, dev)
#损失函数
criterion = get_criterion(opt)  
#优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate) # Be careful when you adjust learning rate, you can refer to the linear scaling rule
#学习率衰减策略：三个参数分别：使用优化器对象、多少轮循环后更新一次学习率、每次更新lr的gamma倍
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
#初始化评价指标
best_metrics = {'cd_f1scores': -1, 'cd_recalls': -1, 'cd_precisions': -1}
#开始训练
logging.info('STARTING training')
#初始化步长
total_step = -1

for epoch in range(opt.epochs):
    #初始化评价指标'cd_losses': [],'cd_corrects': [],'cd_precisions': [],'cd_recalls': [],'cd_f1scores': [],'learning_rate': [],
    train_metrics = initialize_metrics()   
    val_metrics = initialize_metrics()
    
    """
    Begin Training
    """
    model.train()
    logging.info('SET model mode to train!')
    #初始化迭代器
    batch_iter = 0
    #使用tqdm库创建一个进度条来显示训练进度
    tbar = tqdm(train_loader) 
    #输入图像：灾前、灾后、灾后灾损标签
    for batch_img1, batch_img2, labels in tbar:
        #print('111111111111111111111111111111111111111111')
        #print(batch_img1.shape,batch_img2.shape,labels.shape)
        #日志输出训练进度
        tbar.set_description("epoch {} info ".format(epoch) + str(batch_iter) + " - " + str(batch_iter+opt.batch_size))
        batch_iter = batch_iter+opt.batch_size
        total_step += 1

        # Set variables for training这些输入数据转换为浮点张量并送入设备进行计算
        batch_img1 = batch_img1.float().to(dev)
        batch_img2 = batch_img2.float().to(dev)
        labels = labels.long().to(dev)
    
        #梯度清零 Zero the gradient
        optimizer.zero_grad()

        #获得预测结果 Get model predictions
        cd_preds = model(batch_img1, batch_img2) #这里batch_img1, batch_img2会送入model.foward函数，foward函数需要留两个接口

        #计算损失loss， calculate loss 
        cd_loss = criterion(cd_preds, labels) #损失函数criterion可以根据具体任务选择合适的损失函数
        loss = cd_loss

        #反向传播backprop
        loss.backward()
        #更新模型的参数
        optimizer.step() 

        #cd_preds是一个元组类型，需要从中选择最后一个元素作为最终的预测结果
        cd_preds = cd_preds[-1]
        #通过torch.max函数找到最大值所在的索引，得到最终的预测标签cd_preds
        _, cd_preds = torch.max(cd_preds, 1)
        #根据预测标签和真实标签计算其他的批处理指标，包括准确率、召回率和F1-score等
        # Calculate and log other batch metrics
        cd_corrects = (100 *
                       (cd_preds.squeeze().byte() == labels.squeeze().byte()).sum() /
                       (labels.size()[0] * (opt.patch_size**2)))
        #计算每个标签的指标，然后找到它们的平均值（不考虑标签的不平衡性）
        #如果有些标签在真实样本中没有出现，那么这些标签的召回率和F-score就无法定义(因为他们的分母为0），默认被设置为0.0。
        cd_train_report = prfs(labels.data.cpu().numpy().flatten(),
                               cd_preds.data.cpu().numpy().flatten(),
                            #    average='binary',
                                average='macro',zero_division=1)
                            #    pos_label=1)#可以添加zero_division=1 参数，当召回率和F-score无法定义时，将其设置为1.0

        train_metrics = set_metrics(train_metrics,
                                    cd_loss,
                                    cd_corrects,
                                    cd_train_report,
                                    scheduler.get_last_lr())

        # log the batch mean metrics 输出batch平均值
        mean_train_metrics = get_mean_metrics(train_metrics)
        #将这些批处理指标记录下来，并用于监控训练过程
        for k, v in mean_train_metrics.items():
            writer.add_scalars(str(k), {'train': v}, total_step)

        # clear batch variables from memory #在每个batch的结束时，及时释放内存中的变量，以避免内存溢出的问题
        del batch_img1, batch_img2, labels

    #更新学习率，并记录当前epoch的训练指标
    scheduler.step() 
    logging.info("EPOCH {} TRAIN METRICS".format(epoch) + str(mean_train_metrics))
    
    """
    Begin Validation
    """
    model.eval()
    #不进行梯度回传
    with torch.no_grad():
        for batch_img1, batch_img2, labels in val_loader:
            # Set variables for training 这些输入数据转换为浮点张量并送入设备进行计算
            batch_img1 = batch_img1.float().to(dev)
            batch_img2 = batch_img2.float().to(dev)
            labels = labels.long().to(dev)

            # Get predictions and calculate loss 获得预测结果，计算损失
            cd_preds = model(batch_img1, batch_img2)

            cd_loss = criterion(cd_preds, labels)

            cd_preds = cd_preds[-1]
            _, cd_preds = torch.max(cd_preds, 1)

            # Calculate and log other batch metrics 根据预测结果，计算其他评价指标
            cd_corrects = (100*
                           (cd_preds.squeeze().byte() == labels.squeeze().byte()).sum() /
                           (labels.size()[0] * (opt.patch_size**2)))

            cd_val_report = prfs(labels.data.cpu().numpy().flatten(),
                                 cd_preds.data.cpu().numpy().flatten(),
                                #  average ='binary',)
                                 average='macro',zero_division=1)
                                #  pos_label=1)

            val_metrics = set_metrics(val_metrics,
                                      cd_loss,
                                      cd_corrects,
                                      cd_val_report,
                                      scheduler.get_last_lr())

            # log the batch mean metrics 输出batch平均值
            mean_val_metrics = get_mean_metrics(val_metrics)

            for k, v in mean_train_metrics.items():
                writer.add_scalars(str(k), {'val': v}, total_step)

            # clear batch variables from memory 清除缓存
            del batch_img1, batch_img2, labels

        logging.info("EPOCH {} VALIDATION METRICS".format(epoch)+str(mean_val_metrics))

        """
        if当前轮指标>best指标,则保存为当前最好模型权重参数. Store the weights of good epochs based on validation results 
        """
        if ((mean_val_metrics['cd_precisions'] > best_metrics['cd_precisions'])
                or
                (mean_val_metrics['cd_recalls'] > best_metrics['cd_recalls'])
                or
                (mean_val_metrics['cd_f1scores'] > best_metrics['cd_f1scores'])):

            # Insert training and epoch information to metadata dictionary
            logging.info('updata the model')
            print('updata the model')
            #每次比上次best好，就会保存下来新的权重参数。
            metadata['validation_metrics'] = mean_val_metrics

            # Save model and log
            if not os.path.exists('./tmp'):
                os.mkdir('./tmp')
            with open('./tmp/metadata_epoch_' + str(epoch) + '.json', 'w') as fout:
                json.dump(metadata, fout)

            torch.save(model, './tmp/checkpoint_epoch_'+str(epoch)+'.pt')

            # comet.log_asset(upload_metadata_file_path)
            best_metrics = mean_val_metrics #更新best。这里有一个问题，这个best不一定是所有训练中的best。所以，53轮的测试值比59轮的好

        print('An epoch finished.')
        
writer.close()  # close tensor board
print('Done!')

