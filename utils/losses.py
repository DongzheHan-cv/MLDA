from utils.parser import get_parser_with_args
from utils.metrics import FocalLoss, dice_loss

import torch
import torch.nn.functional as F

parser, metadata = get_parser_with_args()
opt = parser.parse_args()

def hybrid_loss(predictions, target):
    """Calculating the loss"""
    loss = 0

    # gamma=0, alpha=None --> CE
    # focal = FocalLoss(gamma=0, alpha=None)
    focal = FocalLoss(gamma=2, alpha=0.25)
    # focal = MultiCEFocalLoss(class_num=3,gamma=0, alpha=None)

    for prediction in predictions:
        bce = focal(prediction, target)
        
        # dice = dice_loss(prediction, target)
        # loss += bce + dice 
        loss += bce

    return loss



def cross_entropy(input, target, weight=None, reduction='mean',ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    target = target.long()
    input = input[-1]
    # print(input,'111111111111111')
    # print(target,'2222222222222222')
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        #双线性差值
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)
