import os
import torch.utils.data as data
from PIL import Image
from utils import transforms as tr


'''
Load all training and validation data paths
'''
def full_path_loader(data_dir):
    train_data = [i for i in os.listdir(data_dir + 'train/A/') if not
    i.startswith('.')]
    train_data.sort()

    valid_data = [i for i in os.listdir(data_dir + 'val/A/') if not
    i.startswith('.')]
    valid_data.sort()

    train_label_paths = []
    val_label_paths = []
    for img in train_data:
        train_label_paths.append(data_dir + 'train/OUT/' + img)
    for img in valid_data:
        val_label_paths.append(data_dir + 'val/OUT/' + img)


    train_data_path = []
    val_data_path = []

    for img in train_data:
        train_data_path.append([data_dir + 'train/', img])
    for img in valid_data:
        val_data_path.append([data_dir + 'val/', img])

    train_dataset = {}
    val_dataset = {}
    for cp in range(len(train_data)):
        train_dataset[cp] = {'image': train_data_path[cp],
                         'label': train_label_paths[cp]}
    for cp in range(len(valid_data)):
        val_dataset[cp] = {'image': val_data_path[cp],
                         'label': val_label_paths[cp]}


    return train_dataset, val_dataset

'''
Load all testing data paths
'''
def full_test_loader(data_dir):

    test_data = [i for i in os.listdir(data_dir + 'test/A/') if not
                    i.startswith('.')]
    test_data.sort()
    # print('-----------------', len(test_data))
    #test_data = my_sort(test_data)
    # print(test_data)
    test_label_paths = []
    for img in test_data:
        test_label_paths.append(data_dir + 'test/OUT/' + img)

    test_data_path = []
    for img in test_data:
        test_data_path.append([data_dir + 'test/', img])

    test_dataset = {}
    for cp in range(len(test_data)):
        test_dataset[cp] = {'image': test_data_path[cp],
                           'label': test_label_paths[cp]}

    return test_dataset

# 自定义排序函数
def my_sort(name_list):
    # print(name_list)
    #注意测试集格式如果是test-：
    nums = [int(i.split("_")[1].split(".")[0]) for i in name_list]
    #注意测试集格式如果是test：
    # nums = [int(i.split('.')[0][4:]) for i in name_list]

    nums.sort()
    #注意测试集格式如果是test—：
    sorted_names = [f"test_{num}.png" for num in nums]
    #注意测试集格式如果是test：
    # sorted_names = [f"test{num}.png" for num in nums]
    # print(sorted_names)
    return sorted_names

def cdd_loader(img_path, label_path, aug):
    dir = img_path[0]
    name = img_path[1]

    img1 = Image.open(dir + 'A/' + name)
    img2 = Image.open(dir + 'B/' + name)
    label = Image.open(label_path).convert('L')
    sample = {'image': (img1, img2), 'label': label}

    if aug:
        sample = tr.train_transforms(sample)
    else:
        sample = tr.test_transforms(sample)

    return sample['image'][0], sample['image'][1], sample['label']


class CDDloader(data.Dataset):

    def __init__(self, full_load, aug=False):

        self.full_load = full_load
        self.loader = cdd_loader
        self.aug = aug

    def __getitem__(self, index):

        img_path, label_path = self.full_load[index]['image'], self.full_load[index]['label']

        return self.loader(img_path,
                           label_path,
                           self.aug)

    def __len__(self):
        return len(self.full_load)
