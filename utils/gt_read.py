import os
import cv2
import numpy as np
path = '../xview/test/OUT'

gths = os.listdir(path)
index_img = 0

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

gths = my_sort(gths)

for gth in gths:
    print(gth)

    img = cv2.imread(os.path.join(path,gth))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(img.shape)
    size = 1024
    b = [[0 for col in range(size)] for row in range(size)]
    g = [[0 for col in range(size)] for row in range(size)]
    r = [[0 for col in range(size)] for row in range(size)]

    plate = img // 50
    for i in range(size):
        for j in range(size):
            if plate[i][j] == 0:
                b[i][j] = (138)
                g[i][j] = (0)
                r[i][j] = (1)
            elif plate[i][j] == 1:
                b[i][j] = (50)
                g[i][j] = (179)
                r[i][j] = (50)
            elif plate[i][j] == 2:
                b[i][j] = (189)
                g[i][j] = (250)
                r[i][j] = (254)
            elif plate[i][j] == 3:
                b[i][j] = (0)
                g[i][j] = (165)
                r[i][j] = (253)
            elif plate[i][j] == 4:
                b[i][j] = (0)
                g[i][j] = (41)
                r[i][j] = (253)


    r = np.array(r)
    g = np.array(g)
    b = np.array(b)
    # b = cd_preds.copy()
    # cd_preds = np.vstack((cd_preds,a,b,a))
    # print(cd_preds.shape)
    # cd_preds = cd_preds[0]* 50
    cd_preds = cv2.merge([b, g, r])
    # print(cd_preds)
    # file_path = './output_mixdata_epoch8/' + str(index_img).zfill(5)
    # cv2.imwrite(file_path + '.png', cd_preds)
    file_path = '../color-data/xview/' + 'test_' + str(index_img + 1)
    print(file_path)
    cv2.imwrite(file_path + '.png', cd_preds)

    index_img += 1
