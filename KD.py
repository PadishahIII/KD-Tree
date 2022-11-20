import math
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# label type
SMURF_TYPE = 1
NORMAL_TYPE = 0

# 阈值
THRESHOLD = 1200


class KD_node:
    def __init__(self, point=None, splitDim=None, left=None, right=None):
        self.point = point  # 数据点的特征向量
        self.splitDim = splitDim  # 切分的维度
        self.left = left  # 左儿子
        self.right = right  # 右儿子


def BuildKDTree(root, data):
    length = len(data)
    if length == 0:
        return
    # 方差
    max_var = 0
    dimension = len(data[0]) - 1
    splitDim = 0
    for i in range(1, dimension):
        d_list = []
        for t in data:
            d_list.append(t[i])
        var = CalVariance(d_list)
        if var > max_var:
            max_var = var
            splitDim = i

    # 根据划分域的数据对数据点进行排序
    data.sort(key=lambda t: t[splitDim])
    # data = np.array(data)
    # 选择下标为len / 2的点作为分割点
    x = int(length / 2)
    point = data[x]
    root = KD_node(point, splitDim)
    # 递归的对切分到左儿子和右儿子的数据再建树
    x1 = int(x / 2)
    root.left = BuildKDTree(root.left, data[0:x1])
    root.right = BuildKDTree(root.right, data[(x1 + 1):length])
    return root


def CalVariance(l):
    l = list(map(float, l))
    return np.var(np.array(l))


def CalDistance(pt1, pt2):  # 欧式距离
    pt1 = list(map(float, pt1))
    pt2 = list(map(float, pt2))
    vt1 = np.array(pt1)
    vt2 = np.array(pt2)
    return np.sqrt(np.sum(np.square(vt1 - vt2)))


def searchANN(root, target):
    # 初始化为root的节点
    nearest_point = root.point  # 最近邻点的特征向量
    min_dist = CalDistance(target, nearest_point)
    nodeList = []
    current_node = root
    # 二分查找
    while current_node:
        nodeList.append(current_node)
        distance = CalDistance(target, current_node.point)
        if min_dist > distance:
            nearest_point = current_node.point
            min_dist = distance
        splitDim = current_node.splitDim

        if target[splitDim] <= current_node.point[splitDim]:
            current_node = current_node.left
        else:
            current_node = current_node.right

    # 回溯查找
    while nodeList:
        back_point = nodeList.pop()
        back_splitDim = back_point.splitDim
        if abs(target[back_splitDim] - back_point.point[back_splitDim]) < min_dist:
            if target[back_splitDim] < back_point.point[back_splitDim]:
                current_node = back_point.right
            else:
                current_node = back_point.left
            if current_node:
                nodeList.append(current_node)
                curDist = CalDistance(target, current_node.point)
                if min_dist > curDist:
                    min_dist = curDist
                    nearest_point = current_node.point
    return nearest_point


# 数据预处理


def import_data(path):
    df = pd.read_csv(path, header=None)
    df_no_label = df.drop(columns=df.columns.size-1, axis=1)  # 去掉label
    return df.values.tolist(), df_no_label.values.tolist()


def main_process(test_data_with_label, train_data_no_label, test_data_no_label, test_smurf_num, test_normal_num):
    if len(test_data_no_label) != len(test_data_with_label):
        raise Exception("测试数据有误")
    train_start = time.time()
    root = KD_node()
    root = BuildKDTree(root, train_data_no_label)  # 用无标签数据建模
    train_end = time.time()
    train_time = train_end - train_start
    smurf_recognized = 0  # 测试出的smurf数目
    normal_recognized = 0
    smurf_from_normal = 0  # 本为normal，误报成smurf
    normal_from_smurf = 0  # 本为smurf，漏报成normal
    test_time_start = time.time()

    dis_list_normal = []
    dis_list_smurf = []

    def is_normal(point):
        nearest = searchANN(root, point)
        dis = CalDistance(point, nearest)
        if dis < THRESHOLD:
            return True, dis
        else:
            return False, dis

    for i in range(len(test_data_with_label)):
        # searchANN(root, test_data_with_label[i])
        res, dis = is_normal(test_data_no_label[i])
        if(abs(test_data_with_label[i][-1] - NORMAL_TYPE) < 1e-7):
            dis_list_normal.append(dis)
        else:
            dis_list_smurf.append(dis)
        if res is True and abs(test_data_with_label[i][-1] - NORMAL_TYPE) < 1e-7:
            normal_recognized += 1
        elif res is True and abs(test_data_with_label[i][-1]-SMURF_TYPE) < 1e-7:
            normal_recognized += 1
            normal_from_smurf += 1
        elif res is False and abs(test_data_with_label[i][-1]-NORMAL_TYPE) < 1e-7:
            smurf_recognized += 1
            smurf_from_normal += 1
        elif res is False and abs(test_data_with_label[i][-1]-SMURF_TYPE) < 1e-7:
            smurf_recognized += 1
    test_time_end = time.time()
    test_time = test_time_end - test_time_start
    DR = smurf_recognized / test_smurf_num  # 检测率
    FPR = smurf_from_normal / test_normal_num  # 误报率
    return train_time, test_time, DR, FPR


train_set_with_label, train_set_no_label = import_data(
    'train_set.csv')  # list(list)
test_set_with_label, test_set_no_label = import_data('test_set.csv')
train_size = len(train_set_with_label)
test_smurf_num = 0
test_normal_num = 0
for label in test_set_with_label:
    if abs(label[-1]-SMURF_TYPE) < 1e-7:
        test_smurf_num += 1
    else:
        test_normal_num += 1

print("Size of train dataset:" + str(train_size))
print("Size of test dataset:" + str(len(test_set_with_label)))
print("Number of smurf data in test dataset:" + str(test_smurf_num))
print("Number of normal data in test dataset:" + str(test_normal_num))
train_time, test_time, DR, FPR = main_process(
    test_data_with_label=test_set_with_label, train_data_no_label=train_set_no_label, test_data_no_label=test_set_no_label, test_smurf_num=test_smurf_num, test_normal_num=test_normal_num)

threshold_list = list()
DR_list = list()
FPR_list = list()
# 优化Threshold
# for THRESHOLD in range(1006, 1539, 100):
#    threshold_list.append(THRESHOLD)
#    train_time, test_time, DR, FPR = main_process(
#        test_data_with_label=test_set_with_label, train_data_no_label=train_set_no_label, test_data_no_label=test_set_no_label, test_smurf_num=test_smurf_num, test_normal_num=test_normal_num)
#    DR_list.append(DR)
#    FPR_list.append(FPR)
#l1 = plt.plot(threshold_list, DR_list, 'g--', label='DR')
#l2 = plt.plot(threshold_list, FPR_list, 'r--', label='FPR')
# plt.xlabel("threshold")
# plt.ylabel("radio")
# plt.legend()
# plt.show()
print("Time used for train:" + str(train_time) + "s")
print("Time used for test:" + str(test_time) + "s")
print("检测率DR:"+str('%.4f' % DR))
print("误报率FPR:"+str('%.4f' % FPR))
