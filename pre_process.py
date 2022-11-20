import math
import pandas as pd


INTERVAL_BOTTOM = 0
INTERVAL_TOP = 1000

TRAIN_SET_SIZE = 3200
TEST_NORMAL_SIZE = 800
TEST_SMURF_SIZE = 800

NORMAL_TYPE = 0
SMURF_TYPE = 1

TMP_FILE = 'tmp'
RAW_DATA_FILE = "kddcup.data_10_percent"
TRAIN_FILE = "train_set.csv"
TEST_FILE = "test_set.csv"

# 从原始数据中截取
f = open(RAW_DATA_FILE, 'r')
line = 0
l = ''
data = []
while True:
    l = f.readline()
    if l == '':
        break
    data.append(l)
length = len(data)
normal_set = []
smurf_set = []
normal_num = 0
smurf_num = 0
for i in data:
    label = i.split(",")[-1]
    flag0 = False
    flag1 = False
    if label.startswith("normal"):
        if normal_num < TRAIN_SET_SIZE+TEST_NORMAL_SIZE:
            normal_set.append(i)
            normal_num += 1
        else:
            flag0 = True
    elif label.startswith("smurf"):
        if smurf_num < TEST_SMURF_SIZE:
            smurf_set.append(i)
            smurf_num += 1
        else:
            flag1 = True
    if flag0 and flag1:  # 两个都满了
        break

print("##################################################")
print("读取原始数据...")
print("Num of normal data:"+str(len(normal_set)))
print("Num of smurf data:"+str(len(smurf_set)))


############


# 将label列数值化，去掉字符型的列
def pre_process(file):
    df = pd.read_csv(file, header=None)
    df_new = pd.DataFrame()
    discard_num_0 = 0

    # 将label列数值化
    col_label = df[df.columns.size-1]
    df = df.drop(columns=df.columns.size-1, axis=1)
    col_label_new = col_label.map(
        lambda x: NORMAL_TYPE if x.startswith('normal') else SMURF_TYPE)

    for i in range(df.columns.size):
        col = df[i]
        if not str(col[0]).isdigit():
            # 不是整数
            l = str(col[0]).split(".")
            if not (len(l) == 2 and l[0] != None and l[0] != '' and l[1] != None and l[1] != '' and l[0].isdigit() and l[1].isdigit()):
                # 不是浮点数
                discard_num_0 += 1
                continue

        max_mem = max(col)
        min_mem = min(col)
        # if abs(max_mem-min_mem) < 1e-7:  # 去除完全一致的列
        #    discard_num_1 += 1
        #    continue
        col_new = pd.Series(dtype='float64')
        col_new = col.map(lambda x: ((x-min_mem)*(INTERVAL_TOP -
                                                  INTERVAL_BOTTOM)/(max_mem-min_mem)) if not abs(max_mem-min_mem) < 1e-7 else (min_mem))
        df_new = pd.concat([df_new, col_new], axis=1)
    # 添加label列
    df_new = pd.concat([df_new, col_label_new], axis=1)
    print("discard column num:"+str(discard_num_0))  # +"+"+str(discard_num_1)
    print(df_new.shape)
    return df_new.values.tolist()


whole_set_raw = normal_set + smurf_set
tmp_file = open("tmp", "w")
tmp_file.write(''.join(whole_set_raw))
tmp_file.close()

print("##################################################")
print("对原始数据进行预处理...")
whole_set_pro = pre_process(TMP_FILE)
whole_set_str = list()
train_set = list()
test_normal_set = list()
test_smurf_set = list()
normal_set = list()
smurf_set = list()
for row in whole_set_pro:
    str_row = pd.Series(row).map(lambda x: '%.4f' % x)
    if abs((row[-1])-NORMAL_TYPE) < 1e-7:
        normal_set.append(str_row)
    else:
        smurf_set.append(str_row)
# 划分训练和测试集
train_set = normal_set[:TRAIN_SET_SIZE]
test_normal_set = normal_set[TRAIN_SET_SIZE:TRAIN_SET_SIZE+TEST_NORMAL_SIZE]
test_smurf_set = smurf_set[:TEST_SMURF_SIZE]


print("Size of train dataset:"+str(len(train_set)) +
      " , colnum:"+str(len(train_set[0])))
print("Number of normal data in test dataset:"+str(len(test_normal_set)) +
      " , colnum:"+str(len(test_normal_set[0])))
print("Number of smurf data in test dataset:"+str(len(test_smurf_set)) +
      " , colnum:"+str(len(test_smurf_set[0])))
print("预处理完成")


############
f.close()
f = open(TRAIN_FILE, "w")
f.write('\n'.join(','.join(x) for x in train_set))
f.close()
f = open(TEST_FILE, 'w')
f.write('\n'.join(','.join(x) for x in test_normal_set))
f.write("\n")
f.write('\n'.join(','.join(x) for x in test_smurf_set))
f.close()
