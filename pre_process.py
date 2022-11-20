# 规格化
import pandas as pd

INTERVAL_BOTTOM = 0
INTERVAL_TOP = 10000


def pre_process(oldfile, newfile):
    df = pd.read_csv(oldfile, header=None)
    df_new = pd.DataFrame()
    discard_num_0 = 0
    #discard_num_1 = 0

    # 将label列数值化
    col_label = df[df.columns.size-1]
    df = df.drop(columns=df.columns.size-1, axis=1)
    col_label_new = col_label.map(lambda x: 0 if x.startswith('normal') else 1)

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
        col_new = col.map(lambda x: (x-min_mem)*(INTERVAL_TOP -
                                                 INTERVAL_BOTTOM)/(max_mem-min_mem) if not abs(max_mem-min_mem) < 1e-7 else min_mem)
        # df_new.add(col)
        df_new = pd.concat([df_new, col_new], axis=1)
    # 添加label列
    df_new = pd.concat([df_new, col_label_new], axis=1)
    print("discard column num:"+str(discard_num_0))  # +"+"+str(discard_num_1)
    print(df_new.shape)

    f = open(newfile, "w")
    for row in df_new.values.tolist():
        f.write(','.join(('%.2f' % x if not str(x).isdigit() else str(int(x)))
                for x in row))
        f.write("\n")
    f.close()


pre_process("test_set_pre.csv", "test_set.csv")
pre_process("train_set_pre.csv", "train_set.csv")
