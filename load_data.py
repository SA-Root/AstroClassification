import os
import numpy as np
import pandas as pd

# 遍历所有文件夹下的文件
def walk_files(path, endpoint=None):
    data = []
    serial_number = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            # 单独读取每个数据，并记录下编号
            if file_path.endswith(endpoint):
                middle = file_path.split("\\")
                serial_number.append(middle[3].strip(".txt"))

                middle1 = np.loadtxt(file_path,delimiter=',')
                data.append(middle1)

    return data,serial_number


if __name__ == '__main__':

    # # 读取数据文件夹的地址
    # wav_path = r".\data\first_train_data"
    # # 读取数据
    # data_x,serial_number = walk_files(wav_path, endpoint=".txt")
    # data_x = np.array(data_x)
    # serial_number = np.array(serial_number,dtype=np.int_)
    # # 保存
    # np.save(r".\data\first_train_data_x.npy",data_x)
    # np.save(r".\data\first_train_data_serial_number.npy", serial_number)
    #
    serial_number = np.load(r".\data\first_train_data_serial_number.npy")
    serial_number = list(serial_number)
    # print(serial_number)
    y = pd.read_csv(r".\data\first_train_index_20180131.csv")
    y = pd.DataFrame(y)
    y.sort_values('id',inplace=True)
    data_y = y.loc[y['id'].isin(serial_number)]
    data_y = np.array(data_y['type'])
    np.save(r".\data\first_train_data_y.npy", data_y)










