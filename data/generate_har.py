# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import numpy as np
import os
import random

from data.utils.dataset_utils import noisify_label
from utils.HAR_utils import *

random.seed(1)
np.random.seed(1)
data_path = "har/"
dir_path = "har/"
noise_client_ratio = 0.5
add_noise_type = "symmetric"


def generate_har(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "global_test/"
    noisy_path = dir_path + "noisy.json"
    public_path = dir_path + "public/"

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    if not os.path.exists(public_path):
        os.makedirs(public_path)

    # download data
    if not os.path.exists(data_path + 'rawdata/UCI HAR Dataset.zip'):
        os.system(
            f"wget https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip -P {data_path}rawdata/")
    if not os.path.exists(data_path + 'rawdata/UCI HAR Dataset/'):
        os.system(f"unzip {data_path}rawdata/'UCI HAR Dataset.zip' -d {data_path}rawdata/")

    X, y = load_data_har(data_path + 'rawdata/')
    statistic = []
    num_clients = len(y)
    num_classes = len(np.unique(np.concatenate(y, axis=0)))
    for i in range(num_clients):
        statistic.append([])
        for yy in sorted(np.unique(y[i])):
            idx = y[i] == yy
            statistic[-1].append((int(yy), int(len(X[i][idx]))))

    for i in range(num_clients):
        print(f"Client {i}\t Size of data: {len(X[i])}\t Labels: ", np.unique(y[i]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[i]])
        print("-" * 50)

    train_data, test_data = split_data(X, y)

    public_data_size_per_client = 0.1

    # 初始化存储公共数据集的列表
    public_data = {'x': [], 'y': []}

    # 遍历每个客户端的数据
    for client_idx, client_data in enumerate(train_data):
        # 计算当前客户端的样本数量
        client_data_size = len(client_data['y'])

        # 计算当前客户端应该抽取的样本数量
        public_samples = int(client_data_size * public_data_size_per_client)

        # 从客户端数据中随机抽取 10% 的样本（确保可重现性）
        np.random.seed(1 + client_idx)  # 为每个客户端设置不同但固定的种子
        selected_indices = np.random.choice(client_data_size, public_samples, replace=False)

        # 将抽取的数据添加到公共数据集中
        public_data['x'].extend([client_data['x'][i] for i in selected_indices])
        public_data['y'].extend([client_data['y'][i] for i in selected_indices])

        # 从客户端数据中删除已经被抽取的样本
        client_data['x'] = [client_data['x'][i] for i in range(client_data_size) if i not in selected_indices]
        client_data['y'] = [client_data['y'][i] for i in range(client_data_size) if i not in selected_indices]
    data = {
        'x': np.array(public_data['x']),
        'y': np.array(public_data['y'])
    }
    np.savez_compressed(public_path + '/public_dataset.npz', data=data)

    # 可选：检查保存的内容
    print(f"公共数据集已经保存为 {public_path}/public_dataset.npz")

    add_noise = True
    if add_noise == True:
        # 重新设置随机种子确保噪声生成的可重现性
        np.random.seed(1)
        noise_ratios = np.random.uniform(0.3, 0.7, num_clients)
        num_noisy_clients = int(num_clients * noise_client_ratio)
        noisy_client_indices = np.random.choice(num_clients, num_noisy_clients, replace=False)
        noise_ratios_dict = {f"client_{i}": (float(noise_ratios[i]) if i in noisy_client_indices else 0.0)
                             for i in range(num_clients)}

        noisy_sample_indices_dict = {f"client_{i}": [] for i in range(num_clients)}
        # assign data
        for client in range(num_clients):
            if add_noise and client in noisy_client_indices:
                # 为每个客户端设置固定但不同的随机种子确保可重现性
                np.random.seed(1 + client)
                per_client_noise_ratio = noise_ratios[client]
                num_noisy_labels = int(len(train_data[client]['y']) * per_client_noise_ratio)
                noisy_indices = np.random.choice(len(train_data[client]['y']), size=num_noisy_labels, replace=False)
                noisy_label_pairs = []
                for i in range(len(noisy_indices)):
                    noise_need_idx = noisy_indices[i]
                    true_label = train_data[client]['y'][noise_need_idx]
                    noisy_label = noisify_label(true_label, num_classes=num_classes, noise_type=add_noise_type)
                    noisy_label_pairs.append((noise_need_idx, true_label, noisy_label))
                noisy_sample_indices_dict[f"client_{client}"] = [
                    {"index": int(idx), "true_label": int(t), "noisy_label": int(n)}
                    for (idx, t, n) in noisy_label_pairs
                ]
                for (idx, _, noisy_label) in noisy_label_pairs:
                    train_data[client]['y'][idx] = noisy_label
            for i in np.unique(train_data[client]['y']):
                statistic[client].append((int(i), int(sum(train_data[client]['y'] == i))))
        if noise_ratios_dict is not None:
            with open(noisy_path, 'w') as f:
                ujson.dump(noise_ratios_dict, f, indent=4)
            noisy_indices_path = noisy_path.replace(".json", "_indices.json")
            with open(noisy_indices_path, 'w') as f:
                ujson.dump(noisy_sample_indices_dict, f, indent=4)
    # gc.collect()

    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, statistic)


def load_data_har(data_folder):
    str_folder = data_folder + 'UCI HAR Dataset/'
    INPUT_SIGNAL_TYPES = [
        "body_acc_x_",
        "body_acc_y_",
        "body_acc_z_",
        "body_gyro_x_",
        "body_gyro_y_",
        "body_gyro_z_",
        "total_acc_x_",
        "total_acc_y_",
        "total_acc_z_"
    ]

    str_train_files = [str_folder + 'train/' + 'Inertial Signals/' + item + 'train.txt' for item in
                       INPUT_SIGNAL_TYPES]
    str_test_files = [str_folder + 'test/' + 'Inertial Signals/' +
                      item + 'test.txt' for item in INPUT_SIGNAL_TYPES]
    str_train_y = str_folder + 'train/y_train.txt'
    str_test_y = str_folder + 'test/y_test.txt'
    str_train_id = str_folder + 'train/subject_train.txt'
    str_test_id = str_folder + 'test/subject_test.txt'

    X_train = format_data_x(str_train_files)
    X_test = format_data_x(str_test_files)
    Y_train = format_data_y(str_train_y)
    Y_test = format_data_y(str_test_y)
    id_train = read_ids(str_train_id)
    id_test = read_ids(str_test_id)

    X_train, X_test = X_train.reshape((-1, 9, 1, 128)), X_test.reshape((-1, 9, 1, 128))

    X = np.concatenate((X_train, X_test), axis=0)
    Y = np.concatenate((Y_train, Y_test), axis=0)
    ID = np.concatenate((id_train, id_test), axis=0)

    XX, YY = [], []
    for i in np.unique(ID):
        idx = ID == i
        XX.append(X[idx])
        YY.append(Y[idx])

    return XX, YY


if __name__ == "__main__":
    generate_har(dir_path)
