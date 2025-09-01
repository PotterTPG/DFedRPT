import numpy as np
import os
import sys
import random
import ujson
import torch
import torchvision
import torchvision.transforms as transforms
from data.utils.dataset_utils import *

random.seed(25)
np.random.seed(25)
num_clients = 10
num_classes = 100
dir_path = "Cifar100/"


# Allocate data to users
def generate_cifar100(dir_path, num_clients, num_classes, niid, balance, partition, samples_per_class=200):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "global_test/"
    noisy_path = dir_path + "noisy.json"
    public_path = dir_path + "public/"

    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
        return
    if not os.path.exists(public_path):
        os.makedirs(public_path)
    # Get Cifar100 data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    trainset = torchvision.datasets.CIFAR100(
        root=dir_path+"rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(
        root=dir_path+"rawdata", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image_train = []
    dataset_label_train = []
    dataset_image_test = []
    dataset_label_test = []
    dataset_image_train.extend(trainset.data.cpu().detach().numpy())
    dataset_image_test.extend(testset.data.cpu().detach().numpy())
    dataset_label_train.extend(trainset.targets.cpu().detach().numpy())
    dataset_label_test.extend(testset.targets.cpu().detach().numpy())
    dataset_image_train = np.array(dataset_image_train)
    dataset_label_train = np.array(dataset_label_train)
    dataset_image_test = np.array(dataset_image_test)
    dataset_label_test = np.array(dataset_label_test)

    label_sample_counts = {}
    split_ratio = 0.1
    num_train_samples = len(dataset_image_train)

    class_indices = {i: np.where(dataset_label_train == i)[0] for i in range(num_classes)}  # CIFAR-10 有 10 个类别

    split_indices = []
    for i in range(num_classes):
        # 为每个类别设置固定但不同的随机种子确保可重现性
        np.random.seed(25 + i)
        class_idx = np.random.permutation(class_indices[i])
        num_class_samples = len(class_idx)
        num_class_split_samples = int(num_class_samples * split_ratio)
        split_indices.extend(class_idx[:num_class_split_samples])

    remaining_indices = list(set(range(num_train_samples)) - set(split_indices))

    dataset_image_train_split = dataset_image_train[split_indices]
    dataset_label_train_split = dataset_label_train[split_indices]
    public_data_file = os.path.join(public_path, 'public_dataset.npz')
    np.savez_compressed(public_data_file, data={'x': dataset_image_train_split, 'y': dataset_label_train_split})

    dataset_image_train_remaining = dataset_image_train[remaining_indices]
    dataset_label_train_remaining = dataset_label_train[remaining_indices]

    X, y, statistic,noise_ratios_dict = separate_data((dataset_image_train_remaining, dataset_label_train_remaining), num_clients, num_classes,
                                    niid, balance, partition, class_per_client=5,add_noise=True,add_noise_type='symmetric')
    if noise_ratios_dict is not None:
        with open(noisy_path, 'w') as f:
            ujson.dump(noise_ratios_dict, f, indent=4)
    train_data = []
    test_data = []
    for i in range(len(y)):
        train_data.append({'x': X[i], 'y': y[i]})

    test_data.append({'x': dataset_image_test, 'y': dataset_label_test})

    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,statistic, niid, balance, partition)


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_cifar100(dir_path, num_clients, num_classes, niid, balance, partition)
