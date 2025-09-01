import os
import sys

import pandas as pd
import numpy as np
import torch
import ujson
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms

from data.utils.dataset_utils import check, separate_data, save_file

random_seed = 21
torch.manual_seed(random_seed)
np.random.seed(random_seed)
dir_path = "ISIC2018/"
# 数据文件夹路径
num_clients = 10
num_classes = 7

# 相对路径定义
data_folders = {
    "train_input": os.path.join(dir_path, "rawdata", "ISIC2018_Task3_Training_Input"),
    "train_label_dir": os.path.join(dir_path, "rawdata", "ISIC2018_Task3_Training_GroundTruth"),
    "val_input": os.path.join(dir_path, "rawdata", "ISIC2018_Task3_Validation_Input"),
    "val_label_dir": os.path.join(dir_path, "rawdata", "ISIC2018_Task3_Validation_GroundTruth"),
    "test_input": os.path.join(dir_path, "rawdata", "ISIC2018_Task3_Test_Input"),
    "test_label_dir": os.path.join(dir_path, "rawdata", "ISIC2018_Task3_Test_GroundTruth"),
}


# 从目录中找到 CSV 文件
def find_csv_in_dir(directory):
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            return os.path.join(directory, file)
    raise FileNotFoundError(f"No CSV file found in directory: {directory}")



def load_isic_data(input_dir, label_dir):
    images = []
    labels = []
    class_mapping = {'MEL': 0, 'NV': 1, 'BCC': 2, 'AKIEC': 3, 'BKL': 4, 'DF': 5, 'VASC': 6}
    label_file = find_csv_in_dir(label_dir)
    label_data = pd.read_csv(label_file)

    label_dict = {
        row['image']: class_mapping[row[1:].idxmax()]
        for _, row in label_data.iterrows()
    }

    # 获取所有输入图像文件
    input_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    input_prefixes = [os.path.splitext(f)[0] for f in input_files]

    for input_file, prefix in zip(input_files, input_prefixes):
        if prefix in label_dict:
            image_path = os.path.join(input_dir, input_file)
            try:
                image = Image.open(image_path).convert("RGB")
                label = label_dict[prefix]
                images.append(image)
                labels.append(label)
            except Exception as e:
                print(f"Error loading {input_file}: {e}")
        else:
            print(f"No label found for image: {prefix}")
    return images, labels

# 数据预处理
def preprocess_data(images, labels, transform, batch_size):
    """
    Preprocess a dataset in batches, applying transformations to images.

    Args:
        images (list): List of image objects (e.g., PIL images).
        labels (list): List of labels corresponding to the images.
        transform (callable): A function or transform to apply to each image.
        batch_size (int): Number of images to process in each batch.

    Returns:
        list: A list of tensors, where each tensor contains a batch of processed images.
        list: A list of tensors, where each tensor contains a batch of labels.
    """
    processed_images = []
    processed_labels = []

    # Process data in batches
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]

        # Apply transformations to the batch
        batch_processed_images = [transform(img) for img in batch_images]
        batch_processed_labels = batch_labels  # Labels remain unchanged

        # Stack tensors for the current batch
        processed_images.append(torch.stack(batch_processed_images))
        processed_labels.append(torch.tensor(batch_processed_labels))

    return processed_images, processed_labels


def save_partitioned_data(train_images, train_labels, test_images, test_labels, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    public_ratio = 0.1
    train_indices, public_indices = train_test_split(
        range(len(train_images)), test_size=public_ratio, stratify=train_labels.numpy(), random_state=21
    )

    public_images = train_images[public_indices]
    public_labels = train_labels[public_indices]
    train_images = train_images[train_indices]
    train_labels = train_labels[train_indices]
    public_data_file = os.path.join(output_dir, "public_dataset.npz")
    np.savez_compressed(public_data_file, x=public_images.numpy(), y=public_labels.numpy())

    num_samples_per_client = len(train_images) // num_clients
    for client_id in range(num_clients):
        client_images = train_images[client_id * num_samples_per_client:(client_id + 1) * num_samples_per_client]
        client_labels = train_labels[client_id * num_samples_per_client:(client_id + 1) * num_samples_per_client]

        client_data_dir = os.path.join(output_dir, f"client_{client_id}")
        if not os.path.exists(client_data_dir):
            os.makedirs(client_data_dir)

        client_data_file = os.path.join(client_data_dir, "client_data.npz")
        np.savez_compressed(client_data_file, x=client_images.numpy(), y=client_labels.numpy())

    test_data_file = os.path.join(output_dir, "global_test.npz")
    np.savez_compressed(test_data_file, x=test_images.numpy(), y=test_labels.numpy())


def generate_isic_task3(dir_path, num_clients, num_classes, niid, balance, partition):
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

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_images, train_labels = load_isic_data(data_folders["train_input"], data_folders["train_label_dir"])
    val_images, val_labels = load_isic_data(data_folders["val_input"], data_folders["val_label_dir"])
    test_images, test_labels = load_isic_data(data_folders["test_input"], data_folders["test_label_dir"])

    train_images, train_labels = preprocess_data(train_images, train_labels, transform,batch_size=128)
    val_images, val_labels = preprocess_data(val_images, val_labels, transform,batch_size=128)
    test_images, test_labels = preprocess_data(test_images, test_labels, transform,batch_size=128)
    train_images = torch.cat(train_images, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    val_images = torch.cat(val_images, dim=0)
    val_labels = torch.cat(val_labels, dim=0)
    test_images=torch.cat(test_images, dim=0)
    test_labels = torch.cat(test_labels, dim=0)
    train_images = torch.cat([train_images, val_images], dim=0)
    train_labels = torch.cat([train_labels, val_labels], dim=0)
    dataset_image_train = []
    dataset_label_train = []
    dataset_image_test = []
    dataset_label_test = []
    dataset_image_train.extend(train_images.cpu().detach().numpy())
    dataset_image_test.extend(test_images.cpu().detach().numpy())
    dataset_label_train.extend(train_labels.cpu().detach().numpy())
    dataset_label_test.extend(test_labels.cpu().detach().numpy())
    dataset_image_train = np.array(dataset_image_train)
    dataset_label_train = np.array(dataset_label_train)
    dataset_image_test = np.array(dataset_image_test)
    dataset_label_test = np.array(dataset_label_test)

    split_ratio = 0.1
    num_train_samples = len(dataset_image_train)
    class_indices = {i: np.where(dataset_label_train == i)[0] for i in range(num_classes)}  # CIFAR-10 有 10 个类别

    split_indices = []
    for i in range(num_classes):
        # 为每个类别设置固定但不同的随机种子确保可重现性
        np.random.seed(21 + i)
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
    X, y, statistic, noise_ratios_dict = separate_data((dataset_image_train_remaining, dataset_label_train_remaining),
                                                       num_clients, num_classes,
                                                       niid, balance, partition, class_per_client=5, add_noise=True,
                                                       add_noise_type='symmetric')
    if noise_ratios_dict is not None:
        with open(noisy_path, 'w') as f:
            ujson.dump(noise_ratios_dict, f, indent=4)
    train_data = []
    test_data = []
    for i in range(len(y)):
        train_data.append({'x': X[i], 'y': y[i]})

    test_data.append({'x': dataset_image_test, 'y': dataset_label_test})

    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, statistic, niid,
              balance, partition)



if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_isic_task3(dir_path, num_clients, num_classes, niid, balance, partition)
