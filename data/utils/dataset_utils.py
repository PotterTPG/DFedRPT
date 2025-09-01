import random

import numpy as np
import os
import ujson

least_samples = 1
alpha = 0.5
batch_size = 60
np.random.seed(21)
noise_client_ratio = 0.5


def check(config_path, train_path, test_path, num_clients, num_classes, niid=False,
          balance=True, partition=None):
    # check existing dataset
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = ujson.load(f)
        if config['num_clients'] == num_clients and \
                config['num_classes'] == num_classes and \
                config['non_iid'] == niid and \
                config['balance'] == balance and \
                config['partition'] == partition and \
                config['alpha'] == alpha and \
                config['batch_size'] == batch_size:
            print("\nDataset already generated.\n")
            return True

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return False


def noisify_label(true_label, num_classes=10, noise_type="pairflip"):
    if noise_type == "symmetric":
        label_lst = list(range(num_classes))
        label_lst.remove(true_label)
        return random.sample(label_lst, k=1)[0]

    elif noise_type == "pairflip":
        return (true_label - 1) % num_classes


def separate_data(data, num_clients, num_classes, niid=False, balance=False, partition=None, class_per_client=None,
                  add_noise=False, add_noise_type='pairflip'):
    noisy_sample_indices_dict = {f"client_{i}": [] for i in range(num_clients)}  # 新增：用于保存每个客户端中被加噪的索引
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    dataset_content, dataset_label = data
    print(add_noise_type)
    dataidx_map = {}
    # niid代表非独立同分布
    if not niid:
        partition = 'pat'
        class_per_client = num_classes

    if partition == 'pat':
        idxs = np.array(range(len(dataset_label)))
        idx_for_each_class = []
        for i in range(num_classes):
            idx_for_each_class.append(idxs[dataset_label == i])

        class_num_per_client = [class_per_client for _ in range(num_clients)]
        for i in range(num_classes):
            selected_clients = []
            for client in range(num_clients):
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)
            selected_clients = selected_clients[:int(np.ceil((num_clients / num_classes) * class_per_client))]

            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
            if balance:
                num_samples = [int(num_per) for _ in range(num_selected_clients - 1)]
            else:
                num_samples = np.random.randint(max(num_per / 10, least_samples / num_classes), num_per,
                                                num_selected_clients - 1).tolist()
            num_samples.append(num_all_samples - sum(num_samples))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map.keys():
                    dataidx_map[client] = idx_for_each_class[i][idx:idx + num_sample]
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx + num_sample],
                                                    axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1

    elif partition == "dir":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = num_classes
        N = len(dataset_label)

        try_cnt = 1
        while min_size < least_samples:
            if try_cnt > 1:
                print(
                    f'Client data size does not meet the minimum requirement {least_samples}. Try allocating again for the {try_cnt}-th time.')

            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            try_cnt += 1

        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]
    else:
        raise NotImplementedError
    noise_ratios_dict = None
    if add_noise == True:
        noise_ratios = np.random.uniform(0.3, 0.7, num_clients)
        num_noisy_clients = int(num_clients * noise_client_ratio)
        noisy_client_indices = np.random.choice(num_clients, num_noisy_clients, replace=False)
        noise_ratios_dict = {f"client_{i}": (float(noise_ratios[i]) if i in noisy_client_indices else 0.0)
                             for i in range(num_clients)}
    # assign data
    for client in range(num_clients):
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]
        if add_noise and client in noisy_client_indices:
            per_client_noise_ratio = noise_ratios[client]
            num_noisy_labels = int(len(y[client]) * per_client_noise_ratio)
            noisy_indices = np.random.choice(len(y[client]), size=num_noisy_labels, replace=False)
            noisy_label_pairs = []
            for i in range(len(noisy_indices)):
                noise_need_idx = noisy_indices[i]
                true_label = y[client][noise_need_idx]
                noisy_label = noisify_label(true_label, num_classes=num_classes, noise_type=add_noise_type)
                noisy_label_pairs.append((noise_need_idx, true_label, noisy_label))
            noisy_sample_indices_dict[f"client_{client}"] = [
                {"index": int(idx), "true_label": int(t), "noisy_label": int(n)}
                for (idx, t, n) in noisy_label_pairs
            ]
            for (idx, _, noisy_label) in noisy_label_pairs:
                y[client][idx] = noisy_label
        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client] == i))))

    del data
    # gc.collect()

    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    return X, y, statistic, noise_ratios_dict,noisy_sample_indices_dict


def split_dict_values_shuffled(data_dict, num_parts):
    # 创建一个新的字典来存储分割后的结果
    split_dict = {}

    for key, value in data_dict.items():
        # 打乱原始列表的顺序
        random.shuffle(value)

        # 计算每个分段的大小
        part_size = len(value) // num_parts
        remainder = len(value) % num_parts  # 处理余数

        # 分割列表
        parts = []
        start = 0
        for i in range(num_parts):
            end = start + part_size + (1 if i < remainder else 0)
            parts.append(value[start:end])
            start = end

        # 将分割后的值保存到新的字典中
        split_dict[key] = parts

    return split_dict


def separate_data_CifarN(data, num_clients, num_classes, niid=False, balance=False, partition=None,
                         class_per_client=None,
                         add_noise=False, add_noise_type='symmetric'):
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    dataset_content, dataset_label_real, dataset_label_clean = data

    class_indices_real = {i: np.where(dataset_label_real == i)[0] for i in range(num_classes)}
    noise_indices = {i: np.where((dataset_label_real == i) & (dataset_label_real != dataset_label_clean))[0] for i in
                     range(num_classes)}
    clean_indices = {i: np.where((dataset_label_real == i) & (dataset_label_real == dataset_label_clean))[0] for i in
                     range(num_classes)}
    client_per_class_num = {i: (int(len(class_indices_real[i]) // num_clients)) for i in range(num_classes)}

    noisy_client_indices = np.random.choice(num_clients, int(num_clients * noise_client_ratio),
                                            replace=False) if add_noise else []
    dataidx_map = {i: [] for i in range(num_clients)}
    noise_asign_idx = split_dict_values_shuffled(noise_indices, len(noisy_client_indices))
    for i in range(num_classes):
        for noise_id, client_idx in enumerate(noisy_client_indices):
            X[client_idx].extend(dataset_content[noise_asign_idx[i][noise_id]])
            y[client_idx].extend(dataset_label_real[noise_asign_idx[i][noise_id]])
            dataidx_map[client_idx].extend(noise_asign_idx[i][noise_id])
    clean_client_indices = list(set(range(num_clients)) - set(noisy_client_indices))
    client_indices_clean = {i: {cls: [] for cls in clean_indices} for i in clean_client_indices}
    for class_idx, indices in clean_indices.items():
        num_labels = len(indices)
        remaining_labels = num_labels
        for client_idx in clean_client_indices:
            labels_to_assign = client_per_class_num
            client_indices_clean[client_idx][class_idx].extend(indices[:labels_to_assign[class_idx]])
            indices = indices[labels_to_assign[class_idx]:]
            clean_indices[class_idx] = clean_indices[class_idx][labels_to_assign[class_idx]:]
            remaining_labels -= labels_to_assign[class_idx]

    for i in range(num_classes):
        for client_idx in clean_client_indices:
            X[client_idx].extend(dataset_content[client_indices_clean[client_idx][i]])
            y[client_idx].extend(dataset_label_real[client_indices_clean[client_idx][i]])
            dataidx_map[client_idx].extend(client_indices_clean[client_idx][i])
    clean_asign_idx_to_noise_client = split_dict_values_shuffled(clean_indices, len(noisy_client_indices))
    for i in range(num_classes):
        for noise_id, client_idx in enumerate(noisy_client_indices):
            X[client_idx].extend(dataset_content[clean_asign_idx_to_noise_client[i][noise_id]])
            y[client_idx].extend(dataset_label_real[clean_asign_idx_to_noise_client[i][noise_id]])
            dataidx_map[client_idx].extend(clean_asign_idx_to_noise_client[i][noise_id])
    for client_idx in range(num_clients):
        dataidx_map[client_idx].sort()
        for i in np.unique(y[client_idx]):
            statistic[client_idx].append((int(i), int(sum(y[client_idx] == i))))
    # 计算噪声客户端的噪声比例
    noise_ratios = [0 for _ in range(num_clients)]
    for noise_id, client_idx in enumerate(noisy_client_indices):
        noise_num = 0
        for class_idx in range(num_classes):
            noise_num += len(noise_asign_idx[class_idx][noise_id])
        noise_ratios[client_idx] = noise_num / len(y[client_idx])

    noise_ratios_dict = {f"client_{i}": (float(noise_ratios[i]) if i in noisy_client_indices else 0.0)
                         for i in range(num_clients)}
    return X, y, statistic, noise_ratios_dict


def save_file(config_path, train_path, test_path, train_data, test_data, num_clients,
              num_classes, statistic, niid=False, balance=True, partition=None):
    config = {
        'num_clients': num_clients,
        'num_classes': num_classes,
        'non_iid': niid,
        'balance': balance,
        'partition': partition,
        'Size of samples for labels in clients': statistic,
        'alpha': alpha,
        'batch_size': batch_size,
    }

    # gc.collect()
    print("Saving to disk.\n")

    for idx, train_dict in enumerate(train_data):
        with open(train_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_dict)
    for idx, test_dict in enumerate(test_data):
        with open(test_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=test_dict)
    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    print("Finish generating dataset.\n")
