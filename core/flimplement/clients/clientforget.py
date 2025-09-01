import copy
import logging
import time
import numpy as np
import torch
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from torch.cuda.amp import GradScaler, autocast
from torch.nn import KLDivLoss
from torch.utils.data import DataLoader, Subset, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

from flimplement.clients.clientbase import Client
from core.utils.loss import LogitAdjust, SCELoss, GeneralizedCrossEntropy, MeanAbsoluteError, \
    DynamicCrossEntropyLoss
from core.utils.data_utils import read_client_public_data, read_data

logger = logging.getLogger(__name__)


class ClientForget(Client):
    def __init__(self, args, id, train_samples, test_samples, count_by_class, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.count_by_class = count_by_class
        max_index = max(count_by_class.keys())
        self.warm_model = copy.deepcopy(args.model)
        self.model = copy.deepcopy(args.model)
        self.model.to(self.device)
        self.warm_model.to(self.device)
        self.warm_optimizer = torch.optim.SGD(self.warm_model.parameters(), lr=self.learning_rate,
                                              momentum=self.momentum,
                                              weight_decay=self.weight_decay)
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=self.learning_rate, momentum=self.momentum,
                                         weight_decay=self.weight_decay)
        self.class_num_list = [0] * (max_index + 1)
        for key, value in count_by_class.items():
            self.class_num_list[key] = value
        if args.loss == "SCE":
            self.loss = SCELoss(alpha=0.1, beta=1.0, num_classes=self.num_classes)
        elif args.loss == "CE":
            self.loss = DynamicCrossEntropyLoss()
        elif args.loss == "LA":
            self.loss = LogitAdjust(cls_num_list=self.class_num_list)
        elif args.loss == "GCE":
            self.loss = GeneralizedCrossEntropy(num_classes=self.num_classes)
        elif args.loss == "MAE":
            self.loss = MeanAbsoluteError(num_classes=self.num_classes)
        self.check_loss = SCELoss(alpha=0.1, beta=1.0, num_classes=self.num_classes)
        self.kl_loss = KLDivLoss(reduction='batchmean')
        # 新增：遗忘数据索引
        self.forget_indices = []

    def set_forget_indices(self, forget_indices):
        """设置需要遗忘的样本索引（例如10%随机样本）"""
        self.forget_indices = forget_indices
        logger.info(f"Client {self.id} set forget indices: {len(forget_indices)} samples.")

    def load_retain_data(self, batch_size=None):
        """加载保留数据（排除遗忘数据）"""
        if batch_size is None:
            batch_size = self.batch_size
        full_dataset = read_data(self.dataset, self.id, is_train=True)
        retain_indices = [i for i in range(len(full_dataset['y'])) if i not in self.forget_indices]
        x = torch.tensor(full_dataset['x']) if not isinstance(full_dataset['x'], torch.Tensor) else full_dataset['x']
        y = torch.tensor(full_dataset['y']) if not isinstance(full_dataset['y'], torch.Tensor) else full_dataset['y']
        retain_dataset = Subset(TensorDataset(x, y), retain_indices)
        return DataLoader(
            retain_dataset,
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
        )

    def load_forget_data(self, batch_size=None):
        """加载遗忘数据（仅用于评估）"""
        if batch_size is None:
            batch_size = self.batch_size
        full_dataset = read_data(self.dataset, self.id, is_train=True)
        forget_dataset = Subset(full_dataset, self.forget_indices)
        return DataLoader(
            forget_dataset,
            batch_size=batch_size,
            drop_last=False,
            shuffle=False,
        )

    def negate_weights(self):
        """执行权重否定（NoT算法第一步）"""
        state_dict = self.model.state_dict()
        target_layer = 'c1.weight'  # 假设ResNet-18，可根据args.model调整
        if target_layer in state_dict:
            state_dict[target_layer] = -state_dict[target_layer]
            self.model.load_state_dict(state_dict)
            logger.info(f"Client {self.id} negated weights for layer {target_layer}.")
        else:
            logger.error(f"Layer {target_layer} not found in model.")

    def unlearn(self):
        """执行NoT算法的实例级遗忘"""
        # 步骤1：权重否定
        self.negate_weights()

        # 步骤2：微调
        retainloader = self.load_retain_data()
        self.model.train()
        start_time = time.time()
        max_local_epochs = 50  # 参考文章第23页
        scaler = GradScaler()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

        for epoch in range(max_local_epochs):
            for x, y in retainloader:
                if isinstance(x, list):
                    x[0] = x[0].to(self.device, non_blocking=True)
                else:
                    x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                with autocast():
                    if "Cifar100" not in self.dataset:
                        logits, _ = self.model(x)
                    else:
                        logits = self.model(x)
                    loss = self.loss(logits, y)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        logger.info(f"Client {self.id} completed unlearning with {max_local_epochs} epochs.")

    def train_vanilla(self):
        trainloader = self.load_train_data()
        self.model.train()
        start_time = time.time()
        max_local_epochs = 5
        scaler = GradScaler()
        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if isinstance(x, list):
                    x[0] = x[0].to(self.device, non_blocking=True)
                else:
                    x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                with autocast():
                    if "Cifar100" not in self.dataset:
                        logits, _ = self.model(x)
                    else:
                        logits = self.model(x)
                    loss = self.loss(logits, y)
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                scaler.step(self.optimizer)
                scaler.update()
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def train_warm_up(self):
        trainloader = self.load_train_data()
        self.warm_model.train()
        start_time = time.time()
        max_local_epochs = 5
        scaler = GradScaler()
        for epoch in range(max_local_epochs):
            for x, y in trainloader:
                if isinstance(x, list):
                    x[0] = x[0].to(self.device, non_blocking=True)
                else:
                    x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                with autocast():
                    if "Cifar100" not in self.dataset:
                        logits, _ = self.warm_model(x)
                    else:
                        logits = self.warm_model(x)
                    loss = self.check_loss(logits, y)
                self.warm_optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(self.warm_optimizer)
                torch.nn.utils.clip_grad_norm_(parameters=self.warm_model.parameters(), max_norm=10)
                scaler.step(self.warm_optimizer)
                scaler.update()
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def send_full_model_and_aggregate(self, receiver_client):
        state_dict = self.model.state_dict()
        sample_count = self.train_samples
        message = {
            'model_params': state_dict,
            'sample_count': sample_count,
            'sender_id': self.id,
        }
        receiver_client.receive_full_model(message)

    def receive_full_model(self, message):
        sender_id = message['sender_id']
        logger.info(f"Client {self.id} received full model from Client {sender_id}.")
        self.aggregate_full_model(message)

    def aggregate_full_model(self, message):
        received_params = message['model_params']
        received_sample_count = message['sample_count']
        local_sample_count = self.train_samples
        total_samples = local_sample_count + received_sample_count
        state_dict = self.model.state_dict()
        for key in received_params.keys():
            state_dict[key] = (
                                      state_dict[key] * local_sample_count +
                                      received_params[key] * received_sample_count
                              ) / total_samples
        self.model.load_state_dict(state_dict)
        logger.info(f"Client {self.id} aggregated full model parameters.")


    def test_metrics_warm_up(self):
        total_loss = 0.0
        total_samples = 0
        class_loss = np.zeros(self.num_classes, dtype=np.float32)
        class_sample_count = np.zeros(self.num_classes, dtype=np.int32)
        testloaderfull = self.load_public_data()
        self.warm_model.eval()
        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device, non_blocking=True)
                else:
                    x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                if "Cifar100" not in self.dataset:
                    output, _ = self.warm_model(x)
                else:
                    output = self.warm_model(x)
                loss = self.check_loss(output, y)
                total_loss += loss.item() * y.size(0)
                total_samples += y.size(0)
                with torch.no_grad():
                    for c in range(self.num_classes):
                        mask = (y == c)
                        if mask.sum() > 0:
                            class_loss[c] += loss.item() * mask.sum().item()
                            class_sample_count[c] += mask.sum().item()
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(
                    output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(
                    nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        overall_average_loss = total_loss / total_samples if total_samples > 0 else 0.0
        average_class_loss = class_loss / np.maximum(class_sample_count, 1)
        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob,
                                    average='micro')
        y_pred = np.argmax(y_prob, axis=1)
        y_true = np.argmax(y_true, axis=1)
        precision = metrics.precision_score(y_true, y_pred, average='macro')
        recall = metrics.recall_score(y_true, y_pred, average='macro')
        f1 = metrics.f1_score(y_true, y_pred, average='macro')
        return test_acc, test_num, auc, average_class_loss, overall_average_loss, precision, recall, f1

    def train_warm_metrics(self):
        trainloader = self.load_train_data()
        self.warm_model.eval()
        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if isinstance(x, list):
                    x[0] = x[0].to(self.device, non_blocking=True)
                else:
                    x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                if "Cifar100" not in self.dataset:
                    output, _ = self.warm_model(x)
                else:
                    output = self.warm_model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
        return losses, train_num

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.eval()
        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if isinstance(x, list):
                    x[0] = x[0].to(self.device, non_blocking=True)
                else:
                    x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                if "Cifar100" not in self.dataset:
                    output, _ = self.model(x)
                else:
                    output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
        return losses, train_num

    def load_public_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        public_data = read_client_public_data(self.dataset)
        return DataLoader(
            public_data,
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
        )