import copy
import logging
import time
from itertools import chain

import numpy as np
import torch
from numpy import mean
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from torch.cuda.amp import GradScaler, autocast
from torch.nn import KLDivLoss
from torch.utils.data import DataLoader

from flimplement.clients.clientbase import Client
from core.utils.loss import LogitAdjust, SCELoss, GeneralizedCrossEntropy, MeanAbsoluteError, co_teaching_loss, \
    DynamicCrossEntropyLoss
from core.utils.data_utils import read_client_public_data, read_data
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def update_reduce_step(cur_step, num_gradual, tau=0.5):
    return 1.0 - tau * min(cur_step / num_gradual, 1)

class ClientTR(Client):
    def __init__(self, args, id, train_samples, test_samples, count_by_class, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.count_by_class = count_by_class
        max_index = max(count_by_class.keys())
        self.warm_model = copy.deepcopy(args.model)
        self.model1 = copy.deepcopy(args.model)
        self.model2 = copy.deepcopy(args.model)
        self.model1.to(self.device)
        self.model2.to(self.device)
        self.warm_model.to(self.device)
        self.warm_optimizer = torch.optim.SGD(self.warm_model.parameters(), lr=self.learning_rate,
                                              momentum=self.momentum,
                                              weight_decay=self.weight_decay)
        self.optimizer = torch.optim.SGD(chain(self.model1.parameters(), self.model2.parameters()),
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

    def train_vanilla(self):
        trainloader = self.load_train_data()  # 加载训练数据
        self.model1.train()
        start_time = time.time()
        max_local_epochs = 5
        scaler = GradScaler()
        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device, non_blocking=True)
                else:
                    x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                with autocast():
                    if "Cifar100" not in self.dataset:
                        logits, _ = self.model1.forward(x)
                    else:
                        logits = self.model1.forward(x)
                    loss = self.loss(logits, y)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    parameters=filter(lambda p: p.requires_grad, self.model1.parameters()),
                    max_norm=10
                )

                scaler.step(self.optimizer)
                scaler.update()
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def train_noise(self):
        trainloader = self.load_train_data()  # 加载训练数据
        start_time = time.time()
        self.model1.train()
        self.model2.train()
        max_local_epochs = self.local_epochs
        scaler = GradScaler()
        for epoch in range(max_local_epochs):
            tau = 0.85
            rt = update_reduce_step(cur_step=epoch, num_gradual=5, tau=tau)
            for i, (x, y) in enumerate(trainloader):
                if isinstance(x, list):
                    x[0] = x[0].to(self.device, non_blocking=True)
                else:
                    x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                with autocast():
                    if "Cifar100" not in self.dataset:
                        out1, _ = self.model1(x)
                        out2, _ = self.model2(x)
                    else:
                        out1 = self.model1(x)
                        out2 = self.model2(x)
                    loss1 = self.loss(out1, y, reduction='none')
                    loss2 = self.loss(out2, y, reduction='none')
                    model1_loss, model2_loss = co_teaching_loss(model1_loss=loss1, model2_loss=loss2, rt=rt)

                self.optimizer.zero_grad()
                scaler.scale(model1_loss).backward(retain_graph=True)  # 为model1计算梯度
                scaler.scale(model2_loss).backward()  # 为model2计算梯度
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    parameters=filter(lambda p: p.requires_grad, self.model1.parameters()),
                    max_norm=10
                )
                torch.nn.utils.clip_grad_norm_(
                    parameters=filter(lambda p: p.requires_grad, self.model2.parameters()),
                    max_norm=10
                )
                scaler.step(self.optimizer)
                scaler.update()
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def train_warm_up(self):
        trainloader = self.load_train_data()  # 加载训练数据
        self.warm_model.train()
        start_time = time.time()
        max_local_epochs = 5
        scaler = GradScaler()
        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(
                    trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device, non_blocking=True)
                else:
                    x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                with autocast():
                    if "Cifar100" not in self.dataset:
                        logits, _ = self.warm_model.forward(x)
                    else:
                        logits = self.warm_model.forward(x)
                    loss = self.check_loss(logits, y)

                self.warm_optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(self.warm_optimizer)

                torch.nn.utils.clip_grad_norm_(
                    parameters=filter(lambda p: p.requires_grad, self.warm_model.parameters()),
                    max_norm=10
                )

                scaler.step(self.warm_optimizer)
                scaler.update()
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time



    def send_full_model_and_aggregate(self, receiver_client):
        state_dict = self.model1.state_dict()

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
        state_dict = self.model1.state_dict()
        for key in received_params.keys():
            state_dict[key] = (
                                      state_dict[key] * local_sample_count +
                                      received_params[key] * received_sample_count
                              ) / total_samples

        self.model1.load_state_dict(state_dict)
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
                if type(x) == type([]):
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

    def load_public_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        public_data = read_client_public_data(self.dataset)
        return DataLoader(
            public_data,
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
        )

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        self.model1.eval()

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
                    output, _ = self.model1(x)
                else:
                    output = self.model1(x)
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

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob,
                                    average='micro')
        y_pred = np.argmax(y_prob, axis=1)
        y_true = np.argmax(y_true, axis=1)
        precision = metrics.precision_score(y_true, y_pred, average='macro')
        recall = metrics.recall_score(y_true, y_pred, average='macro')
        f1 = metrics.f1_score(y_true, y_pred, average='macro')
        return test_acc, test_num, auc, precision, recall, f1

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model1.eval()
        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device, non_blocking=True)
                else:
                    x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                if "Cifar100" not in self.dataset:
                    output, _ = self.model1(x)
                else:
                    output = self.model1(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
        return losses, train_num
