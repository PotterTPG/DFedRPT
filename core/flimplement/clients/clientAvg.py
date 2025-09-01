import copy
import torch
import numpy as np
import time
from torch.cuda.amp import autocast, GradScaler
from flimplement.clients.clientbase import Client
import torch.nn as nn


class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.loss = nn.CrossEntropyLoss().to(self.device)
        self.grad_norm = []

    def train(self):
        trainloader = self.load_train_data()
        self.model.train()
        scaler = GradScaler()

        start_time = time.time()
        max_local_epochs = self.local_epochs

        total_gradient_norm = 0.0  # 总梯度范数
        total_batches = 0  # 总批次数

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                with autocast():
                    output, _ = self.model(x)
                    loss = self.loss(output, y)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()

                batch_norm = 0.0
                scaler.unscale_(self.optimizer)
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        batch_norm += param_norm.item() ** 2
                batch_norm = batch_norm ** 0.5
                total_gradient_norm += batch_norm
                total_batches += 1

                scaler.step(self.optimizer)
                scaler.update()

        self.grad_norm.append(total_gradient_norm / total_batches)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
