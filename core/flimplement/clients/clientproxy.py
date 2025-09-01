import torch
import torch.nn.functional as F
import time
import copy
from torch.cuda.amp import autocast, GradScaler

from torch import nn

from flimplement.clients.clientbase import Client


class clientProxy(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break
        self.loss = nn.CrossEntropyLoss()
        self.privacy = False
        self.proxy_model = copy.deepcopy(args.proxy_model)
        self.proxy_optimizer = torch.optim.SGD(self.proxy_model.parameters(), lr=self.learning_rate, momentum=0.9,
                                               weight_decay=0.0001)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.sample_size = None
        self.loss_weight = 0.5
        self.dp_sigma = 1.0
        self.local_weight = 0.8

    def train(self):
        trainloader = self.load_train_data()
        self.model.train()
        self.proxy_model.train()
        self.sample_size = len(trainloader.dataset)

        scaler = GradScaler()
        proxy_scaler = GradScaler()

        start_time = time.time()
        max_local_epochs = self.local_epochs

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                with autocast():
                    output, _ = self.model(x)
                    proxy_output = self.proxy_model(x)
                    output_softmax = F.softmax(output.detach(), dim=1)
                    proxy_output_softmax = F.softmax(proxy_output.detach(), dim=1)

                    ce_loss = self.loss(output, y)
                    kl_loss = self.kl_loss(torch.log(output_softmax), proxy_output_softmax)
                    proxy_ce_loss = self.loss(proxy_output, y)
                    proxy_kl_loss = self.kl_loss(torch.log(proxy_output_softmax), output_softmax)

                    loss = ce_loss * self.loss_weight + kl_loss * (1 - self.loss_weight)
                    proxy_loss = proxy_ce_loss * self.loss_weight + proxy_kl_loss * (1 - self.loss_weight)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                self.proxy_optimizer.zero_grad()
                proxy_scaler.scale(proxy_loss).backward()
                proxy_scaler.step(self.proxy_optimizer)
                proxy_scaler.update()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        return self.proxy_model

    def receive_model(self, proxy_model):
        self.proxy_model_received = proxy_model

    def aggregate(self):

        other_sum = self.proxy_model_received
        if other_sum == None:
            self.proxy_model = self.proxy_model
        else:
            for self_sum_parm, other_sum_parm in zip(self.proxy_model.parameters(), other_sum.parameters()):
                self_sum_parm.data = (self_sum_parm.data * self.local_weight + other_sum_parm.data.clone() * (
                        1 - self.local_weight))
