import torch
import numpy as np
import time
import copy
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from torch import nn

from flimplement.clients.clientbase import Client
from core.utils.loss import PerturbedGradientDescent
from torch.cuda.amp import autocast, GradScaler
class clientDitto(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.mu = args.mu
        self.plocal_epochs = args.plocal_epochs
        self.loss = nn.CrossEntropyLoss()
        self.model_per = copy.deepcopy(self.model)
        self.optimizer_per = PerturbedGradientDescent(
            self.model_per.parameters(), lr=self.learning_rate, mu=self.mu)
        self.learning_rate_scheduler_per = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_per,
            gamma=args.learning_rate_decay_gamma
        )

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()

        self.model.train()
        scaler = GradScaler()

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
                    loss = self.loss(output, y)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
            self.learning_rate_scheduler_per.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def ptrain(self):
        trainloader = self.load_train_data()
        start_time = time.time()

        self.model_per.train()

        scaler = GradScaler()

        max_local_epochs = self.plocal_epochs

        for epoch in range(max_local_epochs):
            for x, y in trainloader:
                if isinstance(x, (list, tuple)):
                    x = [x_item.to(self.device) for x_item in x]
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                with autocast():
                    output, _ = self.model_per(x)
                    loss = self.loss(output, y)

                self.optimizer_per.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer_per)
                self.optimizer_per.step(self.model.parameters(), self.device)
                scaler.update()

            self.train_time_cost['total_cost'] += time.time() - start_time

    def test_metrics_personalized(self):
        testloaderfull = self.load_test_data()
        self.model_per.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output, _ = self.model_per(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(F.softmax(output).detach().cpu().numpy())
                y_true.append(label_binarize(y.detach().cpu().numpy(), classes=np.arange(self.num_classes)))


        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        return test_acc, test_num, auc

    def train_metrics_personalized(self):
        trainloader = self.load_train_data()
        self.model_per.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output, _ = self.model_per(x)
                loss = self.loss(output, y)

                gm = torch.cat([p.data.view(-1) for p in self.model.parameters()], dim=0)
                pm = torch.cat([p.data.view(-1) for p in self.model_per.parameters()], dim=0)
                loss += 0.5 * self.mu * torch.norm(gm - pm, p=2)

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num
