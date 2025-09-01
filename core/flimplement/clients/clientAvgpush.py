import time
from torch.cuda.amp import autocast, GradScaler

from flimplement.clients.clientbase import Client
import torch.nn as nn

class clientPush(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.local_weight = 0.8
        self.loss = nn.CrossEntropyLoss().to(self.device)

    def train(self):
        trainloader = self.load_train_data()
        self.model.train()

        self.local_model_received = None
        scaler = GradScaler()
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
                    loss = self.loss(output, y)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        return self.model

    def receive_values(self, local_model):
        self.local_model_received = local_model

    def aggregate(self):

        other_sum = self.local_model_received
        if other_sum == None:
            self.model = self.model
        else:

            for self_sum_parm, other_sum_parm in zip(self.model.parameters(), other_sum.parameters()):
                self_sum_parm.data = (self_sum_parm.data * self.local_weight + other_sum_parm.data.clone() * (
                            1 - self.local_weight))
