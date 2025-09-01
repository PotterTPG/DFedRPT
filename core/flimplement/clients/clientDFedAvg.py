import time

from torch import nn
from torch.cuda.amp import autocast, GradScaler
from flimplement.clients.clientbase import Client


class clientDFedAvg(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.new_W = {key: value.clone() for key, value in self.model.state_dict().items()}
        self.old_W = {key: value.clone() for key, value in self.model.state_dict().items()}
        self.loss = nn.CrossEntropyLoss().to(self.device)

    def train(self):
        trainloader = self.load_train_data()
        self.model.train()


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

    def update_W(self):
        self.model.load_state_dict(self.new_W)
