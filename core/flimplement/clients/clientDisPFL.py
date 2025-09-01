import copy
import time

import math
import numpy as np
import torch
from torch import nn

from flimplement.clients.clientbase import Client
from torch.cuda.amp import autocast, GradScaler

class clientDisPFL(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.CELoss = nn.CrossEntropyLoss().to(self.device)
        self.local_sample_number = train_samples
        self.loss = nn.CrossEntropyLoss().to(self.device)
        self.args = args
        self.spa_ratio = args.dense_ratio
        self.mask_local = {}
        self.updates_matrix = copy.deepcopy(args.model)
        for param in self.updates_matrix.model.parameters():
            param.requires_grad = False
        self.w_per_global = copy.deepcopy(args.model)
        for param in self.w_per_global.model.parameters():
            param.requires_grad = False
        self.mask_shared = None
        self.model_last_round = copy.deepcopy(args.model)
        for param in self.model_last_round.model.parameters():
            param.requires_grad = False
        self.mask_shared_last_round = None
        self.nei_index = None

    def train(self, w, round):
        scaler = GradScaler()
        trainloader = self.load_train_data()
        num_comm_params = self.model.count_communication_params(w)
        self.model.set_model_params(w)
        self.model.set_masks(self.mask_local)
        self.model.train()
        self.model.set_id(self.id)

        max_local_epochs = self.local_epochs

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                with autocast():
                    logits, _ = self.model(x)
                    loss = self.CELoss(logits, y)
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    parameters=filter(lambda p: p.requires_grad, self.model.parameters()),
                    max_norm=10
                )
                scaler.step(self.optimizer)
                scaler.update()

        weights = self.model.get_model_params()
        self.model.set_model_params(weights)

        update = {}
        for name in weights:
            update[name] = weights[name] - w[name]

        print("-----------------------------------")
        gradient = None
        if not self.args.static:
            if not self.args.dis_gradient_check:
                with autocast():
                    gradient = self.model.screen_gradients(trainloader, self.device)
                masks, num_remove = self.fire_mask(self.mask_local, weights, round)
                masks = self.regrow_mask(masks, num_remove, gradient)

        sparse_flops_per_data = self.model.count_training_flops_per_sample()
        full_flops = self.model.count_full_flops_per_sample()
        print("training flops per data {}".format(sparse_flops_per_data))
        print("full flops for search {}".format(full_flops))
        training_flops = max_local_epochs * self.local_sample_number * sparse_flops_per_data + \
                         self.batch_size * full_flops

        # uplink params
        num_comm_params += self.model.count_communication_params(update)
        return masks, weights, update, training_flops, num_comm_params


    def fire_mask(self, masks, weights, round):
        drop_ratio = self.args.anneal_factor / 2 * (1 + np.cos((round * np.pi) / self.args.global_rounds))
        new_masks = copy.deepcopy(masks)

        num_remove = {}
        for name in masks:
            num_non_zeros = torch.sum(masks[name])
            num_remove[name] = math.ceil(drop_ratio * num_non_zeros)
            temp_weights = torch.where(masks[name] > 0, torch.abs(weights[name]),
                                       100000 * torch.ones_like(weights[name]))
            x, idx = torch.sort(temp_weights.view(-1).to(self.device))
            new_masks[name].view(-1)[idx[:num_remove[name]]] = 0
        return new_masks, num_remove

    # we only update the private components of client's mask
    def regrow_mask(self, masks, num_remove, gradient=None):
        new_masks = copy.deepcopy(masks)
        for name in masks:
            if gradient is None:
                raise ValueError("Gradient required but not provided")

            mask_device = masks[name].device
            grad_device = gradient[name].device if gradient is not None else None

            if gradient is not None and grad_device != mask_device:
                gradient[name] = gradient[name].to(mask_device)

            if not self.args.dis_gradient_check:
                temp = torch.where(masks[name] == 0, torch.abs(gradient[name]),
                                   -100000 * torch.ones_like(gradient[name]))
                sort_temp, idx = torch.sort(temp.view(-1), descending=True)
                new_masks[name].view(-1)[idx[:num_remove[name]]] = 1
            else:
                temp = torch.where(masks[name] == 0, torch.ones_like(masks[name]), torch.zeros_like(masks[name]))
                idx = torch.multinomial(temp.flatten(), num_remove[name], replacement=False)
                new_masks[name].view(-1)[idx] = 1

        return new_masks
