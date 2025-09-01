import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:1")
from torch.optim import Optimizer

class LogitAdjust(nn.Module):
    def __init__(self, cls_num_list, tau=1, weight=None):
        super(LogitAdjust, self).__init__()
        cls_num_list = torch.Tensor(cls_num_list).to(device)
        cls_num_list = torch.where(cls_num_list == 0, torch.tensor(1e-8, device=cls_num_list.device), cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        m_list = torch.where(torch.isinf(m_list), torch.full_like(m_list, -1e6), m_list)
        self.m_list = m_list.view(1, -1)
        self.weight = weight

    def forward(self, x, target, reduction='mean'):
        x_m = x + self.m_list
        x_m = torch.clamp(x_m, min=-1e4, max=1e4)
        loss = F.cross_entropy(x_m, target, weight=self.weight, reduction='none')

        # 根据 reduction 参数返回结果
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'none':
            return loss
        else:
            raise ValueError(f"Invalid reduction mode: {reduction}. Choose 'mean', 'sum', or 'none'.")


class LA_KD(nn.Module):
    def __init__(self, cls_num_list, tau=1):
        super(LA_KD, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)

    def forward(self, x, target, soft_target, w_kd, reduction='mean'):
        x_m = x + self.m_list
        log_pred = torch.log_softmax(x_m, dim=-1)
        log_pred = torch.where(torch.isinf(log_pred), torch.full_like(log_pred, 0), log_pred)

        kl = F.kl_div(log_pred, soft_target, reduction='none')
        kl = kl.sum(dim=1)

        nll = F.nll_loss(log_pred, target, reduction='none')

        loss = w_kd * kl + (1 - w_kd) * nll

        # 根据 reduction 参数返回结果
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'none':
            return loss
        else:
            raise ValueError(f"Invalid reduction mode: {reduction}. Choose 'mean', 'sum', or 'none'.")


class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, labels, reduction='mean'):
        ce = self.cross_entropy(pred, labels)

        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))

        loss = self.alpha * ce + self.beta * rce

        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'none':
            return loss
        else:
            raise ValueError(f"Invalid reduction mode: {reduction}. Choose 'mean', 'sum', or 'none'.")


class GeneralizedCrossEntropy(nn.Module):
    def __init__(self, num_classes, q=0.7):
        super(GeneralizedCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.q = q

    def forward(self, pred, labels, reduction='mean'):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        gce = (1.0 - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q

        # 根据 reduction 参数返回结果
        if reduction == 'mean':
            return gce.mean()
        elif reduction == 'sum':
            return gce.sum()
        elif reduction == 'none':
            return gce
        else:
            raise ValueError(f"Invalid reduction mode: {reduction}. Choose 'mean', 'sum', or 'none'.")


class MeanAbsoluteError(nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(MeanAbsoluteError, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels, reduction='mean'):
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        mae = 1.0 - torch.sum(label_one_hot * pred, dim=1)

        # 根据 reduction 参数返回结果
        if reduction == 'mean':
            return self.scale * mae.mean()
        elif reduction == 'sum':
            return self.scale * mae.sum()
        elif reduction == 'none':
            return self.scale * mae
        else:
            raise ValueError(f"Invalid reduction mode: {reduction}. Choose 'mean', 'sum', or 'none'.")


def co_teaching_loss(model1_loss, model2_loss, rt):
    _, model1_sm_idx = torch.topk(model1_loss, k=int(int(model1_loss.size(0)) * rt), largest=False)
    _, model2_sm_idx = torch.topk(model2_loss, k=int(int(model2_loss.size(0)) * rt), largest=False)

    # co-teaching
    model1_loss_filter = torch.zeros((model1_loss.size(0))).to(device)
    model1_loss_filter[model2_sm_idx] = 1.0
    model1_loss = (model1_loss_filter * model1_loss).sum()

    model2_loss_filter = torch.zeros((model2_loss.size(0))).to(device)
    model2_loss_filter[model1_sm_idx] = 1.0
    model2_loss = (model2_loss_filter * model2_loss).sum()

    return model1_loss, model2_loss


class DynamicCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(DynamicCrossEntropyLoss, self).__init__()

    def forward(self, input, target, reduction='mean'):
        loss = F.cross_entropy(input, target, reduction=reduction)
        return loss


class PerturbedGradientDescent(Optimizer):
    def __init__(self, params, lr=0.01, mu=0.0):
        default = dict(lr=lr, mu=mu)
        super().__init__(params, default)

    @torch.no_grad()
    def step(self, global_params, device):
        for group in self.param_groups:
            for p, g in zip(group['params'], global_params):
                g = g.to(device)
                d_p = p.grad.data + group['mu'] * (p.data - g.data)
                p.data.add_(d_p, alpha=-group['lr'])
