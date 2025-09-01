import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import numpy as np
import copy


class CNN(nn.Module):
    def __init__(self, in_features=3, num_classes=10, dropout_rate=0.25, top_bn=False):
        self.dropout_rate = dropout_rate
        self.top_bn = top_bn
        super(CNN, self).__init__()
        self.c1 = nn.Conv2d(in_features, 128,
                            kernel_size=3, stride=1, padding=1)
        self.c2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.c3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.c4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.c5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.c6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.c7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
        self.c8 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)
        self.c9 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0)
        self.l_c1 = nn.Linear(128, num_classes)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(512)
        self.bn8 = nn.BatchNorm2d(256)
        self.bn9 = nn.BatchNorm2d(128)

    def forward(self, x):
        h = x
        h = self.c1(h)
        h = F.leaky_relu(self.bn1(h), negative_slope=0.01)
        h = self.c2(h)
        h = F.leaky_relu(self.bn2(h), negative_slope=0.01)
        h = self.c3(h)
        h = F.leaky_relu(self.bn3(h), negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.dropout2d(h, p=self.dropout_rate)

        h = self.c4(h)
        h = F.leaky_relu(self.bn4(h), negative_slope=0.01)
        h = self.c5(h)
        h = F.leaky_relu(self.bn5(h), negative_slope=0.01)
        h = self.c6(h)
        h = F.leaky_relu(self.bn6(h), negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.dropout2d(h, p=self.dropout_rate)

        h = self.c7(h)
        h = F.leaky_relu(self.bn7(h), negative_slope=0.01)
        h = self.c8(h)
        h = F.leaky_relu(self.bn8(h), negative_slope=0.01)
        h = self.c9(h)
        h = F.leaky_relu(self.bn9(h), negative_slope=0.01)
        h = F.avg_pool2d(h, kernel_size=h.data.shape[2])

        feature = h.view(h.size(0), h.size(1))

        logit = self.l_c1(feature)
        if self.top_bn:
            logit = call_bn(self.bn_c1, logit)
        return logit, feature


def call_bn(bn, x):
    return bn(x)


# batchsize*9*128*1
# class HARCNN(nn.Module):
#     def __init__(self, in_channels=9, dim_hidden=64 * 26, num_classes=6, conv_kernel_size=(1, 9),
#                  pool_kernel_size=(1, 2)):
#         super().__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels, 32, kernel_size=conv_kernel_size),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=conv_kernel_size),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2)
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(dim_hidden, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Linear(512, num_classes)
#         )
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.conv2(out)
#         out = torch.flatten(out, 1)
#         out = self.fc(out)
#         return out, 2


# ====================================================================================================================

class Mclr_Logistic(nn.Module):
    def __init__(self, input_dim=1 * 28 * 28, num_classes=10):
        super(Mclr_Logistic, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output


class HARCNN(nn.Module):
    def __init__(self,in_channels=9, dim_hidden=64 * 32, num_classes=6, conv_kernel_size=(9, 1),
                 pool_kernel_size=(2, 1)):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=conv_kernel_size, padding=(0, 4)),  # 输出: [B, 32, 1, 128]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2)  # 输出: [B, 32, 1, 64]
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=conv_kernel_size, padding=(0, 4)),  # 输出: [B, 64, 1, 64]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2)  # 输出: [B, 64, 1, 32]
        )

        self.dropout = nn.Dropout(0.3)

        # 输出为 [B, 64*1*32] = [B, 2048]
        self.fc_embed = nn.Sequential(
            nn.Linear(dim_hidden, 1024),
            nn.ReLU()
        )

        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):  # x: [B, 9, 1, 128]
        out = self.conv1(x)  # -> [B, 32, 1, 64]
        out = self.conv2(out)  # -> [B, 64, 1, 32]
        out = self.dropout(out)
        out = out.view(out.size(0), -1)  # -> [B, 2048]
        embedding = self.fc_embed(out)  # -> [B, 1024]
        output = self.classifier(embedding)  # -> [B, num_classes]
        return output, embedding

# HARCNN_Split模型

class HARCNN_Split(nn.Module):
    def __init__(self, in_channels=9, dim_hidden=1664, num_classes=6, conv_kernel_size=(1, 9), pool_kernel_size=(1, 2)):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=conv_kernel_size, padding=(0, 4)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=conv_kernel_size, padding=(0, 4)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2)
        )
        self.dropout = nn.Dropout(0.3)

        self.fc_embed = nn.Sequential(
            nn.Linear(dim_hidden, 512),
            nn.ReLU(),
            nn.Linear(512, 128)  # 或 256
        )

        self.fc_logits = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

        # self.decoder = nn.Sequential(
        #     nn.Linear(64, dim_hidden),  # 64 × 1 × 32 = 2048
        #     nn.ReLU(),
        #     nn.Unflatten(1, (64, 1, 32)),
        #
        #     # 32 → 64
        #     nn.ConvTranspose2d(64, 32, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1), output_padding=(0, 0)),
        #     nn.ReLU(),
        #
        #     # 64 → 128
        #     nn.ConvTranspose2d(32, 9, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1), output_padding=(0, 0)),
        # )
        self.decoder = nn.Sequential(
            nn.Linear(128, dim_hidden),
            nn.ReLU(),
            nn.Unflatten(1, (64, 1, 32)),

            nn.ConvTranspose2d(64, 32, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1), output_padding=(0, 0)),
            nn.BatchNorm2d(32),  # 添加
            nn.ReLU(),

            nn.ConvTranspose2d(32, 9, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1), output_padding=(0, 0)),
        )

    def forward(self, x, return_recon=False):
        out = self.conv1(x)
        # print("After conv1:", out.shape)

        out = self.conv2(out)
        # print("After conv2:", out.shape)

        out = self.dropout(out)
        out_flat = out.view(out.size(0), -1)
        # print("Flattened shape:", out_flat.shape)
        embedding = self.fc_embed(out_flat)
        logits = self.fc_logits(embedding)

        if return_recon:
            x_recon = self.decoder(embedding)
            return logits, embedding, x_recon
        else:
            return logits, embedding


class FedAvgCNNProxy(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        # base
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                      32,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                      64,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True)
        )
        # head
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc(out)
        return out


class DisModelTrainer(nn.Module):
    def __init__(self, model, dataset, erk_power_scale, device=torch.device('cuda')):
        super(DisModelTrainer, self).__init__()
        self.id = None
        self.model = model
        self.masks = None
        self.erk_power_scale = erk_power_scale
        self.dataset = dataset

    def set_masks(self, masks):
        self.masks = masks
        # self.model.set_masks(masks)

    def init_masks(self, params, sparsities):
        masks = {}
        for name in params:
            masks[name] = torch.zeros_like(params[name])
            dense_numel = int((1 - sparsities[name]) * torch.numel(masks[name]))
            if dense_numel > 0:
                temp = masks[name].view(-1)
                perm = torch.randperm(len(temp))
                perm = perm[:dense_numel]
                temp[perm] = 1
        return masks

    def forward(self, x):
        output = self.model(x)
        return output

    def calculate_sparsities(self, params, tabu=[], distribution="ERK", sparse=0.5):
        spasities = {}
        if distribution == "uniform":
            for name in params:
                if name not in tabu:
                    spasities[name] = 1 - self.args.dense_ratio
                else:
                    spasities[name] = 0
        elif distribution == "ERK":
            print('initialize by ERK')
            total_params = 0
            for name in params:
                total_params += params[name].numel()
            is_epsilon_valid = False
            dense_layers = set()

            density = sparse
            while not is_epsilon_valid:
                divisor = 0
                rhs = 0
                raw_probabilities = {}
                for name in params:
                    if name in tabu:
                        dense_layers.add(name)
                    n_param = np.prod(params[name].shape)
                    n_zeros = n_param * (1 - density)
                    n_ones = n_param * density

                    if name in dense_layers:
                        rhs -= n_zeros
                    else:
                        rhs += n_ones
                        raw_probabilities[name] = (
                                                          np.sum(params[name].shape) / np.prod(params[name].shape)
                                                  ) ** self.erk_power_scale
                        divisor += raw_probabilities[name] * n_param
                epsilon = rhs / divisor
                max_prob = np.max(list(raw_probabilities.values()))
                max_prob_one = max_prob * epsilon
                if max_prob_one > 1:
                    is_epsilon_valid = False
                    for mask_name, mask_raw_prob in raw_probabilities.items():
                        if mask_raw_prob == max_prob:
                            (f"Sparsity of var:{mask_name} had to be set to 0.")
                            dense_layers.add(mask_name)
                else:
                    is_epsilon_valid = True

            # With the valid epsilon, we can set sparsities of the remaning layers.
            for name in params:
                if name in dense_layers:
                    spasities[name] = 0
                else:
                    spasities[name] = (1 - epsilon * raw_probabilities[name])
        return spasities

    def get_model_params(self):
        return copy.deepcopy(self.model.state_dict())

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def get_trainable_params(self):
        dict = {}
        for name, param in self.model.named_parameters():
            dict[name] = param
        return dict

    def count_communication_params(self, update_to_server):
        num_non_zero_weights = 0
        for name, value in self.get_model_params().items():
            num_non_zero_weights += torch.count_nonzero(value)
        return num_non_zero_weights

    def set_id(self, trainer_id):
        self.id = trainer_id

    def screen_gradients(self, train_data, device):
        model = self.model
        model.to(device)
        model.eval()
        # # # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        # # sample one epoch  of data
        model.zero_grad()
        (x, labels) = next(iter(train_data))
        x, labels = x.to(device), labels.to(device)
        log_probs, _ = model.forward(x)
        loss = criterion(log_probs, labels.long())
        loss.backward()
        gradient = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradient[name] = param.grad.to("cpu")
            else:
                gradient[name] = torch.zeros_like(param.data).cpu()
        return gradient

    def count_training_flops_per_sample(self):
        return count_training_flops(self.model, self.dataset)

    def count_full_flops_per_sample(self):
        return count_training_flops(self.model, self.dataset, full=True)


def count_training_flops(model, dataset, full=False):
    flops = 3 * count_model_param_flops(model, dataset, full=full)
    return flops


def count_model_param_flops(model=None, dataset=None, multiply_adds=True, full=False):
    prods = {}

    def save_hook(name):
        def hook_per(self, input, output):
            prods[name] = np.prod(input[0].shape)

        return hook_per

    list_1 = []

    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))

    list_2 = {}

    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)

    list_conv = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0
        if not full:
            num_weight_params = (self.weight.data != 0).float().sum()
        else:
            num_weight_params = torch.numel(self.weight.data)
        assert self.weight.numel() == kernel_ops * output_channels, "Not match"
        flops = (num_weight_params * (
            2 if multiply_adds else 1) + bias_ops * output_channels) * output_height * output_width * batch_size
        # logging.info("-------")
        # logging.info("sparsity{}".format(num_weight_params/torch.numel(self.weight.data)))
        # logging.info("A{}".format(flops))
        list_conv.append(flops)

    list_linear = []

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1
        if not full:
            weight_ops = (self.weight.data != 0).float().sum() * (2 if multiply_adds else 1)
            bias_ops = (self.bias.data != 0).float().sum() if self.bias is not None else 0
        else:
            weight_ops = torch.numel(self.weight.data) * (2 if multiply_adds else 1)
            bias_ops = torch.numel(self.bias.data) if self.bias is not None else 0
        flops = batch_size * (weight_ops + bias_ops)
        # logging.info("L{}".format(flops))
        list_linear.append(flops)

    list_bn = []

    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement() * 2)

    list_relu = []

    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling = []

    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = 0
        flops = (kernel_ops + bias_ops) * output_channels * output_height * output_width * batch_size

        list_pooling.append(flops)

    list_upsample = []

    # For bilinear upsample
    def upsample_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        flops = output_height * output_width * output_channels * batch_size * 12
        list_upsample.append(flops)

    def foo(handles, net):

        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                handles += [net.register_forward_hook(conv_hook)]
            if isinstance(net, torch.nn.Linear):
                handles += [net.register_forward_hook(linear_hook)]
            # if isinstance(net, torch.nn.BatchNorm2d):
            #     net.register_forward_hook(bn_hook)
            # if isinstance(net, torch.nn.ReLU):
            #     net.register_forward_hook(relu_hook)
            # if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
            #     net.register_forward_hook(pooling_hook)
            # if isinstance(net, torch.nn.Upsample):
            #     net.register_forward_hook(upsample_hook)
            return
        for c in childrens:
            foo(handles, c)

    # if model == None:
    #     model = torchvision.models.alexnet()
    handles = []
    foo(handles, model)
    if dataset == "emnist" or dataset == "mnist" or "fmnist" in dataset:
        input_channel = 1
        input_res = 28
        input_size = 28
    elif dataset.startswith("Cifar10"):
        input_channel = 3
        input_res = 32
        input_size = 32
    elif dataset.startswith("Cifar100"):
        input_channel = 3
        input_res = 32
        input_size = 32
    elif dataset == "tiny":
        input_channel = 3
        input_res = 64
        input_size = 64
    elif dataset == "ISIC2018":
        input_channel = 3
        input_res = 224
        input_size=224
    elif dataset == "har":
        input_channel = 9
        input_size=1
        input_res = 128
    device = next(model.parameters()).device
    input = Variable(torch.rand(input_channel, input_size, input_res).unsqueeze(0), requires_grad=True).to(device)

    out = model(input)

    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling) + sum(
        list_upsample))
    for handle in handles:
        handle.remove()
    return total_flops
