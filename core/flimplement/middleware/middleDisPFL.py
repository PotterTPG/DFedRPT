import copy
import time
from collections import Counter
import numpy as np
import torch


from flimplement.clients.clientDisPFL import clientDisPFL
from core.utils.disPFL_util import hamming_distance
from core.utils.data_utils import read_client_data
from flimplement.clients.clientbase import Client
from flimplement.middleware.middlebase import Middle
import logging
logger = logging.getLogger(__name__)

class DFedDisPFL(Middle):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.args = args
        self.global_rounds = args.global_rounds
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.clients = []
        self.selected_clients = []
        self.eval_gap = args.eval_gap
        self.save_folder_name = args.save_folder_name
        self.global_model = copy.deepcopy(args.model)
        self.stat_info = None
        self.mask_pers_local = []
        self.set_clients(clientDisPFL)
        self.init_stat_info()
        logger.info(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        logger.info("Finished creating server and clients.")

        params = self.global_model.get_trainable_params()
        w_global = self.global_model.get_model_params()
        if self.args.uniform:
            sparsities = self.global_model.calculate_sparsities(params, distribution="uniform",
                                                                sparse=self.args.dense_ratio)
        else:
            sparsities = self.global_model.calculate_sparsities(params, sparse=self.args.dense_ratio)
        if not self.args.different_initial:
            temp = self.global_model.init_masks(params, sparsities)
            for client in self.clients:
                client.mask_local = copy.deepcopy(temp)
                client.mask_shared = copy.deepcopy(client.mask_local)
        elif not self.args.diff_spa:
            for client in self.clients:
                client.mask_local = copy.deepcopy(self.global_model.init_masks(params, sparsities))
                client.mask_shared = copy.deepcopy(client.mask_local)
        else:
            divide = 5
            p_divide = [0.2, 0.4, 0.6, 0.8, 1.0]
            for i, client in zip(self.clients):
                sparsities = self.global_model.calculate_sparsities(params, sparse=p_divide[i % divide])
                temp = self.global_model.init_masks(params, sparsities)
                client.mask_local = temp
                client.mask_shared = copy.deepcopy(client.mask_local)
                client.spa_ratio = p_divide[i % divide]
        for client in self.clients:
            for name in client.mask_local:
                client.model.model.state_dict()[name].copy_(w_global[name] * client.mask_local[name])
                client.updates_matrix.model.state_dict()[name].copy_(
                    torch.zeros_like(client.updates_matrix.model.state_dict()[name]))
            client.model_last_round.load_state_dict(client.model.state_dict())
            client.mask_shared_last_round = copy.deepcopy(client.mask_shared)

        self.dist_locals = np.zeros(shape=(self.num_clients, self.num_clients))
        self.Budget = []
        self.acc_list = []
        self.loss_list = []
        self.global_rounds_list = []
        self.active_ratio = args.active_ratio

    def train(self):
        global w_local_mdl
        for i in range(self.global_rounds + 1):
            self.global_rounds_list.append(i)
            self.selected_clients = self.select_clients()
            if i % self.eval_gap == 0:  # eval_gap应该是评估间隔，如果轮数能被评估间隔整除，就进行一次模型评估
                logger.info(f"\n-------------Round number: {i}-------------")
                logger.info("\nEvaluate model")
                self.evaluate()
                self.save_round_result(i)
            active_this_round_client = np.array(
                [1 if client in self.selected_clients else 0 for client in self.clients])
            for client in self.clients:
                client.model_last_round = client.model
                client.mask_shared_last_round = client.mask_shared
                if active_this_round_client[client.id] == 0:
                    logger.info(
                        '@@@@@@@@@@@@@@@@ Client Drop this round CM({}) with spasity {}: {}'.format(i,
                                                                                                    client.spa_ratio,
                                                                                                    client.id))

                logger.info(
                    '@@@@@@@@@@@@@@@@ Training Client CM({}) with spasity {}: {}'.format(i, client.spa_ratio,
                                                                                         client.id))
                # 记录当前mask变化了多少
                self.dist_locals[client.id][client.id], total_dis = hamming_distance(client.mask_shared_last_round,
                                                                                     client.mask_local)
                logger.info("local mask changes: {} / {}".format(self.dist_locals[client.id][client.id], total_dis))
                if active_this_round_client[client.id] == 0:
                    nei_indexs = np.array([])
                else:
                    nei_indexs = self._benefit_choose(i, client.id, self.num_clients,
                                                      self.num_join_clients, self.dist_locals[client.id], total_dis,
                                                      self.args.cs, active_this_round_client)
                if client.id not in nei_indexs:
                    nei_indexs = np.append(nei_indexs, client.id)

                nei_indexs = np.sort(nei_indexs)
                for tmp_idx in nei_indexs:
                    if tmp_idx != client.id:
                        self.dist_locals[client.id][tmp_idx], _ = hamming_distance(client.mask_local,
                                                                                   self.clients[
                                                                                       tmp_idx].mask_shared_last_round)

                if self.args.cs != "full":
                    logger.info("choose client_indexes: {}, accoring to {}".format(str(nei_indexs), self.args.cs))
                else:
                    logger.info("choose client_indexes: {}, accoring to {}".format(str(nei_indexs), self.args.cs))
                if active_this_round_client[client.id] != 0:
                    nei_distances = [self.dist_locals[client.id][i] for i in nei_indexs]
                    logger.info("choose mask diff: " + str(nei_distances))

                    # Update each client's local model and the so-called consensus model
                    if active_this_round_client[client.id] == 1:
                        w_local_mdl, w_per_global_state = self._aggregate_func(client.id, self.num_clients,
                                                                               self.current_num_join_clients,
                                                                               nei_indexs,
                                                                               client.model_last_round, client.mask_local,
                                                                               client.mask_shared_last_round)
                        client.w_per_global.set_model_params(w_per_global_state)
                    else:
                        w_local_mdl = copy.deepcopy(client.model_last_round)
                        client.w_per_global = copy.deepcopy(client.model_last_round)

                client.mask_shared = copy.deepcopy(client.mask_local)

                new_mask, w_local_mdl, updates_matrix_state, training_flops, num_comm_params = client.train(
                    copy.deepcopy(w_local_mdl),
                    i)
                client.updates_matrix.set_model_params(updates_matrix_state)

                client.mask_local = copy.deepcopy(new_mask)

                for key in client.w_per_global.model.state_dict().keys():
                    client.w_per_global.model.state_dict()[key] += client.updates_matrix.model.state_dict()[key]
                self.stat_info["sum_training_flops"] += training_flops
                self.stat_info["sum_comm_params"] += num_comm_params

        for clientOut in self.clients:
            tmp_dist = []
            for clientIn in self.clients:
                tmp, _ = hamming_distance(clientOut.mask_local, clientIn.mask_local)
                tmp_dist.append(tmp.item())
            self.stat_info["mask_dis_matrix"].append(tmp_dist)

        ## uncomment this if u like to save the final mask; Note masks for Resnet could be large, up to 1GB for 100 clients
        if self.args.save_masks:
            saved_masks = [{} for index in range(len(self.mask_pers_local))]
            for index, client in enumerate(self.clients):
                for name in client.mask_local:
                    saved_masks[index][name] = client.mask_local[name].data.bool()
            self.stat_info["final_masks"] = saved_masks
        return

    def _client_sampling(self, i, num_clients, client_num_per_round):
        if num_clients == client_num_per_round:
            client_indexes = [client_index for client_index in range(num_clients)]
        else:
            num_clients = min(client_num_per_round, num_clients)
            np.random.seed(i)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(num_clients), num_clients, replace=False)
        logger.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _benefit_choose(self, i, cur_clnt, num_clients, client_num_per_round, dist_local, total_dist,
                        cs=False, active_this_round_client=None):

        if cs == "random":
            # Random selection of available clients
            num_clients = min(client_num_per_round, num_clients)
            client_indexes = np.random.choice(range(num_clients), num_clients, replace=False)
            while cur_clnt in client_indexes:
                client_indexes = np.random.choice(range(num_clients), num_clients, replace=False)

        elif cs == "ring":
            # Ring Topology in Decentralized setting
            left = (cur_clnt - 1 + num_clients) % num_clients
            right = (cur_clnt + 1) % num_clients
            client_indexes = np.asarray([left, right])

        elif cs == "full":
            # Fully-connected Topology in Decentralized setting
            client_indexes = np.array(np.where(active_this_round_client == 1)).squeeze()
            client_indexes = np.delete(client_indexes, int(np.where(client_indexes == cur_clnt)[0]))

        elif cs == "random_selectOne":
            num_clients = min(client_num_per_round, num_clients)
            available_clients = list(set(range(num_clients)) - {cur_clnt})  # 创建一个不包括 cur_clnt 的客户端列表
            client_indexes = np.random.choice(available_clients, 1, replace=False)  # 从修改后的列表中选择
        return client_indexes

    def _aggregate_func(self, cur_idx, num_clients, client_num_per_round, nei_indexs, w_per_mdls_lstrd, mask_pers,
                        mask_pers_shared_lstrd):
        logger.info('Doing local aggregation!')

        # Initialize the count_mask as zeros with the same structure as the shared masks
        count_mask = copy.deepcopy(mask_pers_shared_lstrd)
        for k in count_mask.keys():
            count_mask[k] = count_mask[k] - count_mask[k]
            for client in nei_indexs:
                count_mask[k] += self.clients[client].mask_shared_last_round[k]
        for k in count_mask.keys():
            count_mask[k] = count_mask[k].float()

            mask = count_mask[k] != 0

            output = torch.zeros_like(count_mask[k], dtype=torch.float)

            output[mask] = 1.0 / count_mask[k][mask]

            count_mask[k] = output
        w_tmp = copy.deepcopy(w_per_mdls_lstrd.model.state_dict())
        bn_layers = [name for name, module in self.clients[client].model_last_round.model.named_modules()
                     if isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.BatchNorm1d)]
        bn_keys = set()
        for layer in bn_layers:
            bn_keys.update({f"{layer}.running_mean", f"{layer}.running_var", f"{layer}.weight", f"{layer}.bias", f"{layer}.num_batches_tracked"})

        for k in w_tmp.keys():
            if k in bn_keys:  # 如果是 BatchNorm 参数，则跳过
                continue
            w_tmp[k] = w_tmp[k] - w_tmp[k]
            for client in nei_indexs:
                w_tmp[k] += count_mask[k] * self.clients[client].model_last_round.model.state_dict()[k]
        w_p_g = copy.deepcopy(w_tmp)
        for name in mask_pers:
            w_tmp[name] = w_tmp[name] * mask_pers[name]
        return w_tmp, w_p_g


    def _local_test_on_all_clients(self, tst_results_ths_round, i):
        logger.info(
            "################local_test_on_all_clients after local training in communication round: {}".format(
                i))
        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }
        for client_idx in range(self.num_clients):
            # test data
            test_metrics['num_samples'].append(copy.deepcopy(tst_results_ths_round[client_idx]['test_total']))
            test_metrics['num_correct'].append(copy.deepcopy(tst_results_ths_round[client_idx]['test_correct']))
            test_metrics['losses'].append(copy.deepcopy(tst_results_ths_round[client_idx]['test_loss']))

            """
                Note: CI environment is CPU-based computing. 
                The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
                """
            if self.args.ci == 1:
                break

        # # test on test dataset
        test_acc = sum([test_metrics['num_correct'][i] / test_metrics['num_samples'][i] for i in
                        range(self.num_clients)]) / self.num_clients
        test_loss = sum([np.array(test_metrics['losses'][i]) / np.array(test_metrics['num_samples'][i]) for i in
                         range(self.num_clients)]) / self.num_clients

        stats = {'test_acc': test_acc, 'test_loss': test_loss}

        logger.info(stats)
        self.stat_info["old_mask_test_acc"].append(test_acc)

    def _local_test_on_all_clients_new_mask(self, tst_results_ths_round, i):
        logger.info(
            "################local_test_on_all_clients before local training in communication round: {}".format(
                i))
        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }
        for client_idx in range(self.num_clients):

            # test data
            test_metrics['num_samples'].append(copy.deepcopy(tst_results_ths_round[client_idx]['test_total']))
            test_metrics['num_correct'].append(copy.deepcopy(tst_results_ths_round[client_idx]['test_correct']))
            test_metrics['losses'].append(copy.deepcopy(tst_results_ths_round[client_idx]['test_loss']))

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break

        # # test on test dataset
        test_acc = sum([test_metrics['num_correct'][i] / test_metrics['num_samples'][i] for i in
                        range(self.num_clients)]) / self.num_clients
        test_loss = sum([np.array(test_metrics['losses'][i]) / np.array(test_metrics['num_samples'][i]) for i in
                         range(self.num_clients)]) / self.num_clients

        stats = {'test_acc': test_acc, 'test_loss': test_loss}

        logger.info(stats)
        self.stat_info["new_mask_test_acc"].append(test_acc)

    def record_avg_inference_flops(self, w_global, mask_pers=None):
        inference_flops = []
        for client_idx in range(self.num_clients):

            if mask_pers == None:
                inference_flops += [self.model_trainer.count_inference_flops(w_global)]
            else:
                w_per = {}
                for name in mask_pers[client_idx]:
                    w_per[name] = w_global[name] * mask_pers[client_idx][name]
                inference_flops += [self.model_trainer.count_inference_flops(w_per)]
        avg_inference_flops = sum(inference_flops) / len(inference_flops)
        self.stat_info["avg_inference_flops"] = avg_inference_flops

    def init_stat_info(self):
        self.stat_info = {}
        self.stat_info["label_num"] = self.num_classes
        self.stat_info["sum_comm_params"] = 0
        self.stat_info["sum_training_flops"] = 0
        self.stat_info["avg_inference_flops"] = 0
        self.stat_info["old_mask_test_acc"] = []
        self.stat_info["new_mask_test_acc"] = []
        self.stat_info["final_masks"] = []
        self.stat_info["mask_dis_matrix"] = []

    def select_clients(self):
        active_client_index = np.random.choice([0, 1], size=self.num_clients,
                                               p=[1.0 - self.active_ratio, self.active_ratio])
        active_clients = [client for client, active in zip(self.clients, active_client_index) if active]
        if self.random_join_ratio:
            self.current_num_join_clients = \
                np.random.choice(range(self.num_join_clients, self.num_clients + 1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients  # 设置当前客户端数量
        selected_clients = active_clients

        return selected_clients
