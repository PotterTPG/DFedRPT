import time
import logging
import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from flimplement.clients.clientforget import ClientForget
from flimplement.middleware.middlebase import Middle

logger = logging.getLogger(__name__)


class Middleforget(Middle):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.args = args
        self.args.forget_round=150
        self.set_clients(ClientForget)
        logging.info(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        logging.info("Finished creating server and clients.")
        self.Budget = []
        self.acc_list = []
        self.loss_list = []
        self.global_rounds_list = []
        self.client_class_loss_matrix = np.zeros((self.num_clients, self.num_classes), dtype=np.float32)
        self.clients_loss = []
        # 新增：遗忘相关
        self.forget_triggered = False  # 标记是否已执行遗忘

    def get_high_loss_indices(self, client, proportion=0.5):
        """按损失排序，选择高损失样本（50%）作为噪声样本"""
        trainloader = client.load_train_data()
        losses = []
        indices = []
        client.model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                if isinstance(x, list):
                    x = x[0].to(client.device, non_blocking=True)
                else:
                    x = x.to(client.device, non_blocking=True)
                y = y.to(client.device, non_blocking=True)
                logits,_ = client.model(x)
                loss = client.loss(logits, y, reduction='none')
                losses.extend(loss.cpu().numpy().tolist())
                indices.extend(range(i * client.batch_size, min((i + 1) * client.batch_size, client.train_samples)))
        losses = np.array(losses)
        num_forget = int(proportion * client.train_samples)
        top_indices = np.argsort(losses)[-num_forget:]
        return [indices[i] for i in top_indices]

    def train(self):
        # -------------------------- warmup phase ------------------------------
        if self.args.warmup:
            for i in range(self.args.stage1):
                self.selected_clients = self.select_clients()
                s_t = time.time()
                for client in self.selected_clients:
                    client.train_warm_up()
                logging.info(f"Train time: {time.time() - s_t}")
            self.evaluate_warm_up()

        # -------------------------- client identification ------------------------------
        metrics = np.copy(self.client_class_loss_matrix)
        for i in range(metrics.shape[0]):
            for j in range(metrics.shape[1]):
                if np.isnan(metrics[i, j]):
                    metrics[i, j] = np.nanmin(metrics[:, j])

        for j in range(metrics.shape[1]):
            metrics[:, j] = (metrics[:, j] - metrics[:, j].min()) / (metrics[:, j].max() - metrics[:, j].min())
        vote = []
        scaler = StandardScaler()
        metrics_scaled = scaler.fit_transform(metrics)
        metrics_scaled_cleaned = metrics_scaled[~np.isnan(metrics_scaled).any(axis=1)]
        for i in range(9):
            gmm = GaussianMixture(n_components=2, random_state=i).fit(metrics_scaled_cleaned)
            gmm_pred = gmm.predict(metrics_scaled_cleaned)

            noisy_clients = np.where(gmm_pred == np.argmax(gmm.means_.sum(1)))[0]
            noisy_clients = set(list(noisy_clients))
            vote.append(noisy_clients)
        cnt = []
        for i in vote:
            cnt.append(vote.count(i))
        noisy_clients = list(vote[cnt.index(max(cnt))])

        logging.info(f"detect noisy client id: {noisy_clients}")

        for idx in noisy_clients:
            logging.info(
                f"client {idx}: total loss = {self.clients_loss[idx]:.2f}, class wise closs = {self.client_class_loss_matrix[idx]}")
        logging.info(f"Noisy clients detected: {noisy_clients}")
        # -------------------------- communicate and local train ------------------------------
        for round_number in range(self.global_rounds):
            s_t = time.time()
            self.global_rounds_list.append(round_number)

            # 触发遗忘
            if round_number == self.args.forget_round and not self.forget_triggered:
                logging.info(f"\nTriggering unlearning at round {round_number} for noisy clients: {noisy_clients}")
                for client_id in noisy_clients:
                    client = self.clients[client_id]
                    # 选择50%高损失样本
                    forget_indices = self.get_high_loss_indices(client, proportion=0.3)
                    client.set_forget_indices(forget_indices)
                    client.unlearn()
                    logging.info(f"Client {client_id} unlearned {len(forget_indices)} samples (proportion: 0.50)")
                self.forget_triggered = True
                logging.info(f"Unlearning completed for noisy clients.")
            else:
                # 正常训练
                self.selected_clients = self.select_clients()
                client_to_receiver_map = {}
                for client in self.selected_clients:
                    if self.args.cs == "ring":
                        receiver_index = (client.id + 1) % self.num_clients
                    else:
                        receiver_index = (client.id + round_number + 1) % self.num_clients
                    client_to_receiver_map[client.id] = receiver_index
                for client_id, receiver_id in client_to_receiver_map.items():
                    sender = self.clients[client_id]
                    receiver = self.clients[receiver_id]
                    if client_id not in noisy_clients or not self.forget_triggered:
                        sender.send_full_model_and_aggregate(receiver)
                    receiver.train_vanilla()

            # 评估
            if round_number % self.eval_gap == 0:
                logger.info(f"\n-------------Round number: {round_number}-------------")
                logger.info("\nEvaluate model")
                self.evaluate()
                self.save_round_result(round_number)
            print(f"Training time for round {round_number}: {time.time() - s_t:.2f} seconds")
            self.Budget.append(time.time() - s_t)

        logging.info(f"\nBest accuracy.")
        logging.info(max(self.rs_test_acc))
        logging.info(f"\nAverage time cost per round.")
        logging.info(sum(self.Budget[1:]) / len(self.Budget[1:]))

    def evaluate_warm_up(self):
        stats = self.test_metrics_warm_up()
        stats_train = self.train_warm_metrics()
        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        logging.info("Evaluating after warm-up...")
        logging.info("Averaged Train Loss: {:.4f}".format(train_loss))
        logging.info("Averaged Test Accuracy: {:.4f}".format(test_acc))
        logging.info("Std Test Accuracy: {:.4f}".format(np.std(accs)))

    def train_warm_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_warm_metrics()
            num_samples.append(ns)
            losses.append(cl * 1.0)
            print(f'Client {c.id}: train loss: {cl * 1.0 / ns}')
        ids = [c.id for c in self.clients]
        return ids, num_samples, losses

    def test_metrics_warm_up(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()

        num_samples = []
        tot_correct = []
        tot_auc = []
        tot_precision = []
        tot_recall = []
        tot_f1 = []
        for c in self.clients:
            ct, ns, auc, local_class_loss, local_loss, precision, recall, f1 = c.test_metrics_warm_up()
            self.client_class_loss_matrix[c.id, :] = local_class_loss
            self.clients_loss.append(local_loss)
            tot_correct.append(ct * 1.0)
            logging.info(
                f'Client {c.id}: Acc: {ct * 1.0 / ns}, AUC: {auc},Precision:{precision}, recall:{recall}, f1:{f1},total_loss:{local_loss},total_class_loss:{local_class_loss}')
            tot_auc.append(auc * ns)
            num_samples.append(ns)
            tot_precision.append(precision * ns)
            tot_recall.append(recall * ns)
            tot_f1.append(f1 * ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc, tot_precision, tot_recall, tot_f1




    def save_round_result(self, round_number):
        self.acc_list.append(self.rs_test_acc[-1])
        logging.info(f"Round {round_number} accuracy saved: {self.rs_test_acc[-1]}")
