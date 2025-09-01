import numpy as np
import time
from threading import Thread

from flimplement.clients.clientDitto import clientDitto
from flimplement.middleware.middlebase import Middle
import logging

logger = logging.getLogger(__name__)


class Ditto(Middle):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_clients(clientDitto)

        logger.info(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        logger.info("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                logger.info(f"\n-------------Round number: {i}-------------")
                logger.info("\nEvaluate global models")
                self.evaluate()
                self.save_round_result(i)

            for client in self.selected_clients:
                client.ptrain()
                client.train()

            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            logger.info(f"{'-' * 25} time cost {'-' * 25} {self.Budget[-1]}")

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        logger.info("\nBest accuracy.")
        logger.info(max(self.rs_test_acc))
        logger.info("\nAverage time cost per round.")
        logger.info(sum(self.Budget[1:]) / len(self.Budget[1:]))

    def test_metrics_personalized(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()

        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics_personalized()
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics_personalized(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]

        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics_personalized()
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    def evaluate_personalized(self, acc=None, loss=None):
        stats = self.test_metrics_personalized()
        stats_train = self.train_metrics_personalized()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]

        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        logger.info("Averaged Train Loss: {:.4f}".format(train_loss))
        logger.info("Averaged Test Accurancy: {:.4f}".format(test_acc))
        logger.info("Averaged Test AUC: {:.4f}".format(test_auc))
        logger.info("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        logger.info("Std Test AUC: {:.4f}".format(np.std(aucs)))
