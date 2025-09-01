import logging
import time
from core.flimplement.clients.clientDFedAvg import clientDFedAvg
from flimplement.middleware.middlebase import Middle

logger = logging.getLogger(__name__)


class DFedAvg(Middle):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_clients(clientDFedAvg)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []

    def train(self):
        self.send_models()
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate")
                self.evaluate()
                self.save_round_result(i)

            for client in self.selected_clients:
                client.update_W()
                client.train()
                client.old_W = {key: value for key, value in client.model.named_parameters()}

            self.aggregate_models()

            self.Budget.append(time.time() - s_t)
            logger.info(f"{'-' * 25} time cost {'-' * 25} {self.Budget[-1]}")

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        logging.info(f"use DFedAvg to evaluate")
        logging.info(f"\nBest accuracy.")
        logging.info(max(self.rs_test_acc))
        logging.info(f"\nAverage time cost per round.")
        logging.info(sum(self.Budget[1:]) / len(self.Budget[1:]))

    def aggregate_models(self):
        # Consider a simple cycle network
        for client in self.selected_clients:
            neighbor1 = self.clients[(client.id - 1) % self.num_clients]
            neighbor2 = self.clients[(client.id + 1) % self.num_clients]
            for key in client.old_W.keys():
                client.new_W[key] = (neighbor1.old_W[key] + neighbor2.old_W[key]) / 2
