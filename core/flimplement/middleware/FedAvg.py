import time
from flimplement.clients.clientAvg import clientAVG
from flimplement.middleware.middlebase import Middle
import logging

logger = logging.getLogger(__name__)


class FedAvg(Middle):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_clients(clientAVG)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                logger.info(f"\n-------------Round number: {i}-------------")
                logger.info("\nEvaluate global model")
                self.evaluate()
                self.save_round_result(i)

            for client in self.selected_clients:
                client.train()


            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            logger.info(f"{'-' * 25} time cost {'-' * 25} {self.Budget[-1]}")

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))
