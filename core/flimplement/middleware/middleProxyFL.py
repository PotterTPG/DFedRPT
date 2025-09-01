import time
import numpy as np
from flimplement.clients.clientproxy import clientProxy
import logging

from flimplement.middleware.middlebase import Middle

logger = logging.getLogger(__name__)



class ProxyFL(Middle):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_clients(clientProxy)
        logger.info(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        logger.info("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.acc_list = []
        self.loss_list = []
        self.global_rounds_list = []

    def train(self):
        for i in range(self.global_rounds + 1):
            self.selected_clients = self.select_clients()
            s_t = time.time()
            self.global_rounds_list.append(i)

            if i % self.eval_gap == 0:
                logger.info(f"\n-------------Round number: {i}-------------")
                logger.info("\nEvaluate model")
                self.evaluate()
                self.save_round_result(i)
            for client in self.selected_clients:
                receiver = (client.id + 2**i) % self.num_clients
                proxy_model = client.train()
                for other_client in self.selected_clients:
                    if other_client.id == self.selected_clients[receiver].id:
                        other_client.receive_model(proxy_model)

            for client in self.selected_clients:
                client.aggregate()


            self.Budget.append(time.time() - s_t)
            logger.info(f"{'-' * 25} time cost {'-' * 25} {self.Budget[-1]:.2f}")
        logger.info("\nBest accuracy.")
        logger.info(max(self.rs_test_acc))
        logger.info("\nAverage time cost per round.")
        logger.info(sum(self.Budget[1:])/len(self.Budget[1:]))



    def select_clients(self):
        self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients,replace=False))  # 从总客户端中随机选择 当前客户端数量 这么多个客户端，不允许重复选择，再将选择结果组成为一个列表

        return selected_clients

