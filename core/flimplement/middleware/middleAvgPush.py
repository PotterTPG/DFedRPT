import time

import numpy as np
import logging
logger = logging.getLogger(__name__)

import csv

from flimplement.clients.clientAvgpush import clientPush
from flimplement.middleware.middlebase import Middle


class AvgPush(Middle):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_clients(clientPush)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.acc_list = []
        self.loss_list = []
        self.global_rounds_list = []
        self.row =[self.global_rounds + 1]
        self.current_rounds = 0




    def train(self):

        for i in range(self.global_rounds + 1):
            self.selected_clients = self.select_clients()
            s_t = time.time()
            self.current_rounds = i
            self.global_rounds_list.append(i)

            if i % self.eval_gap == 0:
                logger.info(f"\n-------------Round number: {i}-------------")
                logger.info("\nEvaluate model")
                self.evaluate()
                self.save_round_result(i)

            for client in self.selected_clients:
                receiver = (client.id + 2**i) % self.num_clients
                local_model = client.train()
                for other_client in self.selected_clients:
                    if other_client.id == self.selected_clients[receiver].id:
                        other_client.receive_values(local_model)

            for client in self.selected_clients:
                client.aggregate()
        print("\nBest accuracy.")
        print(max(self.rs_test_acc))



    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = self.clients

        return selected_clients