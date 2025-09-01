#!/usr/bin/env bash

# GPU 0 上的任务
(
python main.py -algo FedProNo -data fmnistPairwise_dir_0.5 --device cuda:0 --cs random_selectOne --loss GCE &&
python main.py -algo FedProNo -data fmnistPairwise_dir_0.5 --device cuda:0 --cs random_selectOne --loss MAE &&
python main.py -algo Ditto -data fmnistPairwise_dir_0.5 --device cuda:0
) &
##
# GPU 1 上的任务
(
    python main.py -algo DisPFL -data fmnistPairwise_dir_0.5 --device cuda:1 --cs ring &
    python main.py -algo ProxyFL -data fmnistPairwise_dir_0.5 --device cuda:1 &
    python main.py -algo AvgPush -data fmnistPairwise_dir_0.5 --device cuda:1 --cs random_selectOne &
    python main.py -algo FedAvg -data fmnistPairwise_dir_0.5 --device cuda:1 &
    python main.py -algo DFedAvg -data fmnistPairwise_dir_0.5 --device cuda:1
) &

wait
