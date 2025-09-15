import argparse
import os
import time
import warnings
import logging
from datetime import datetime
import torchvision
from torch.nn import Sequential

from core.flimplement.middleware.middleDisPFL import DFedDisPFL
from core.flimplement.middleware.middleProxyFL import ProxyFL
from flimplement.middleware.FedAvg import FedAvg
from flimplement.middleware.middleAvgPush import AvgPush
from flimplement.middleware.middleDFedAvg import DFedAvg
from flimplement.middleware.middleDitto import Ditto
from flimplement.middleware.middleRPT import middleRPT

from flimplement.model.models import *
from flimplement.model.resnet import resnet18

warnings.simplefilter("ignore")

torch.manual_seed(520)  # 改随机种子

# hyper-params for Text tasks
vocab_size = 98635  # 98635 for AG_News and 399198 for Sogou_News
max_len = 200
emb_dim = 32


def run(args):
    time_list = []
    model_str = args.model_str
    proxy_model_str = args.proxy_model
    if args.algorithm == "ProxyFL":
        args.use_proxy = True

    for i in range(args.prev, args.times):
        logger.info(f"\n============= Running time: {i}th =============")
        logger.info("Creating server and clients ...")
        start = time.time()


        if model_str == "cnn":  # non-convex
            if "mnist" in args.dataset:
                args.model = CNN(in_features=1, num_classes=args.num_classes).to(args.device)
            elif "Cifar100" == args.dataset:
                args.model = CNN(in_features=3, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = CNN(in_features=3, num_classes=args.num_classes).to(args.device)
            elif "omniglot" in args.dataset:
                args.model = CNN(in_features=1, num_classes=args.num_classes).to(args.device)
                # args.model = CifarNet(num_classes=args.num_classes).to(args.device)
            elif "ISIC2018" in args.dataset:
                args.model = CNN(in_features=3, num_classes=args.num_classes).to(args.device)
        elif model_str == "harcnn":
            args.model = HARCNN_Split(9, dim_hidden=2048, num_classes=args.num_classes, conv_kernel_size=(1, 9),
                                    pool_kernel_size=(1, 2)).to(args.device)
        elif model_str == "resnet":  # non-convex
            if "mnist" in args.dataset:
                args.model = resnet18(num_classes=args.num_classes).to(args.device)
            elif "Cifar100" == args.dataset:
                args.model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = resnet18(num_classes=args.num_classes).to(args.device)
            elif "omniglot" in args.dataset:
                args.model = resnet18(num_classes=args.num_classes).to(args.device)
                # args.model = CifarNet(num_classes=args.num_classes).to(args.device)
            elif "ISIC2018" in args.dataset:
                args.model = resnet18(num_classes=args.num_classes).to(args.device)
        else:
            raise NotImplementedError


        # Generate args.proxy_model
        if args.use_proxy == True:
            if proxy_model_str == "mlr":  # convex
                if "mnist" in args.dataset:
                    args.proxy_model = Mclr_Logistic(1 * 28 * 28, num_classes=args.num_classes).to(args.device)
                if "Cifar10" in args.dataset:
                    args.proxy_model = Mclr_Logistic(3 * 32 * 32, num_classes=args.num_classes).to(args.device)
                else:
                    args.proxy_model = Mclr_Logistic(60, num_classes=args.num_classes).to(args.device)

            elif proxy_model_str == "cnn":  # non-convex
                if "mnist" in args.dataset:
                    args.proxy_model = FedAvgCNNProxy(in_features=1, num_classes=args.num_classes, dim=1024).to(
                        args.device)
                elif "Cifar10" in args.dataset:
                    args.proxy_model = FedAvgCNNProxy(in_features=3, num_classes=args.num_classes, dim=1600).to(
                        args.device)
                elif "omniglot" in args.dataset:
                    args.proxy_model = FedAvgCNNProxy(in_features=1, num_classes=args.num_classes, dim=33856).to(
                        args.device)
                elif "ISIC2018" in args.dataset:
                    args.proxy_model = FedAvgCNNProxy(in_features=3, num_classes=args.num_classes, dim=179776).to(
                        args.device)
                elif "har" in args.dataset:
                    args.proxy_model = HARCNN_Split(9, dim_hidden=2048, num_classes=args.num_classes,
                                                    conv_kernel_size=(1, 9),
                                                    pool_kernel_size=(1, 2)).to(args.device)
                else:
                    args.proxy_model = FedAvgCNNProxy(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)
            else:
                raise NotImplementedError
            logger.info(args.proxy_model)

        if args.algorithm == "ProxyFL":
            server = ProxyFL(args, i)  # 初始化客户端和服务器（挑选参与训练的客户端以及训练速度慢的客户端）
        elif args.algorithm == "DisPFL":
            args.model = DisModelTrainer(args.model, args.dataset, args.erk_power_scale, args.device)
            server = DFedDisPFL(args, i)
        elif args.algorithm == "FedProNo":
            server = middleRPT(args, i)
        elif args.algorithm == "DFedAvg":
            server = DFedAvg(args, i)
        elif args.algorithm == "AvgPush":
            server = AvgPush(args, i)
        elif args.algorithm == "FedAvg":
            server = FedAvg(args, i)
        elif args.algorithm == "Ditto":
            server = Ditto(args, i)
        else:
            raise NotImplementedError
        logger.info(args.model)

        server.train()  # 训练

        time_list.append(time.time() - start)

    logger.info(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    logger.info(f"Acc:{server.rs_test_acc}")
    logger.info(f"loss:{server.rs_train_loss}")
    logger.info(f"precision:{server.rs_test_precision}")
    logger.info(f"recall:{server.rs_test_recalls}")
    logger.info(f"f1:{server.rs_test_f1}")
    # results_store(server)

    logger.info("All done!")


if __name__ == "__main__":
    total_start = time.time()
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test",
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda:1",
                        choices=["cpu", "cuda:1", "cuda:0"])
    parser.add_argument('-did', "--device_id", type=str, default="1")
    parser.add_argument('-data', "--dataset", type=str, default="fmnist")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model_str", type=str, default="cnn")
    parser.add_argument('-lbs', "--batch_size", type=int, default=64)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.1,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)

    parser.add_argument('-gr', "--global_rounds", type=int, default=150)
    parser.add_argument('-ls', "--local_epochs", type=int, default=5,
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedProNo")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=10,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-dp', "--privacy", type=bool, default=False,
                        help="differential privacy")
    parser.add_argument('-dps', "--dp_sigma", type=float, default=0.0)
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)

    parser.add_argument('-opt', "--optimizer", type=str, default='SGD',
                        help="Optimizer")  # ----------新加的-----------
    parser.add_argument('-sgd_momentum', default=0.0, type=float, help='sgd momentum')
    parser.add_argument('-sgd_weight_decay', default=1e-5, type=float, help='sgd weight decay')
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    # pFedMe / PerAvg / FedProx / FedAMP / FedPHP / GPFL
    parser.add_argument('-bt', "--beta", type=float, default=0.0)
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="Regularization weight")
    parser.add_argument('-mu', "--mu", type=float, default=0.0)
    parser.add_argument('-K', "--K", type=int, default=5,
                        help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01,
                        help="personalized learning rate to caculate theta aproximately using K steps")
    # FedFomo
    parser.add_argument('-M', "--M", type=int, default=5,
                        help="Server only sends M client models to one client at each round")
    # FedMTL
    parser.add_argument('-itk', "--itk", type=int, default=4000,
                        help="The iterations for solving quadratic subproblems")
    # FedAMP
    parser.add_argument('-alk', "--alphaK", type=float, default=1.0,
                        help="lambda/sqrt(GLOABL-ITRATION) according to the paper")
    parser.add_argument('-sg', "--sigma", type=float, default=1.0)
    # APFL
    parser.add_argument('-al', "--alpha", type=float, default=1.0)
    # Ditto / FedRep
    parser.add_argument('-pls', "--plocal_epochs", type=int, default=1)
    # MOON
    parser.add_argument('-tau', "--tau", type=float, default=1.0)
    # FedBABU
    parser.add_argument('-fte', "--fine_tuning_epochs", type=int, default=10)
    # APPLE
    parser.add_argument('-dlr', "--dr_learning_rate", type=float, default=0.0)
    parser.add_argument('-L', "--L", type=float, default=1.0)
    # FedGen
    parser.add_argument('-nd', "--noise_dim", type=int, default=512)
    parser.add_argument('-glr', "--generator_learning_rate", type=float, default=0.005)
    parser.add_argument('-hd', "--hidden_dim", type=int, default=512)
    parser.add_argument('-se', "--server_epochs", type=int, default=1000)
    parser.add_argument('-lf', "--localize_feature_extractor", type=bool, default=False)
    # SCAFFOLD / FedGH
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=1.0)
    # FedALA
    parser.add_argument('-et', "--eta", type=float, default=1.0)
    parser.add_argument('-s', "--rand_percent", type=int, default=80)
    parser.add_argument('-p', "--layer_idx", type=int, default=2,
                        help="More fine-graind than its original paper.")
    # FedKD
    parser.add_argument('-mlr', "--mentee_learning_rate", type=float, default=0.005)
    parser.add_argument('-Ts', "--T_start", type=float, default=0.95)
    parser.add_argument('-Te', "--T_end", type=float, default=0.98)
    # FedAvgDBE
    parser.add_argument('-mo', "--momentum", type=float, default=0.1)
    parser.add_argument('-klw', "--kl_weight", type=float, default=0.0)

    # FedNH
    parser.add_argument('--FedNH_smoothing', default=0.9, type=float, help='moving average parameters')
    parser.add_argument('--FedNH_server_adv_prototype_agg', default=True,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']), help='FedNH server adv agg')
    parser.add_argument('--FedNH_client_adv_prototype_agg', default=True,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']), help='FedNH client adv agg')
    parser.add_argument('--no_norm', default=False, type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                        help='Use group/batch norm or not')
    parser.add_argument('--use_sam', default=False, type=lambda x: (str(x).lower() in ['true', '1', 'yes']),
                        help='Use SAM optimizer')
    parser.add_argument('--FedNH_head_init', default="orthogonal", type=str, help='FedNH head init')
    parser.add_argument('--FedNH_lr_scheduler', default="stepwise", type=str, help='FedNH learning rate scheduler')
    parser.add_argument('-use_proxy', type=bool, default=False, help="use proxy_model or not")

    # ProxyFL
    parser.add_argument('-proxy_m', "--proxy_model", type=str, default="cnn")

    # PPDFL
    parser.add_argument('--seq_length', type=int, default=10, help='DGFL sequence length.')
    parser.add_argument('--epsilon', type=float, default=1, help="epsilon-greedy parameter.")
    parser.add_argument('--top_k', type=float, default=0.2, help="proportion of recipients to be selected.")

    # DFedAvgM
    parser.add_argument('--Mbeta', type=float, default=0.9, help="parameter beta for momentum.")
    parser.add_argument('--itr_K', type=int, default=2, help="local training iteration times.")

    # DisPFL
    parser.add_argument('--dense_ratio', type=float, default=0.5, help='local density ratio')

    parser.add_argument('--anneal_factor', type=float, default=0.5, help='anneal factor for pruning')

    parser.add_argument("--cs", type=str, default='randomSelectOne')

    parser.add_argument("--erk_power_scale", type=float, default=1)

    parser.add_argument("--dis_gradient_check", action='store_true')

    parser.add_argument("--uniform", action='store_true')

    parser.add_argument("--save_masks", action='store_true')

    parser.add_argument("--different_initial", action='store_false')

    parser.add_argument("--diff_spa", action='store_true')

    parser.add_argument("--active_ratio", type=float, default=1.0)
    parser.add_argument("--static", action='store_true')
    parser.add_argument('--bzt', type=bool, default=False, help="whether choose byzantine scenario.")
    parser.add_argument('--malicious_ratio', type=float, default=0.2, help="the ratio of malicious.")
    parser.add_argument('--poison_ratio', type=float, default=0.5, help="the ratio of poison data in malicious.")
    parser.add_argument('--malicious_ids', type=list, default=[-1], help="malicious indexes.")
    parser.add_argument('-warmup', action='store_false', help='warm up')
    parser.add_argument('--stage1', type=int, default=15, help='stage 1 rounds')
    parser.add_argument('--loss', "--loss", type=str, default="CE")
    timestamp = datetime.now().strftime("%m%d_%H%M_%S")
    parser.add_argument('--timestamp', type=str, default=timestamp)
    args = parser.parse_args()
    directory = os.path.join('results', args.dataset)
    os.makedirs(directory, exist_ok=True)
    log_filename = f"{timestamp}_{args.loss}_{args.dataset}_{args.algorithm}_{args.model_str}.log"
    log_path = os.path.join(directory, log_filename)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(log_path, mode='a', encoding='utf-8'),
                            logging.StreamHandler()
                        ])
    logger = logging.getLogger(__name__)


    logger.info("=" * 50)

    logger.info("Algorithm: {}".format(args.algorithm))
    logger.info("Local batch size: {}".format(args.batch_size))
    logger.info("Local epochs: {}".format(args.local_epochs))
    logger.info("Local learing rate: {}".format(args.local_learning_rate))
    logger.info("Local learing rate decay: {}".format(args.learning_rate_decay))
    logger.info("Use loss: {}".format(args.loss))
    if args.learning_rate_decay:
        logger.info("Local learing rate decay gamma: {}".format(args.learning_rate_decay_gamma))
    logger.info("Total number of clients: {}".format(args.num_clients))
    logger.info("Clients join in each round: {}".format(args.join_ratio))
    logger.info("Clients randomly join: {}".format(args.random_join_ratio))
    logger.info("Client drop rate: {}".format(args.client_drop_rate))
    logger.info("Client select regarding time: {}".format(args.time_select))
    if args.time_select:
        logger.info("Time threthold: {}".format(args.time_threthold))
    logger.info("Running times: {}".format(args.times))
    logger.info("Dataset: {}".format(args.dataset))
    logger.info("Number of classes: {}".format(args.num_classes))
    logger.info("Backbone: {}".format(args.model_str))
    logger.info("Using device: {}".format(args.device))
    logger.info("Using DP: {}".format(args.privacy))
    if args.privacy:
        logger.info("Sigma for DP: {}".format(args.dp_sigma))
    logger.info("Auto break: {}".format(args.auto_break))
    if not args.auto_break:
        logger.info("Global rounds: {}".format(args.global_rounds))
    # if args.device == "cuda":
    #     logger.info("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    logger.info("DLG attack: {}".format(args.dlg_eval))
    if args.dlg_eval:
        logger.info("DLG attack round gap: {}".format(args.dlg_gap))
    logger.info("Total number of new clients: {}".format(args.num_new_clients))
    logger.info("Fine tuning epoches on new clients: {}".format(args.fine_tuning_epoch_new))
    logger.info("=" * 50)

    run(args)
