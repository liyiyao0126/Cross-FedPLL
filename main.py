import argparse
import logging
import os
import sys
import copy
import numpy as np
import torch
from torch import dtype
from fedavg import FedAvgSimulator
from PLCF import FederatedSimulator, ConceptDrift
from dataset.cifar_PLL import load_cifar10, load_cifar100, CIFAR_Augmentention
from dataset.svhn_PLL import load_svhn, SVHN_Augmentention
from dataset.tinyimagenet_PLL import load_tinyimagenet, TinyImageNetDataset,TransformFixMatch
from dataset.eurosat_PLL import load_eurosat, EuroSAT_Augmentation
from model.wideresnet import WideResNet
from utils.parser import set_parser
from utils.tool import set_seed
import torch.multiprocessing as mp
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"

def original_prepare_fed_data(dataset, num_clients):
    data_len = len(dataset)
    indices = np.random.permutation(data_len)
    client_indices = np.array_split(indices, num_clients)
    return [torch.utils.data.Subset(dataset, idxs) for idxs in client_indices]

def prepare_fed_data(dataset,test_dataset, num_clients,args):
    total_samples = len(dataset)
    if args.dataset not in ["eurosat"]:
        indices = np.random.permutation(total_samples)
        dataset.images = dataset.images[indices]
        dataset.true_labels = dataset.true_labels[indices]
        dataset.given_label_matrix = dataset.given_label_matrix[indices]
    num_samples_per_client = total_samples // num_clients
    drift_simulator = ConceptDrift(args)
    clients_datasets = []

    for client_id in range(num_clients):
        start_idx = client_id * num_samples_per_client
        end_idx = (client_id + 1) * num_samples_per_client if client_id != num_clients - 1 else total_samples

        client_images = dataset.images[start_idx:end_idx]
        client_labels = dataset.true_labels[start_idx:end_idx]
        client_partialY = dataset.given_label_matrix[start_idx:end_idx]


        drifted_true_labels = drift_simulator.apply_drift(client_id, client_labels)
        drifted_partialY = drift_simulator.apply_drift(client_id, client_partialY.numpy())

        for idx in range(len(drifted_true_labels)):
            true_label = int(drifted_true_labels[idx])
            if drifted_partialY[idx, true_label] != 1:
                drifted_partialY[idx, true_label] = 1
        if args.dataset == "cifar10" or args.dataset == "cifar100":
            client_dataset = CIFAR_Augmentention(
                client_images,
                torch.from_numpy(drifted_partialY).float(),
                drifted_true_labels,
                dataset.transform
            )
        elif args.dataset == "SVHN":
            client_dataset = SVHN_Augmentention(
                client_images,
                torch.from_numpy(drifted_partialY).float(),
                drifted_true_labels,
                dataset.transform
            )
        elif args.dataset == "tinyimage":
            client_dataset = TinyImageNetDataset(
                client_images,
                torch.from_numpy(drifted_partialY).float(),
                drifted_true_labels,
                TransformFixMatch(dataset.mean, dataset.std, size_image=64),
                (0.4802, 0.4481, 0.3975),(0.2302, 0.2265, 0.2262)
            )
        elif args.dataset == "eurosat":
            client_dataset = EuroSAT_Augmentation(
                client_images,
                torch.from_numpy(drifted_partialY).float(),
                drifted_true_labels,
                dataset.transform
            )

        clients_datasets.append(client_dataset)

    clients_test_datasets = []
    for client_id in range(num_clients):
        client_test = copy.deepcopy(test_dataset)

        if args.dataset == "cifar10" or args.dataset == "cifar100":
            drifted_labels = drift_simulator.apply_drift(client_id, client_test.targets)
            client_test.targets = drifted_labels

        elif args.dataset == "SVHN":
            drifted_labels = drift_simulator.apply_drift(client_id, client_test.labels)
            client_test.labels = drifted_labels
        elif args.dataset == "tinyimage":
            drifted_labels = drift_simulator.apply_drift(client_id, client_test.targets)
            client_test.targets = drifted_labels
        elif args.dataset == "eurosat":
            drifted_labels = drift_simulator.apply_drift(client_id, client_test.labels)
            client_test.labels = drifted_labels

        clients_test_datasets.append(client_test)

    for cid, client_dataset in enumerate(clients_datasets):
        if args.dataset == 'eurosat':
            partialY = client_dataset.given_label_matrix
        else:
            partialY = client_dataset.given_label_matrix.numpy()
        true_labels = client_dataset.true_labels
        error_count = 0

        for i in range(len(true_labels)):
            if partialY[i, int(true_labels[i])] != 1:
                error_count += 1

        print(f" - Client {cid}:  {error_count}")
        assert error_count == 0
    return clients_datasets, clients_test_datasets




def main():
    args = set_parser()
    args.num_gpus = torch.cuda.device_count()

    set_seed(args)
    if args.dataset == 'cifar10' or args.dataset == 'SVHN' or args.dataset == 'eurosat':
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        args.num_classes = 100
    elif args.dataset == 'tinyimage':
        args.num_classes = 200
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    args.device = device
    if args.dataset == "cifar10":
        train_all_dataset, test_dataset = load_cifar10(args.partial_rate, root=args.data_path)
    elif args.dataset == "cifar100":
        train_all_dataset, test_dataset = load_cifar100(args.partial_rate, root=args.data_path,
                                                        hierarchical=args.use_hierarchical)
    elif args.dataset == "SVHN":
        train_all_dataset, test_dataset = load_svhn(args.partial_rate, root=args.data_path + "svhn/")
    elif args.dataset == "tinyimage":
        train_all_dataset, test_dataset = load_tinyimagenet(args.partial_rate, root=args.data_path)
    elif args.dataset == "eurosat":
        train_all_dataset, test_dataset = load_eurosat(args.partial_rate, root=args.data_path)
    logging.basicConfig(format='[%(asctime)s] - %(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        level=logging.DEBUG)

    args.model_name = args.out
    logging.info(args)
    client_datasets,  client_test_datasets = prepare_fed_data(train_all_dataset, test_dataset, args.num_clients,args)

    print("\nClient datasets details:")
    for i, client_dataset in enumerate(client_datasets):
        print(f"Client {i + 1}: Length = {len(client_dataset)}")
    if args.method == 'fedavg':
        simulator = FedAvgSimulator(args, client_datasets, client_test_datasets, test_dataset)
    else:
        simulator = FederatedSimulator(args, client_datasets, client_test_datasets, test_dataset)
    print("Starting federated training...")

    for round in range(args.num_rounds):
        logging.info(f"=========== Federated Round [{round + 1}/{args.num_rounds}] ===========")
        simulator.client_train(round)
        simulator.server_aggregation(round)


if __name__ == '__main__':
    main()