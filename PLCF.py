import logging
import copy
import os
import time
import torch.multiprocessing as mp
import numpy as np
import torch
from collections import OrderedDict

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import sys

from model.cnn import CNN
from model.resnet import ResNet18,ResNet18_Gray
from model.wideresnet import WideResNet
from utils.tool import set_seed
from utils.function import warmup, eval_train, select, mix_train, test_double, original_mix_train
import torch.nn.functional as F


class FederatedSimulator:
    def __init__(self, args, client_datasets, client_test_datasets, test_dataset):
        self.args = args
        self.num_clients = args.num_clients
        self.test_loader = self._prepare_test_data(test_dataset)
        self.client_test_loaders = [
            self._prepare_test_data(test_data)
            for test_data in client_test_datasets
        ]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.local_model = ResNet18(args.num_classes).to(self.device)
        self.global_model = ResNet18(args.num_classes).to(self.device)
        self.clients = []
        self.initial_local_params = copy.deepcopy(self.local_model.state_dict())
        self.initial_global_params = copy.deepcopy(self.global_model.state_dict())
        for client_id in range(self.num_clients):
            logging.info(f"Initializing client {client_id + 1}...")
            client = Client(client_id, args, client_datasets[client_id], self.client_test_loaders[client_id], self.local_model, self.global_model)
            self.clients.append(client)

    def _prepare_test_data(self, test_dataset):
        return DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            sampler=SequentialSampler(test_dataset),
            num_workers=self.args.num_workers,
            pin_memory=True,

        )

    def test_global_model(self):
        self.global_model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in self.test_loader:
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device)
                outputs = self.global_model(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)

        return total_loss / total, 100.0 * correct / total

    def client_train(self, round):
        for client in self.clients:
            client.train(round,self.initial_local_params,self.initial_global_params)

    def server_aggregation(self,round):
        global_weights = []
        total_samples = sum(len(c.dataset) for c in self.clients)

        for client in self.clients:
            client.load_from_disk(round)
            client_params = {
                k: v.cpu()
                for k, v in client.global_model.state_dict().items()
            }
            weight = len(client.dataset) / total_samples
            global_weights.append((
                client_params,
                weight
            ))

        aggregated_params = OrderedDict()
        for key in global_weights[0][0].keys():
            aggregated_params[key] = sum(
                param[key] * weight for param, weight in global_weights
            )
        self.global_model.load_state_dict(aggregated_params)

        test_loss, test_acc = self.test_global_model()

        header = "role,test_loss,test_acc(%)"
        log_data = f"server,{test_loss:.5f},{test_acc:.2f}"

        if not os.path.exists(os.path.join("./result/", self.args.out + '_server_results.csv')):
            with open(os.path.join("./result/", self.args.out + '_server_results.csv'), 'w') as f:
                f.write(header + '\n')

        if round == 0:
            with open(os.path.join("./result/", self.args.out + '_server_results.csv'), 'w') as f:
                f.write(header + '\n')
        else:
            with open(os.path.join("./result/", self.args.out + '_server_results.csv'), 'a') as f:
                f.write(log_data + '\n')

        logging.info(f"\n\033[1;35m[Server] Test Loss: {test_loss:.4f} | Acc: {test_acc:.2f}%\033[0m")
        for client in self.clients:
            client.global_model.load_state_dict(aggregated_params)
            client.save_to_disk_global_model(round)

class Client:
    def __init__(self, cid, args, dataset,test_loader, local_model, global_model):
        self._need_state_recovery = True
        self.annotated_pool = None
        self.samples_per_epoch = None
        self.cid = cid
        self.args = args
        self.dataset = dataset
        self.test_loader = test_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.local_model = local_model.to(self.device)
        self.global_model = global_model.to(self.device)
        if args.optimizer == 'sgd':
            self.local_optimizer = torch.optim.SGD(
                self.local_model.parameters(),
                lr=args.lr,
                momentum=0.9,
                weight_decay=args.wdecay
            )
            self.global_optimizer = torch.optim.SGD(
                self.global_model.parameters(),
                lr=args.lr,
                momentum=0.9,
                weight_decay=args.wdecay
            )
        elif args.optimizer == 'adam':
            self.local_optimizer = torch.optim.Adam(
                self.local_model.parameters(),
                lr=args.lr,
                weight_decay=args.wdecay
            )
            self.global_optimizer = torch.optim.Adam(
                self.global_model.parameters(),
                lr=args.lr,
                weight_decay=args.wdecay
            )

        self.localEOB = np.zeros((args.k, len(dataset), args.num_classes))
        self.globalEOB = np.zeros((args.k, len(dataset), args.num_classes))
        self.train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
        self.train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=self.train_sampler(dataset),
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
        self.prev_clean_idx1 = None
        self.prev_clean_idx2 = None
        self.prev_sl1 = None
        self.prev_sl2 = None
        self.num_classes = args.num_classes
        self.N_k = np.zeros(self.num_classes)
        for idx in range(len(dataset)):
            label = dataset.given_label_matrix[idx].item()
            self.N_k[label] += 1
        self.N_k = torch.from_numpy(self.N_k).float().to(self.device)

    def save_to_disk(self, round):
        from utils.storage import DiskStateManager
        DiskStateManager.save_client_state(self,base_path=os.path.join("./checkpoints/", self.args.out), round=round)
    def save_to_disk_global_model(self,round):
        from utils.storage import DiskStateManager
        DiskStateManager.save_to_disk_global_model(self, base_path=os.path.join("./checkpoints/", self.args.out), round=round)

    def load_from_disk(self, round):
        from utils.storage import DiskStateManager
        try:
            DiskStateManager.load_client_state(self,base_path=os.path.join("./checkpoints/", self.args.out), round=round)
            self._need_state_recovery = False
        except FileNotFoundError:
            self._init_default_states()

    def _init_default_states(self):
        if self._need_state_recovery:
            self.prev_clean_idx1 = np.array([], dtype=np.int64)
            self.prev_clean_idx2 = np.array([], dtype=np.int64)
            self.prev_sl1 = np.array([], dtype=np.int64)
            self.prev_sl2 = np.array([], dtype=np.int64)
            self._need_state_recovery = False
    def train(self, round, initial_local_params, initial_global_params):

        if round == 0:
            self.local_model.load_state_dict(initial_local_params)
            self.global_model.load_state_dict(initial_global_params)
        elif  round == 1:
            self.load_from_disk(round - 1)
        else:
            self.load_from_disk(round - 1)
            self.local_model.load_state_dict(self._prev_global_state)

        self._prev_global_state = copy.deepcopy(self.global_model.state_dict())

        for epoch in range(self.args.epochs):
            self._train_epoch(epoch, round)
        self.save_to_disk(round)

    def _train_epoch(self, epoch, round):
        begin_epoch = time.time()
        sr1 = 0
        acc1 = 0
        sr2 = 0
        acc2 = 0


        logging.info(f"device of epoch {epoch} client {self.cid}: {self.device}")
        if round == 0 and epoch < self.args.warm_up:
            # We will opensource function.py when the paper is accepted
            train_loss1 = warmup(self.args, self.train_loader, self.local_model, self.local_optimizer, epoch,self.device)
            train_loss2 = warmup(self.args, self.train_loader, self.global_model, self.global_optimizer, epoch,self.device)
        else:
            max_retries = 500
            retry_count = 0

            if epoch == self.args.warm_up:
                while retry_count < max_retries:
                    # 执行样本选择
                    clean_idx1, sr1, sl1 = select(self.globalEOB, self.dataset.given_label_matrix.numpy(), self.args)
                    clean_idx2, sr2, sl2 = select(self.localEOB, self.dataset.given_label_matrix.numpy(), self.args)

                    if len(clean_idx1) > 0 and len(clean_idx2) > 0:
                        break

                    logging.warning(f"Empty selection at first selection (epoch {epoch}), retrying warmup {retry_count + 1}/{max_retries}")
                    train_loss1 = warmup(self.args, self.train_loader, self.local_model, self.local_optimizer, epoch,self.device)
                    train_loss2 = warmup(self.args, self.train_loader, self.global_model, self.global_optimizer, epoch,self.device)

                    self.localEOB = eval_train(self.local_model, self.localEOB, self.train_loader, self.args, self.device)
                    self.globalEOB = eval_train(self.global_model, self.globalEOB, self.train_loader, self.args,self.device)

                    retry_count += 1

                if len(clean_idx1) == 0 or len(clean_idx2) == 0:
                    raise RuntimeError("Failed to select valid samples after maximum retries")

            else:
                clean_idx1, sr1, sl1 = select(self.globalEOB, self.dataset.given_label_matrix.numpy(), self.args)
                clean_idx2, sr2, sl2 = select(self.localEOB, self.dataset.given_label_matrix.numpy(), self.args)

            if len(clean_idx1) == 0:
                print("clean_idx1: ", clean_idx1)
                clean_idx1 = self.prev_clean_idx1.copy()
                sl1 = self.prev_sl1.copy()


            if len(clean_idx2) == 0:
                print("clean_idx2: ", clean_idx2)
                clean_idx2 = self.prev_clean_idx2.copy()
                sl2 = self.prev_sl2.copy()

            if epoch < self.args.warm_up:
                pass
            elif epoch == self.args.warm_up:
                self.select_samples_to_annotate(0)
            else:
                annotated_indices, annotated_labels = self.select_samples_to_annotate(epoch)
                if annotated_indices:
                    merged_indices1, index1 = np.unique(np.concatenate([clean_idx1, annotated_indices]),
                                                        return_index=True)
                    clean_idx1 = merged_indices1[np.argsort(index1)]
                    sl1 = np.concatenate([sl1, annotated_labels])[np.sort(index1)][:len(clean_idx1)].astype(np.int64)

                    merged_indices2, index2 = np.unique(np.concatenate([clean_idx2, annotated_indices]),
                                                        return_index=True)
                    clean_idx2 = merged_indices2[np.argsort(index2)]
                    sl2 = np.concatenate([sl2, annotated_labels])[np.sort(index2)][:len(clean_idx2)].astype(np.int64)

            self.prev_clean_idx1 = clean_idx1.copy()
            self.prev_clean_idx2 = clean_idx2.copy()
            self.prev_sl1 = sl1.copy()
            self.prev_sl2 = sl2.copy()

            train_loss1, acc1 = self._mix_train(self.local_model, self.local_optimizer, clean_idx1, sl1,epoch)
            train_loss2, acc2 = self._mix_train(self.global_model, self.global_optimizer, clean_idx2, sl2,epoch)

        if self.args.epochs == 200:
            self.local_scheduler.step()
            self.global_scheduler.step()
        self.localEOB = eval_train(self.local_model, self.localEOB, self.train_loader, self.args,self.device)
        self.globalEOB = eval_train(self.global_model, self.globalEOB, self.train_loader, self.args,self.device)
        self.valloss, self.valacc = test_double(self.args, self.test_loader, self.local_model, self.global_model,self.device)
        header = f'round,client,epoch,time(s),train_loss_l,train_loss_g,selected_ratio1(%),selcted_acc1(%),selected_ratio2(%),selcted_acc2(%),test_loss,test_acc(%)'
        data_row = (
            f'{round+1:03d},'
            f'{(self.cid + 1):03d},'
            f'{(epoch + 1):03d},'
            f'{int((time.time() - begin_epoch)):05d},'
            f'{train_loss1:.6f},'
            f'{train_loss2:.6f},'
            f'{sr1:.4f},'
            f'{acc1:.2f},'
            f'{sr2:.4f},'
            f'{acc2:.2f},'
            f'{self.valloss:.5f},'
            f'{self.valacc:.2f}'
        )



        if round == 0 and epoch == 0 and self.cid == 0:
            with open(os.path.join("./result/", self.args.out + '_training_results.csv'), 'w') as f:
                f.write(header + '\n')
        with open(os.path.join("./result/", self.args.out + '_training_results.csv'), 'a') as f:
            f.write(data_row + '\n')

        header_out = f'{"round":>8},{"client":>8},{"epoch":>8},{"time(s)":>8},{"train_loss_l":>12},{"train_loss_g":>12},{"selected_ratio1(%)":>18},{"selcted_acc1(%)":>16},{"selected_ratio2(%)":>18},{"selcted_acc2(%)":>16},{"test_loss":>12},{"test_acc(%)":>12}'
        data_row_out = (
            f'{(round + 1):>8},'
            f'{(self.cid + 1):>8},'
            f'{(epoch + 1):>8},'
            f'{int((time.time() - begin_epoch)):>8},'
            f'{train_loss1:>12.6f},'
            f'{train_loss2:>12.6f},'
            f'{sr1:>18.4f},'
            f'{acc1:>16.2f},'
            f'{sr2:>18.4f},'
            f'{acc2:>16.2f},'
            f'{self.valloss:>12.5f},'
            f'{self.valacc:>12.2f}'
        )

        logging.info("\n\033[1;36m" + header_out + "\n\033[1;32m" + data_row_out + "\033[0m")

    def _mix_train(self, model, optimizer, clean_idx, selected_labels, epoch):


        labeled_dataset = copy.deepcopy(self.dataset)
        labeled_dataset.init_index()
        labeled_dataset.set_index(clean_idx, reture_truelabel=True, selected_labels=selected_labels)

        labeled_loader = DataLoader(
            labeled_dataset,
            batch_size=self.args.batch_size,
            sampler=self.train_sampler(labeled_dataset),
            num_workers=self.args.num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=False
        )

        if self.args.original_mix == 1:
            return original_mix_train(self.args, labeled_loader, self.train_loader,
            model, optimizer, epoch)
        else:
            return mix_train(
                self.args, labeled_loader, self.train_loader,
                model, optimizer, epoch, self.device
            )

    def select_samples_to_annotate(self, epoch):
        if epoch == 0:
            num_samples = len(self.dataset)
            num_to_annotate = int(num_samples * self.args.annotation_ratio/self.args.num_rounds)

            personalized_indices = self.select_personalized_samples()
            all_annotated_indices = personalized_indices[:num_to_annotate]

            true_labels = self.get_true_labels(all_annotated_indices)
            self.annotated_pool = list(zip(all_annotated_indices, true_labels))

            self.samples_per_epoch = max(1, len(self.annotated_pool) // (self.args.epochs-self.args.warm_up))
            return [], []

        if self.annotated_pool:
            current_batch = self.annotated_pool[:self.samples_per_epoch*(epoch-self.args.warm_up)]
            indices = [x[0] for x in current_batch]
            labels = [x[1] for x in current_batch]
            return indices, labels

        return [], []

    def select_personalized_samples(self):
        T = self.args.annotation_T

        local_probs_history = torch.from_numpy(self.localEOB[-T:]).float().to(self.device)
        global_probs_history = torch.from_numpy(self.globalEOB[-T:]).float().to(self.device)

        weighted_local_probs = []
        weighted_global_probs = []

        for t in range(T):
            local_probs = F.softmax(local_probs_history[t], dim=1)
            global_probs = F.softmax(global_probs_history[t], dim=1)

            weighted_local = local_probs * self.N_k.unsqueeze(0)
            weighted_global = global_probs * self.N_k.unsqueeze(0)

            weighted_local_norm = weighted_local / (weighted_local.sum(dim=1, keepdim=True) + 1e-12)
            weighted_global_norm = weighted_global / (weighted_global.sum(dim=1, keepdim=True) + 1e-12)

            weighted_local_probs.append(weighted_local_norm)
            weighted_global_probs.append(weighted_global_norm)

        weighted_local_probs = torch.stack(weighted_local_probs)
        weighted_global_probs = torch.stack(weighted_global_probs)

        kl_divs = []
        for i in range(len(self.dataset)):
            local_i = weighted_local_probs[:, i, :]
            global_i = weighted_global_probs[:, i, :]

            kl_per_t = F.kl_div(
                input=torch.log(local_i + 1e-12),
                target=global_i,
                reduction='none'
            ).sum(dim=1)  # [T]

            avg_kl = kl_per_t.mean()
            kl_divs.append(avg_kl)

        kl_divs = torch.stack(kl_divs)  # [N]

        num_select = int(len(kl_divs) * self.args.annotation_ratio)
        topk_indices = torch.topk(kl_divs, k=num_select, largest=True).indices
        return topk_indices.cpu().numpy()

    def get_true_labels(self, indices):
        true_labels = []
        for idx in indices:
            label = self.dataset.true_labels[idx]
            true_labels.append(label)
        return true_labels


class ConceptDrift:
    def __init__(self, args):

        self.num_clients = args.num_clients
        self.drift_rate = args.drift_rate
        self.client_configs = {}
        num_drifted = int(self.num_clients * self.drift_rate)
        drifted_clients = np.random.choice(
            self.num_clients,
            size=num_drifted,
            replace=False
        )

        for cid in range(self.num_clients):
            rng = np.random.RandomState(cid + args.seed)

            if cid in drifted_clients:
                if args.dataset == 'cifar10':
                    swap_pair = tuple(rng.choice(10, size=2, replace=False))
                elif args.dataset == 'cifar100':
                    swap_pair = tuple(rng.choice(100, size=2, replace=False))
                elif args.dataset == 'SVHN':
                    swap_pair = tuple(rng.choice(10, size=2, replace=False))
                elif args.dataset == 'tinyimage':
                    swap_pair = tuple(rng.choice(200, size=2, replace=False))
                elif args.dataset == 'eurosat':
                    swap_pair = tuple(rng.choice(10, size=2, replace=False))
                else:
                    raise ValueError("Invalid dataset")
                self.client_configs[cid] = {
                    'drifted': True,
                    'swap_pair': swap_pair
                }
            else:
                self.client_configs[cid] = {'drifted': False}

        self._log_drift_config()

    def apply_drift(self, client_id, labels):
        cfg = self.client_configs[client_id]

        if not cfg['drifted']:
            return labels

        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()

        original, swap = cfg['swap_pair']

        swap_map = {
            original: swap,
            swap: original
        }

        return np.vectorize(
            lambda x: swap_map.get(x, x),
            otypes=[np.int64]
        )(labels)

    def _log_drift_config(self):
        drifted_clients = [cid for cid, cfg in self.client_configs.items() if cfg['drifted']]
        drift_ratio = len(drifted_clients) / self.num_clients * 100

        logging.info("\n\033[1;35m[Concept Drift Configuration]\033[0m")
        logging.info(f"Total clients: {self.num_clients}")
        logging.info(f"Drift rate: {self.drift_rate:.2f}")
        logging.info(f"Drifted clients: {len(drifted_clients)} ({drift_ratio:.1f}%)")

        sample_size = self.num_clients
        sample_clients = list(self.client_configs.items())[:sample_size]

        header = f'{"Client":<8}| {"Drifted":<10}| Swap Pair'
        separator = '-' * 35
        logging.debug("\n\033[1;36mSample Configurations:\n" + header + "\n" + separator)

        for cid, cfg in sample_clients:
            status = "Yes" if cfg['drifted'] else "No"
            pair = cfg.get('swap_pair', 'N/A')
            logging.debug(f"{cid:<8}| {status:<10}| {pair}")