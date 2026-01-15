import os
import numpy as np
import torch

class DiskStateManager:
    @staticmethod
    def save_client_state(client, base_path="./checkpoints", round=None):
        client_path = os.path.join(base_path, f"client_{client.cid}")
        round_path = os.path.join(client_path, f"round_{round}")

        os.makedirs(round_path, exist_ok=True)

        torch.save(client.local_model.state_dict(), os.path.join(round_path, "local_model.pt"))
        torch.save(client.global_model.state_dict(), os.path.join(round_path, "global_model.pt"))
        torch.save(
            client._prev_global_state,
            os.path.join(round_path, "prev_global_state.pt")
        )
        np.savez_compressed(
            os.path.join(round_path, "arrays.npz"),
            prev_clean_idx1=client.prev_clean_idx1,
            prev_clean_idx2=client.prev_clean_idx2,
            prev_sl1=client.prev_sl1,
            prev_sl2=client.prev_sl2,
            globalEOB=client.globalEOB,
            localEOB=client.localEOB,
            annotated_pool=np.array(client.annotated_pool, dtype=object)
        )

        with open(os.path.join(round_path, "meta.txt"), 'w') as f:
            f.write(f"samples_per_epoch={client.samples_per_epoch}")

    @staticmethod
    def load_client_state(client, base_path="./checkpoints", round=None):
        client_path = os.path.join(base_path, f"client_{client.cid}")
        round_path = os.path.join(client_path, f"round_{round}")

        model_path1 = os.path.join(round_path, "local_model.pt")
        model_path2 = os.path.join(round_path, "global_model.pt")
        if os.path.exists(model_path1):
            client.local_model.load_state_dict(torch.load(model_path1))
        if os.path.exists(model_path2):
            client.global_model.load_state_dict(torch.load(model_path2))

        prev_state_path = os.path.join(round_path, "prev_global_state.pt")
        if os.path.exists(prev_state_path):
            client._prev_global_state = torch.load(prev_state_path)
            if hasattr(client, 'device'):
                client._prev_global_state = {
                    k: v.to(client.device)
                    for k, v in client._prev_global_state.items()
                }
        array_path = os.path.join(round_path, "arrays.npz")
        if os.path.exists(array_path):
            arrays = np.load(array_path, allow_pickle=True)
            client.prev_clean_idx1 = arrays['prev_clean_idx1']
            client.prev_clean_idx2 = arrays['prev_clean_idx2']
            client.prev_sl1 = arrays['prev_sl1']
            client.prev_sl2 = arrays['prev_sl2']
            client.globalEOB = arrays['globalEOB']
            client.localEOB = arrays['localEOB']
            client.annotated_pool = list(arrays['annotated_pool'])

        meta_path = os.path.join(round_path, "meta.txt")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                for line in f:
                    if 'samples_per_epoch' in line:
                        client.samples_per_epoch = int(line.split('=')[1])
    @staticmethod

    def save_to_disk_global_model(client, base_path="./checkpoints", round=None):
        client_path = os.path.join(base_path, f"client_{client.cid}")
        round_path = os.path.join(client_path, f"round_{round}")

        os.makedirs(round_path, exist_ok=True)
        torch.save(client.global_model.state_dict(), os.path.join(round_path, "global_model.pt"))
