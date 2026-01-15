import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import random_split
from .randaugment import RandAugmentMC


eurosat_mean = (0.344, 0.380, 0.408)
eurosat_std = (0.203, 0.137, 0.116)
class TransformFixMatch(object):
    def __init__(self, mean, std, norm=True, size_image=32):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect')])
        self.weak2 = transforms.Compose([
            transforms.RandomHorizontalFlip(),])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size_image,
                                  padding=int(size_image*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        self.norm = norm

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        if self.norm:
            return self.normalize(weak), self.normalize(strong), self.normalize(x)
        else:
            return weak, strong

class EuroSAT_Augmentation(Dataset):

    def __init__(self, image_paths, given_label_matrix,labels, transform):

        self.image_paths = image_paths
        self.images = image_paths
        # self.labels = labels
        self.true_labels = labels
        self.given_label_matrix = given_label_matrix
        self.transform = transform
        self.init_index()
        self.return_truelabel = False
    def __len__(self):
        return len(self.images_index)

    def __getitem__(self, index):
        img_path = self.images_index[index]
        img = Image.open(img_path).convert('RGB')
        label = self.given_label_matrix_index[index]
        each_true_label = self.true_labels_index[index]

        augmented = self.transform(img)
        if self.return_truelabel:
            each_selected_label = self.selected_label[index]
            return (augmented, label, each_true_label, each_selected_label, index)
        else:
            return (augmented, label, index)

    def init_index(self):
        self.return_truelabel = False
        self.true_labels_index = self.true_labels
        self.images_index = self.images
        self.given_label_matrix_index = self.given_label_matrix

    def set_index(self, indexes=None, reture_truelabel=False, selected_labels=None):
        self.return_truelabel = reture_truelabel

        if indexes is None:
            self.init_index()
            return

        if isinstance(indexes, torch.Tensor):
            if indexes.dtype == torch.bool:
                indexes = torch.nonzero(indexes).flatten()
            indexes = indexes.cpu().numpy()

        if isinstance(indexes, np.ndarray):
            if indexes.dtype == bool:
                indexes = np.where(indexes)[0]
            indexes = indexes.astype(int).tolist()

        self.true_labels_index = [self.true_labels[i] for i in indexes]
        self.images_index = [self.images[i] for i in indexes]

        if isinstance(self.given_label_matrix, torch.Tensor):
            tensor_indexes = torch.tensor(indexes, dtype=torch.long)
            self.given_label_matrix_index = self.given_label_matrix[tensor_indexes]
        else:
            self.given_label_matrix_index = [self.given_label_matrix[i] for i in indexes]

        if selected_labels is not None:
            self.selected_label = selected_labels


class SimpleTestDataset(Dataset):

    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = torch.tensor(labels).long()
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index]).convert('RGB')
        return self.transform(img), self.labels[index]


def load_eurosat(partial_rate, root="./data/eurosat", seed=42):

    all_image_paths = []
    all_labels = []
    class_to_idx = {}
    if not os.path.exists(root):
        raise FileNotFoundError(f"not found: {root}")

    class_dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]

    if not class_dirs:
        raise RuntimeError(f"No class directories")
    for class_idx, class_name in enumerate(sorted(os.listdir(root))):
        class_dir = os.path.join(root, class_name)
        if not os.path.isdir(class_dir):
            continue

        class_to_idx[class_name] = class_idx
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, img_name)
                all_image_paths.append(img_path)
                all_labels.append(class_idx)

    num_total = len(all_image_paths)
    num_test = int(num_total * 0.1)
    num_train = num_total - num_test

    np.random.seed(seed)
    torch.manual_seed(seed)

    indices = np.random.permutation(num_total)
    train_indices = indices[:num_train]
    test_indices = indices[num_train:]

    train_image_paths = [all_image_paths[i] for i in train_indices]
    train_labels = [all_labels[i] for i in train_indices]
    test_image_paths = [all_image_paths[i] for i in test_indices]
    test_labels = [all_labels[i] for i in test_indices]

    train_label_tensor = torch.tensor(train_labels).long()
    partialY = generate_uniform_cv_candidate_labels(train_label_tensor, partial_rate)

    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(eurosat_mean, eurosat_std)
    ])

    train_transform = TransformFixMatch(
        mean=eurosat_mean,
        std=eurosat_std,
        size_image=64
    )

    train_dataset = EuroSAT_Augmentation(
        image_paths=train_image_paths,
        labels=train_labels,
        given_label_matrix=partialY.float(),
        transform=train_transform
    )
    train_dataset.class_to_idx = class_to_idx
    test_dataset = SimpleTestDataset(
        image_paths=test_image_paths,
        labels=test_labels,
        transform=test_transform
    )
    test_dataset.class_to_idx = class_to_idx


    return train_dataset, test_dataset
def generate_uniform_cv_candidate_labels(train_labels, partial_rate=0.1):
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0
    transition_matrix =  np.eye(K)
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0],dtype=bool))] = partial_rate

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        partialY[j, :] = torch.from_numpy((random_n[j, :] < transition_matrix[train_labels[j], :]) * 1)

    print("Finish Generating Candidate Label Sets!\n")
    return partialY