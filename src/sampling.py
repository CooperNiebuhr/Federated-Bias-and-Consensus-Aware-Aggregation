#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users, partition_seed=None):
    """
    Create a non-IID partition of MNIST using labeled shards.
    If partition_seed is provided, the partition is fully deterministic.
    """

    import numpy as np

    # Use a controlled RNG if seed given, else use global np.random
    if partition_seed is not None:
        rng = np.random.RandomState(partition_seed)
    else:
        rng = np.random

    num_shards, num_imgs = 200, 300   # 60k = 300 shards * 200 imgs
    idx_shard = list(range(num_shards))
    dict_users = {i: np.array([], dtype=np.int64) for i in range(num_users)}

    # All MNIST indices
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort by label
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # assign 2 shards per user
    for i in range(num_users):
        # deterministic if rng is seeded
        selected = rng.choice(idx_shard, 2, replace=False)
        for shard in selected:
            start = shard * num_imgs
            end = (shard + 1) * num_imgs
            dict_users[i] = np.concatenate((dict_users[i], idxs[start:end]), axis=0)

        # remove those shards from the pool
        idx_shard = list(set(idx_shard) - set(selected))

    return dict_users



def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)


def dirichlet_noniid(dataset, num_users, alpha=0.5, partition_seed=None, min_size=10):
    """
    Dirichlet label-skew split.
    Deterministic if partition_seed is provided.

    Returns: dict_users {client_id: np.ndarray of indices}
    """
    labels = _get_targets(dataset)
    num_classes = int(labels.max() + 1)

    if partition_seed is not None:
        rng = np.random.RandomState(partition_seed)
    else:
        rng = np.random

    # indices per class
    idx_by_class = [np.where(labels == c)[0] for c in range(num_classes)]
    for c in range(num_classes):
        rng.shuffle(idx_by_class[c])

    while True:
        dict_users = {i: [] for i in range(num_users)}

        for c in range(num_classes):
            idxs = idx_by_class[c]
            if len(idxs) == 0:
                continue

            # sample class proportions across clients
            props = rng.dirichlet(alpha=np.full(num_users, alpha))

            # convert proportions -> counts
            counts = (props * len(idxs)).astype(int)

            # fix rounding so sum(counts) == len(idxs)
            diff = len(idxs) - counts.sum()
            if diff > 0:
                counts[rng.randint(0, num_users, size=diff)] += 1
            elif diff < 0:
                # rare case due to rounding
                for _ in range(-diff):
                    k = rng.randint(0, num_users)
                    if counts[k] > 0:
                        counts[k] -= 1

            start = 0
            for k in range(num_users):
                end = start + counts[k]
                if end > start:
                    dict_users[k].extend(idxs[start:end].tolist())
                start = end

        sizes = [len(dict_users[k]) for k in range(num_users)]
        if min(sizes) >= min_size:
            return {k: np.array(v, dtype=np.int64) for k, v in dict_users.items()}
        # else: resample (still deterministic for same seed because rng state advances)


# bit of a quick and dirty wrapper to aid in dirichlet sampling implementation
def sample_clients(dataset, num_users, iid=False, unequal=False,
                   dirichlet=False, alpha=0.5, partition_seed=None, min_size=10):
    """
    Unified entry-point: returns dict_users, based on flags.
    """
    if iid:
        # choose iid based on dataset
        if hasattr(dataset, "targets"):
            return cifar_iid(dataset, num_users)
        return mnist_iid(dataset, num_users)

    # non-iid
    if dirichlet:
        return dirichlet_noniid(dataset, num_users, alpha=alpha,
                                partition_seed=partition_seed, min_size=min_size)

    if unequal:
        return mnist_noniid_unequal(dataset, num_users)

    # default shard non-iid based on dataset
    if hasattr(dataset, "targets"):
        return cifar_noniid(dataset, num_users)
    return mnist_noniid(dataset, num_users, partition_seed=partition_seed)

def _get_targets(dataset):
    """
    Return labels as a 1D numpy array for MNIST/CIFAR-like datasets.
    """
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets)
    if hasattr(dataset, "train_labels"):
        return dataset.train_labels.numpy()
    if hasattr(dataset, "train_targets"):
        return dataset.train_targets.numpy()
    raise AttributeError("Dataset has no recognized target field (targets/train_labels/train_targets).")
