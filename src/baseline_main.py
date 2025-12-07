#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import copy
import time
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details
from save_utils import (
    prepare_output_dir,
    save_config,
    save_model,
    save_metrics,
    save_final_results,
    save_loss_plot
)


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    # Prepare standardized output directory for this run.
    # Use a distinct method name so it doesn't collide with your FedBaC runs.
    out_dir = prepare_output_dir(dataset=args.dataset, method="fedavg_baseline_partition_fix")
    save_config(out_dir, vars(args))

    # For fair comparison with federated_main, stick to CPU unless you decide otherwise.
    # If you want GPU later, mirror whatever you do in federated_main.
    device = 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural network
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer perceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        global_model = MLP(dim_in=len_in, dim_hidden=64,
                           dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy initial weights
    global_weights = global_model.state_dict()

    # Training stats
    train_loss, train_accuracy = [], []
    rounds = []
    print_every = 2

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round (FedAvg baseline) : {epoch+1} |\n')

        global_model.train()

        # Sample a fraction of users
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # Local training on each selected client
        for idx in idxs_users:
            local_model = LocalUpdate(
                args=args,
                dataset=train_dataset,
                idxs=user_groups[idx],
                logger=logger
            )
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model),
                global_round=epoch
            )

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # Standard FedAvg aggregation (uniform over selected clients)
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        # Bookkeeping: average local loss
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(
                args=args,
                dataset=train_dataset,
                idxs=user_groups[c],
                logger=logger
            )
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc) / len(list_acc))
        rounds.append(epoch + 1)

        # print global training loss after every 'print_every' rounds
        if (epoch + 1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n FedAvg Baseline Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    # --- Save artifacts for this run (research-paper style) ---
    # 1) Save the final global model
    save_model(out_dir, global_model)

    # 2) Save per-round metrics (loss + train accuracy)
    save_metrics(out_dir, rounds, train_loss, train_accuracy)

    save_loss_plot(out_dir, rounds, train_loss, train_accuracy)

    # 3) Save final summary numbers (easy to reference in the paper)
    save_final_results(
        out_dir,
        final_train_acc=train_accuracy[-1],
        final_test_acc=test_acc,
        final_test_loss=test_loss,
    )

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))
