#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import math
from collections import defaultdict, deque

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details, flatten_tensor_dict
from save_utils import (
    prepare_output_dir,
    save_config,
    save_model,
    save_metrics,
    save_final_results,
    save_loss_plot,
    save_reliability_metrics,
)





class ReliabilityTracker:
    """
    Tracks a small history of consensus cosines per client and converts
    their variance into a reliability score R_i = exp(-alpha * Var).
    """
    def __init__(self, num_clients: int, window: int = 5, alpha: float = 1.0):
        self.window = window
        self.alpha = alpha
        self.history = {
            i: deque(maxlen=window) for i in range(num_clients)
        }

    def update(self, client_id: int, cos_val):
        """
        Push a new cosine value for client_id. cos_val can be None
        (in which case we ignore it).
        """
        if cos_val is None:
            return
        self.history[client_id].append(float(cos_val))

    def get_reliability(self, client_id: int) -> float:
        """
        Return R_i for this client based on the variance of recent cosines.
        If we have <= 1 sample, return a neutral reliability of 1.0.
        """
        hist = self.history[client_id]
        if len(hist) <= 1:
            return 1.0
        var = float(np.var(hist))
        return float(np.exp(-self.alpha * var))


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    # Prepare standardized output directory for this run
    out_dir = prepare_output_dir(dataset=args.dataset, method="fedbac_full")
    save_config(out_dir, vars(args))

    # Device setup
    if getattr(args, "gpu", False) and torch.cuda.is_available():
        if hasattr(args, "gpu_id") and args.gpu_id is not None:
            torch.cuda.set_device(args.gpu_id)
        device = torch.device("cuda")
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[INFO] Using CPU")
    # device = 'cpu'


    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
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

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    rounds = []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    # --- Early stopping state ---
    best_acc = 0.0
    best_round = 0
    best_global_weights = copy.deepcopy(global_weights)
    early_stop_counter = 0
    
    # Initialize server-side momentum vector
    vbar = {
        name: torch.zeros_like(param, device=device)
        for name, param in global_weights.items()
    }
    param_keys = list(global_weights.keys())

        # Reliability tracker hyperparams (fallback defaults)
    H = getattr(args, "reliab_window", 5)
    alpha_R = getattr(args, "reliab_alpha", 1.0)

    reliability = ReliabilityTracker(
        num_clients=args.num_users,
        window=H,
        alpha=alpha_R,
    )

    # Per-round reliability logs
    round_mean_R = []
    round_spearman_R_acc = []



    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        client_update_vecs = []  # flattened v_i for this round
        consensus_scores = []    # C_i for this round

        # for logging
        reliability_scores = []  # R_i for this round
        selected_client_ids = [] # client indices in this round
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()

        # w_t: global weights *before* this round's client updates
        prev_global_weights = copy.deepcopy(global_model.state_dict())

        # v̄_{t-1} flattened (for cosine with v_i)
        vbar_flat_prev = flatten_tensor_dict(vbar, keys=param_keys)
        vbar_norm = torch.norm(vbar_flat_prev)
        eps = 1e-12
        if vbar_norm > eps:
            vbar_flat_prev = vbar_flat_prev / (vbar_norm + eps)

        # hyperparameters for FedBaC consensus
        gamma = getattr(args, "gamma", 1.0)      # sharpness γ
        beta = getattr(args, "beta", 0.9)        # momentum β

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        reliability_scores = []  # R_i for this round

        for idx in idxs_users:
            local_model = LocalUpdate(
                args=args,
                dataset=train_dataset,
                idxs=user_groups[idx],
                logger=logger
            )
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model),
                global_round=epoch,
                vbar_flat=vbar_flat_prev,
                param_keys=param_keys
            )

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

            # --- FedBaC: client update v_i = w_i - w_t, then flatten ---
            v_i = {
                name: w[name] - prev_global_weights[name]
                for name in param_keys
            }
            v_i_flat = flatten_tensor_dict(v_i, keys=param_keys)
            client_update_vecs.append(v_i_flat)

            # --- Consensus score C_i = max(0, cos(v_i, v̄_{t-1}))^γ ---
            cos_for_reliab = None  # default: no cosine this round

            if vbar_norm > eps:
                vi_norm = torch.norm(v_i_flat)
                if vi_norm > eps:
                    cos_i = torch.dot(v_i_flat, vbar_flat_prev) / (
                        vi_norm * vbar_norm + eps
                    )
                    # Clamp numeric noise and apply ReLU + sharpness
                    cos_i = torch.clamp(cos_i, -1.0, 1.0)
                    c_i = torch.clamp(cos_i, min=0.0) ** gamma

                    # Keep raw cosine (before ReLU/gamma) for reliability
                    cos_for_reliab = float(cos_i.detach().cpu().item())
                else:
                    # Zero update: treat as neutral contributor
                    c_i = torch.tensor(1.0, device=device)
                # end if vi_norm
            else:
                # First round or nearly-zero momentum: fall back to FedAvg (C_i = 1)
                c_i = torch.tensor(1.0, device=device)

            consensus_scores.append(c_i)
            selected_client_ids.append(idx)

            # --- Reliability R_i from cosine-history variance ---
            reliability.update(idx, cos_for_reliab)
            R_i_val = reliability.get_reliability(idx)
            reliability_scores.append(torch.tensor(R_i_val, device=device, dtype=torch.float32))

        # Convert C_i and R_i to tensors
        C = torch.stack(consensus_scores)        # shape [m]
        R = torch.stack(reliability_scores)      # shape [m]

        # Combine into α_i ∝ C_i * R_i
        CR = C * R
        if torch.sum(CR) <= 0:
            # Extreme corner case: fall back to uniform FedAvg
            alpha = torch.ones_like(CR) / CR.numel()
            print("Warning: all consensus*reliability scores were zero!")
        else:
            alpha = CR / torch.sum(CR)

        # Log per-round mean R over the sampled clients
        round_mean_R.append(float(R.mean().item()))



        # --- Consensus-weighted aggregation ---
        # w_{t+1} = Σ_i α_i w_i   (equivalent to w_t + Σ_i α_i v_i if Σ α_i = 1)
        global_weights = copy.deepcopy(prev_global_weights)
        for k in global_weights.keys():
            agg_param = torch.zeros_like(prev_global_weights[k])
            for a_i, w_i in zip(alpha, local_weights):
                agg_param += a_i * w_i[k]
            global_weights[k] = agg_param

        global_model.load_state_dict(global_weights)

        # --- FedBaC Momentum Update v̄_t = β v̄_{t-1} + (1 − β) Δw_t ---
        delta_w_t = {
            name: global_weights[name] - prev_global_weights[name]
            for name in global_weights.keys()
        }
        for name in vbar.keys():
            vbar[name] = beta * vbar[name] + (1.0 - beta) * delta_w_t[name]

        # (Optional) keep flattened v̄_t for logging later if you like:
        # vbar_flat = flatten_tensor_dict(vbar, keys=param_keys)


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

        mean_acc = sum(list_acc) / len(list_acc)
        train_accuracy.append(mean_acc)
        rounds.append(epoch + 1)

        # --- Early stopping on client-held-out accuracy ---
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_round = epoch + 1
            best_global_weights = copy.deepcopy(global_weights)
            print(f"[BEST] New best mean acc: {best_acc:.4f} at round {best_round}")


        # --- Reliability–accuracy coupling: Spearman(R_i, acc_i) ---
        R_all = np.array(
            [reliability.get_reliability(c) for c in range(args.num_users)],
            dtype=np.float32,
        )
        acc_all = np.array(list_acc, dtype=np.float32)

        # Spearman via ranking + Pearson on ranks
        if np.all(acc_all == acc_all[0]) or np.all(R_all == R_all[0]):
            # Degenerate case: all equal → zero correlation
            corr_R_acc = 0.0
        else:
            # rank
            R_rank = np.argsort(np.argsort(R_all)).astype(np.float32)
            acc_rank = np.argsort(np.argsort(acc_all)).astype(np.float32)
            R_rank -= R_rank.mean()
            acc_rank -= acc_rank.mean()
            denom = (np.sqrt((R_rank ** 2).sum()) * np.sqrt((acc_rank ** 2).sum()) + 1e-12)
            corr_R_acc = float((R_rank * acc_rank).sum() / denom)

        round_spearman_R_acc.append(corr_R_acc)

        # print global training loss after every 'i' rounds
        if (epoch + 1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}%'.format(100 * train_accuracy[-1]))
            print(f'Mean reliability R: {round_mean_R[-1]:.4f}')
            print(f'Spearman(R, client acc): {round_spearman_R_acc[-1]:.4f}\n')

    # Load best model before final evaluation
    print(f"Loading best model from round {best_round} with mean client-held-out acc {best_acc:.4f}")
    global_model.load_state_dict(best_global_weights)

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    # --- Save artifacts for this run (research-paper style) ---
    # 1) Save the final global model
    save_model(out_dir, global_model)

    # 2) Save per-round metrics (loss + train accuracy)
    save_metrics(out_dir, rounds, train_loss, train_accuracy)
    save_loss_plot(out_dir, rounds, train_loss, train_accuracy)

    # 3) Save per-round reliability metrics
    save_reliability_metrics(out_dir, rounds, round_mean_R, round_spearman_R_acc)

    # 4) Save final summary numbers 
    save_final_results(
    out_dir,
    final_train_acc=best_acc,
    final_test_acc=test_acc,
    final_test_loss=test_loss,
    )


    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

        # PLOTTING (optional)
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss)
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    loss_path = os.path.join(out_dir, "train_loss.png")
    plt.savefig(loss_path)
    plt.close()

    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy)
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    acc_path = os.path.join(out_dir, "train_accuracy.png")
    plt.savefig(acc_path)
    plt.close()

