#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

# using: 
# python .\src\federated_main.py --dataset cifar --model cnn --iid 0 --dirichlet 1 --alpha 0.5 --partition_seed 123 --sample_seed 456 --seed 999 --num_users=50 --local_ep=5 --epochs=30

def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments 
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    
    # FedBaC / consensus & reliability hyperparams
    parser.add_argument('--lambda_reg', type=float, default=0.0,
                        help='regularization strength for client-side consensus regularizer')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='consensus sharpness exponent γ for FedBaC')
    parser.add_argument('--beta', type=float, default=0.9,
                        help='server momentum coefficient β for FedBaC')
    parser.add_argument('--reliab_window', type=int, default=5,
                        help='window size H for reliability variance estimate')
    parser.add_argument('--reliab_alpha', type=float, default=1.0,
                        help='scale α for reliability exp(-α Var)')
    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    # device arguments
    parser.add_argument('--gpu', action='store_true',
                        help="Use CUDA if available.")
    parser.add_argument('--gpu_id', type=int, default=0,
                        help="CUDA device id (default: 0).")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--partition_seed', type=int, default=0,
                    help="Seed controlling the non-IID data split")
    # partition / sampling arguments
    parser.add_argument('--dirichlet', type=int, default=0,
                        help='Use Dirichlet label-skew partition when non-IID (1=yes, 0=no).')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Dirichlet concentration parameter (smaller => more non-IID).')
    parser.add_argument('--min_client_size', type=int, default=10,
                        help='Minimum samples per client for Dirichlet partition (resamples until satisfied).')
    parser.add_argument('--sample_seed', type=int, default=0,
                        help="Seed controlling client sampling each round (np.random.choice).")

    args = parser.parse_args()
    return args
