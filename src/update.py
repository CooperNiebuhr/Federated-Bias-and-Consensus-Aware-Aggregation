#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from utils import flatten_tensor_dict


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round, vbar_flat=None, param_keys=None):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        # Regularizer strength; default 0.0 if not set
        lambda_reg = getattr(self.args, "lambda_reg", 0.0)
        eps = 1e-12

        # Prepare normalized server momentum direction v̂ if provided
        v_hat = None
        if vbar_flat is not None:
            vbar_flat = vbar_flat.to(self.device)
            v_norm = torch.norm(vbar_flat)
            if v_norm > eps:
                v_hat = vbar_flat / (v_norm + eps)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                task_loss = self.criterion(log_probs, labels)

                reg_loss = torch.tensor(0.0, device=self.device)

                # Only apply regularizer if we have a non-zero momentum direction and λ>0
                if (v_hat is not None) and (lambda_reg > 0.0) and (param_keys is not None):
                    # First-order gradient of the task loss
                    grads = torch.autograd.grad(
                        task_loss,
                        model.parameters(),
                        retain_graph=True,
                        create_graph=True,   # enable 2nd-order term
                    )

                    # Map grads to a dict keyed by parameter names
                    grad_dict = {}
                    for (name, _), g in zip(model.named_parameters(), grads):
                        grad_dict[name] = g

                    # Flatten in the same order as the server uses
                    g_flat = flatten_tensor_dict(grad_dict, param_keys).to(self.device)
                    g_norm = torch.norm(g_flat)

                    if g_norm < eps:
                        reg_loss = torch.tensor(0.0, device=self.device)

                    else:
                        g_hat = g_flat / (g_norm + eps)
                        cos_sim = torch.clamp(torch.dot(g_hat, v_hat), -1.0, 1.0)
                        reg_loss = lambda_reg * (1.0 - cos_sim)

                total_loss = task_loss + reg_loss
                
                if (iter == 0) and (batch_idx == 0):
                    print(f"[Diag] task_loss={task_loss.item():.6f}, "
                        f"reg_loss={reg_loss.item():.6f}, "
                        f"lambda={lambda_reg}")

                total_loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f} (task: {:.6f}, reg: {:.6f})'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader),
                        total_loss.item(), task_loss.item(), reg_loss.item()))
                self.logger.add_scalar('loss', total_loss.item())
                batch_loss.append(total_loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)


    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss
