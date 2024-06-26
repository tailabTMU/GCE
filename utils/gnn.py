import torch
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch


class GraphCalibrationLoss(nn.Module):
    def __init__(self, gamma=0.047):
        super(GraphCalibrationLoss, self).__init__()
        """
        Args:
            gamma (float): Gamma tunable parameter.
                            Default: 0.047
        """
        self.gamma = gamma

    def forward(self, x, y):
        log_prob = -1.0 * (1 + (self.gamma * F.softmax(x, 1))) * F.log_softmax(x, 1)
        loss = log_prob.gather(1, y.unsqueeze(1))
        loss = loss.mean()
        return loss


class BrierScoreLoss(nn.Module):
    def __init__(self, num_classes: int, gamma=0.047, dynamic_gamma=False):
        super(BrierScoreLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.dynamic_gamma = dynamic_gamma

    def __brier_score(self, predicted_probabs, true_labels):
        true_labels = F.one_hot(true_labels, self.num_classes)
        true_labels = true_labels.float()
        squared_diff = torch.square(true_labels - predicted_probabs)
        return torch.sum(squared_diff, dim=1).unsqueeze(1)

    def forward(self, x, y):
        if self.dynamic_gamma:
            brier_scores = self.__brier_score(F.softmax(x, 1), y)
            log_prob = -1.0 * (1 + (brier_scores * F.softmax(x, 1))) * F.log_softmax(x, 1)
            loss = log_prob.gather(1, y.unsqueeze(1))
            loss = loss.mean()
            return loss
        else:
            brier_scores = self.__brier_score(F.softmax(x, 1), y)
            log_prob = -1.0 * (1 + (self.gamma * brier_scores)) * F.log_softmax(x, 1)
            loss = log_prob.gather(1, y.unsqueeze(1))
            loss = loss.mean()
            return loss


class BinCalibrationContributionLoss(nn.Module):
    def __init__(self, gamma=0.047, num_bins=3, dynamic_gamma=False):
        super(BinCalibrationContributionLoss, self).__init__()
        """
        Args:
            gamma (float): Gamma tunable parameter.
                            Default: 0.047
            num_bins (int): Number of bins.
                            Default: 3
            dynamic_gamma (boolean): If the contribution should be used as Gamaa.
                            Default: False
        """
        self.gamma = gamma
        self.num_bins = num_bins
        bin_boundaries = torch.linspace(0, 1, self.num_bins + 1)
        # uniform binning approach with self.num_bins number of bins
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        self.dynamic_gamma = dynamic_gamma

    def __bin_calibration_error_contribution(self, predicted_probabs, true_labels):
        device = predicted_probabs.device

        confidences, predicted_label = torch.max(predicted_probabs, 1)
        confidences = confidences.unsqueeze(1)
        predicted_label = predicted_label.unsqueeze(1)

        accuracies = predicted_label == true_labels.unsqueeze(1)
        accuracies = accuracies.to(device)

        original_contributions = torch.full(confidences.shape, 0).float()
        original_contributions = original_contributions.to(device)
        updated_contributions = torch.full(confidences.shape, 0).float()
        updated_contributions = updated_contributions.to(device)

        # TODO: Check and fix the possibility of having some bins as empty!
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # determine if sample is in bin m (between bin lower & upper)
            in_bin = torch.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
            in_bin = in_bin.to(device)
            # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
            prop_in_bin = torch.mean(in_bin.float())
            prop_in_bin = prop_in_bin.to(device)

            if prop_in_bin.item() > 0:
                # get the accuracy of bin m: acc(Bm)
                accuracy_in_bin = torch.mean(accuracies[in_bin.squeeze(1)].float())
                # get the average confidence of bin m: conf(Bm)
                avg_confidence_in_bin = torch.mean(confidences[in_bin.squeeze(1)])
                # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m
                bin_cal_err = torch.abs(torch.sub(avg_confidence_in_bin, accuracy_in_bin))
                index = (in_bin == True).nonzero(as_tuple=True)[0]
                original_contributions[index] = bin_cal_err
                for i in index:
                    new_in_bin = torch.clone(in_bin)
                    new_in_bin = new_in_bin.to(device)
                    new_in_bin[i] = False
                    # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
                    prop_in_bin = torch.mean(new_in_bin.float())
                    prop_in_bin = prop_in_bin.to(device)

                    if prop_in_bin.item() > 0:
                        # get the accuracy of bin m: acc(Bm)
                        accuracy_in_bin = torch.mean(accuracies[new_in_bin.squeeze(1)].float())
                        # get the average confidence of bin m: conf(Bm)
                        avg_confidence_in_bin = torch.mean(confidences[new_in_bin.squeeze(1)])
                        # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n)
                        bin_cal_err = torch.abs(torch.sub(avg_confidence_in_bin, accuracy_in_bin))
                        updated_contributions[i] = bin_cal_err

        # return torch.abs(torch.sub(original_contributions, updated_contributions))
        return torch.sub(original_contributions, updated_contributions)

    def forward(self, x, y):
        # TODO: Think of an approach that does not require binning!
        if self.dynamic_gamma:
            ece_contribution = self.__bin_calibration_error_contribution(F.softmax(x, 1), y)
            log_prob = -1.0 * (1 + (ece_contribution * F.softmax(x, 1))) * F.log_softmax(x, 1)
            loss = log_prob.gather(1, y.unsqueeze(1))
            loss = loss.mean()
            return loss
        else:
            ece_contribution = self.__bin_calibration_error_contribution(F.softmax(x, 1), y)
            log_prob = -1.0 * (1 + (self.gamma * ece_contribution)) * F.log_softmax(x, 1)
            loss = log_prob.gather(1, y.unsqueeze(1))
            loss = loss.mean()
            return loss


def check_patience(all_losses, best_loss, best_epoch, curr_loss, curr_epoch, counter):
    if curr_loss < best_loss:
        counter = 0
        best_loss = curr_loss
        best_epoch = curr_epoch
    else:
        counter += 1
    return best_loss, best_epoch, counter


def train(model, train_loader, device, criterion, optimizer):
    model.train()

    true_classes = []
    predicted_classes = []
    logit_vals = []
    loss_all = 0
    iteration_count = 0
    for data in train_loader:  # Iterate in batches over the training dataset.
        iteration_count += 1
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

        pred = out.detach().argmax(dim=1)  # Use the class with highest probability.
        predicted_classes += list(pred.cpu().numpy())
        true_classes += list(data.y.cpu().numpy())
        logit_vals += list(out.cpu().detach().numpy())  # Logit values.
        loss_all += loss.detach().item()

    accuracy = accuracy_score(true_classes, predicted_classes)
    loss_val = loss_all / iteration_count

    return accuracy, true_classes, predicted_classes, logit_vals, loss_val, optimizer


def test(model, loader, device, criterion):
    model.eval()

    true_classes = []
    predicted_classes = []
    logit_vals = []
    loss_all = 0
    iteration_count = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        iteration_count += 1
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)

        pred = out.detach().argmax(dim=1)  # Use the class with highest probability.
        predicted_classes += list(pred.cpu().numpy())
        true_classes += list(data.y.cpu().numpy())
        logit_vals += list(out.cpu().detach().numpy())  # Logit values.
        loss = criterion(out, data.y)
        loss_all += loss.item() * data.num_graphs

    accuracy = accuracy_score(true_classes, predicted_classes)
    loss_val = loss_all / iteration_count

    return accuracy, true_classes, predicted_classes, logit_vals, loss_val


def train_node_classifier(model, data, device, criterion, optimizer):
    model.train()
    data = data.to(device)
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask],
                     data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss


def test_node_classifier(model, data, device):
    model.eval()

    data = data.to(device)
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
    predicted_classes = list(pred[data.test_mask].cpu().numpy())
    true_classes = list(data.y[data.test_mask].cpu().numpy())
    logit_vals = list(out[data.test_mask].cpu().detach().numpy())  # Logit values.
    return test_acc, true_classes, predicted_classes, logit_vals

def train_heterogeneous_node_classifier(model, data, criterion, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    mask = data['author'].train_mask
    loss = criterion(out[mask], data['author'].y[mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test_heterogeneous_node_classifier(model, data, split):
    model.eval()

    out = model(data.x_dict, data.edge_index_dict)
    mask = data['author'][split]
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[mask] == data['author'].y[mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(mask.sum())  # Derive ratio of correct predictions.
    predicted_classes = list(pred[mask].cpu().numpy())
    true_classes = list(data['author'].y[mask].cpu().numpy())
    logit_vals = list(out[mask].cpu().detach().numpy())  # Logit values.
    return test_acc, true_classes, predicted_classes, logit_vals