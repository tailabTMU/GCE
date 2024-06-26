import torch
import numpy as np
import os
from torch_geometric.datasets import DBLP
import torch_geometric.transforms as T
from models import HGT

from utils.gnn import train_heterogeneous_node_classifier as train, test_heterogeneous_node_classifier as test, \
    GraphCalibrationLoss
from sklearn.metrics import classification_report
import argparse
import random


def main(gamma):
    cwd = os.getcwd()
    path = os.path
    pjoin = os.path.join

    if path.exists(pjoin(cwd, 'saved_info_heterogeneous_node_classification')) is False:
        os.mkdir(pjoin(cwd, 'saved_info_heterogeneous_node_classification'))

    if path.exists(pjoin(cwd, 'saved_info_heterogeneous_node_classification', f'gcl')) is False:
        os.mkdir(pjoin(cwd, 'saved_info_heterogeneous_node_classification', f'gcl'))

    # We initialize conference node features with a single one-vector as feature:
    dataset = DBLP(root=os.path.join(os.getcwd(), 'data', 'DBLP'), transform=T.Constant(node_types='conference'))

    print()
    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {4}')

    data = dataset[0]  # Get the first graph object.

    print()
    print(data)
    print('===========================================================================================================')
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    print(f'Device: {device}')

    model_name = 'hetero_gnn'
    model = HGT(hidden_channels=64, out_channels=4, num_heads=2, num_layers=1, node_types=data.node_types,
                metadata=data.metadata())
    data, model = data.to(device), model.to(device)
    with torch.no_grad():  # Initialize lazy modules.
        out = model(data.x_dict, data.edge_index_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
    criterion = GraphCalibrationLoss(gamma=gamma)
    gamma_txt = str(gamma).replace('.', '_')
    # setting seeds
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)

    for epoch in range(1, 101):
        loss = train(model, data, criterion, optimizer)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    _, true_classes, predicted_classes, logit_vals = test(model, data, 'test_mask')

    print(classification_report(true_classes, predicted_classes))

    with open(pjoin(cwd, 'saved_info_heterogeneous_node_classification', f'gcl',
                    f'{model_name}_test_logit_vals_{gamma_txt}.txt'),
              'w+') as myfile:
        np.savetxt(myfile, logit_vals)

    with open(pjoin(cwd, 'saved_info_heterogeneous_node_classification', f'gcl',
                    f'{model_name}_test_true_classes_{gamma_txt}.txt'),
              'w+') as myfile:
        np.savetxt(myfile, true_classes)

    with open(pjoin(cwd, 'saved_info_heterogeneous_node_classification', f'gcl',
                    f'{model_name}_test_predicted_classes_{gamma_txt}.txt'),
              'w+') as myfile:
        np.savetxt(myfile, predicted_classes)

    torch.save(model.state_dict(),
               pjoin(cwd, 'saved_info_heterogeneous_node_classification', f'gcl',
                     f'{model_name}_model_{gamma_txt}.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a Heterogeneous Node Classifier with CrossEntropy Loss')
    for gamma in [0.047, 0.019, 0.013, 0.044, 0.022, 0.012, 0.045, 0.027, 0.017, 0.005, 0.001, 0.0001]:
        print('Gamma: ', gamma)
        main(gamma)
        print()
