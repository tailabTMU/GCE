import torch
import numpy as np
import os
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from models import GATNodeClassifier as GAT

from utils.gnn import train_node_classifier as train, test_node_classifier as test
from sklearn.metrics import classification_report
import argparse
import random


def main(args):
    cwd = os.getcwd()
    path = os.path
    pjoin = os.path.join

    if path.exists(pjoin(cwd, 'saved_info_node_classification')) is False:
        os.mkdir(pjoin(cwd, 'saved_info_node_classification'))

    if path.exists(pjoin(cwd, 'saved_info_node_classification', f'uncalibrated_{args.dataset.lower()}')) is False:
        os.mkdir(pjoin(cwd, 'saved_info_node_classification', f'uncalibrated_{args.dataset.lower()}'))

    dataset = Planetoid(root=os.path.join(os.getcwd(), 'data', 'Planetoid'), name=args.dataset,
                        transform=NormalizeFeatures())

    print()
    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]  # Get the first graph object.

    print()
    print(data)
    print('===========================================================================================================')
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Number of training nodes: {data.train_mask.sum()}')
    print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')
    print(f'Device: {device}')

    model_name = args.model
    if model_name == 'gat':
        model = GAT(16, dataset.num_features, dataset.num_classes)
    else:
        raise Exception(f'Unknown model name: {model_name}')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, 101):
        loss = train(model, data, device, criterion, optimizer)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    _, true_classes, predicted_classes, logit_vals = test(model, data, device)

    print(classification_report(true_classes, predicted_classes))

    with open(pjoin(cwd, 'saved_info_node_classification', f'uncalibrated_{args.dataset.lower()}',
                    f'{model_name}_test_logit_vals.txt'),
              'w+') as myfile:
        np.savetxt(myfile, logit_vals)

    with open(pjoin(cwd, 'saved_info_node_classification', f'uncalibrated_{args.dataset.lower()}',
                    f'{model_name}_test_true_classes.txt'),
              'w+') as myfile:
        np.savetxt(myfile, true_classes)

    with open(pjoin(cwd, 'saved_info_node_classification', f'uncalibrated_{args.dataset.lower()}',
                    f'{model_name}_test_predicted_classes.txt'),
              'w+') as myfile:
        np.savetxt(myfile, predicted_classes)

    torch.save(model.state_dict(),
               pjoin(cwd, 'saved_info_node_classification', f'uncalibrated_{args.dataset.lower()}',
                     f'{model_name}_model.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a Node Classifier with CrossEntropy Loss')
    parser.add_argument('--dataset', dest='dataset', type=str, default='Cora',
                        choices=['Cora', 'PubMed'],
                        help='Dataset name (default: Cora)')
    parser.add_argument('--model', dest='model', type=str, default='gat',
                        choices=['gat'],
                        help='Model name (default: gat)')
    main(parser.parse_args())
