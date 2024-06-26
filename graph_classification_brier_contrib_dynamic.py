import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from models import SAGE
from utils.gnn import train, test, BrierScoreLoss
from utils.common import EarlyStopping
from sklearn.metrics import classification_report
import numpy as np
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
import argparse
from tqdm import tqdm
import random
import glob


def main(args):
    cwd = os.getcwd()
    path = os.path
    pjoin = os.path.join

    configs = {
        'proteins': {
            'net_params': {
                'L': 4,
                'hidden_dim': 88,
                'out_dim': 88,
                'residual': True,
                'readout': 'mean',
                'in_feat_dropout': 0.0,
                'dropout': 0.0,
                'batch_norm': True,
                'sage_aggregator': 'max'
            },
            "params": {
                "epochs": 1000,
                "batch_size": 20,
                "init_lr": 7e-5,
                "lr_reduce_factor": 0.5,
                "lr_schedule_patience": 25,
                "min_lr": 1e-6,
                "weight_decay": 0.0,
                "print_epoch_interval": 5,
            },
        },
        'enzymes': {
            "params": {
                "epochs": 1000,
                "batch_size": 20,
                "init_lr": 7e-4,
                "lr_reduce_factor": 0.5,
                "lr_schedule_patience": 25,
                "min_lr": 1e-6,
                "weight_decay": 0.0,
                "print_epoch_interval": 5,
            },
            'net_params': {
                'L': 4,
                'hidden_dim': 90,
                'out_dim': 90,
                'residual': True,
                'readout': 'mean',
                'in_feat_dropout': 0.0,
                'dropout': 0.0,
                'batch_norm': True,
                'sage_aggregator': 'max'
            }
        }
    }

    config = configs[args.dataset]['net_params']
    params = configs[args.dataset]['params']

    if path.exists(pjoin(cwd, 'saved_info_graph_classification')) is False:
        os.mkdir(pjoin(cwd, 'saved_info_graph_classification'))

    if path.exists(pjoin(cwd, 'saved_info_graph_classification', f'brier_score_{args.dataset}')) is False:
        os.mkdir(pjoin(cwd, 'saved_info_graph_classification', f'brier_score_{args.dataset}'))

    dataset = TUDataset(root=os.path.join(os.getcwd(), 'data', 'TUDataset'), name=args.dataset.upper())
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    print()
    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    print(f'Device: {device}')

    indices = np.array(list(range(0, len(dataset))))
    labels = np.array([dataset[i].y.detach().cpu().item() for i in indices])

    skf = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)

    split_number = 0
    avg_test_acc = []
    avg_train_acc = []
    avg_convergence_epochs = []
    gamma_txt = 'dynamic_gamma'

    for train_index, test_index in skf.split(indices, labels):
        # setting seeds
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(1)
        split_number += 1
        train_indices, test_indices = indices[train_index], indices[test_index]
        train_y, test_y = labels[train_index], labels[test_index]
        train_indices, val_indices, train_y, val_y = train_test_split(train_indices, train_y, test_size=0.25,
                                                                      random_state=1,
                                                                      stratify=train_y)

        train_dataset = dataset.index_select(train_indices)
        validation_dataset = dataset.index_select(val_indices)
        test_dataset = dataset.index_select(test_indices)

        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True,
                                  collate_fn=dataset.collate)
        val_loader = DataLoader(validation_dataset, batch_size=params['batch_size'], shuffle=False,
                                collate_fn=dataset.collate)
        test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False,
                                 collate_fn=dataset.collate)

        model_name = 'sage'
        model = SAGE(dataset.num_features, dataset.num_classes, config)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=params['lr_reduce_factor'],
                                                               patience=params['lr_schedule_patience'],
                                                               verbose=False)
        criterion = BrierScoreLoss(num_classes=dataset.num_classes, dynamic_gamma=True)

        ea = EarlyStopping(patience=10, verbose=False,
                           path=pjoin(cwd, 'saved_info_graph_classification', f'brier_score_{args.dataset}',
                                      f'{model_name}_split_{split_number}_model_{gamma_txt}.pt'))
        root_ckpt_dir = pjoin(cwd, 'saved_info_graph_classification', f'brier_score_{args.dataset}')

        last_epoch = 0

        print(f'Training Model for Split {split_number}')
        with tqdm(range(params['epochs'])) as t:
            for epoch in t:
                last_epoch = epoch
                t.set_description('Epoch %d' % epoch)

                epoch_train_acc, _, _, _, epoch_train_loss, optimizer = train(model, train_loader, device, criterion,
                                                                              optimizer)
                epoch_val_acc, _, _, _, epoch_val_loss = test(model, val_loader, device, criterion)

                t.set_postfix(lr=optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                              train_acc=epoch_train_acc, val_acc=epoch_val_acc,
                              )
                # Checking best model
                ea(epoch_val_loss, model, epoch)
                # if ea.early_stop:
                #     break

                # Saving checkpoint
                ckpt_dir = os.path.join(root_ckpt_dir, "RUN_" + str(split_number))
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                torch.save(model.state_dict(),
                           '{}.pkl'.format(pjoin(ckpt_dir, f"epoch_{gamma_txt}_{str(epoch)}")))

                files = glob.glob(pjoin(ckpt_dir, '*.pkl'))
                for file in files:
                    epoch_nb = file.split('_')[-1]
                    epoch_nb = int(epoch_nb.split('.')[0])
                    if epoch_nb < epoch - 1:
                        os.remove(file)

                scheduler.step(epoch_val_loss)

                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break

        # model = ea.load_checkpoint(model)
        test_acc, true_classes, predicted_classes, logit_vals, _ = test(model, test_loader, device, criterion)
        train_acc, _, _, _, _ = test(model, train_loader, device, criterion)
        avg_test_acc.append(test_acc)
        avg_train_acc.append(train_acc)
        # avg_convergence_epochs.append(ea.get_best_epoch())
        avg_convergence_epochs.append(last_epoch)

        print(classification_report(true_classes, predicted_classes))

        with open(pjoin(cwd, 'saved_info_graph_classification', f'brier_score_{args.dataset}',
                        f'{model_name}_split_{split_number}_test_logit_vals_{gamma_txt}.txt'),
                  'w+') as myfile:
            np.savetxt(myfile, logit_vals)

        with open(
                pjoin(cwd, 'saved_info_graph_classification', f'brier_score_{args.dataset}',
                      f'{model_name}_split_{split_number}_test_true_classes_{gamma_txt}.txt'),
                'w+') as myfile:
            np.savetxt(myfile, true_classes)

        with open(pjoin(cwd, 'saved_info_graph_classification', f'brier_score_{args.dataset}',
                        f'{model_name}_split_{split_number}_test_predicted_classes_{gamma_txt}.txt'),
                  'w+') as myfile:
            np.savetxt(myfile, predicted_classes)

    print("AVG CONVERGENCE Time (Epochs): {:.4f}".format(np.mean(np.array(avg_convergence_epochs))))
    # Final test accuracy value averaged over 10-fold
    print("""\n\n\nFINAL RESULTS\n\nTEST ACCURACY averaged: {:.4f} with s.d. {:.4f}""".format(
        np.mean(np.array(avg_test_acc)) * 100, np.std(avg_test_acc) * 100))
    print("\nAll splits Test Accuracies:\n", avg_test_acc)
    print("""\n\n\nFINAL RESULTS\n\nTRAIN ACCURACY averaged: {:.4f} with s.d. {:.4f}""".format(
        np.mean(np.array(avg_train_acc)) * 100, np.std(avg_train_acc) * 100))
    print("\nAll splits Train Accuracies:\n", avg_train_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training a Graph Classifier with Brier Score Loss as Gamma Value')
    parser.add_argument('--dataset', dest='dataset', type=str, default='proteins',
                        choices=['enzymes', 'proteins'],
                        help='Dataset name (default: proteins)')
    main(parser.parse_args())
