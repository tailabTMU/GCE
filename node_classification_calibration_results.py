import os
import pandas as pd
import argparse
import statistics
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, \
    recall_score, precision_score, roc_auc_score, log_loss
import matplotlib.pyplot as plt
from torch.nn import functional as F
import torch
from utils.common import expected_calibration_error
from utils.calibration_metrics import BrierScore, SCELoss


def main():
    cwd = os.getcwd()
    path = os.path
    pjoin = path.join

    models = [
        'gat',
    ]
    model_names = {
        'gat': 'GAT',
        'graphconv': 'GraphConv',
    }

    num_bins = 15

    for dataset in ['PubMed', 'Cora']:
        dfs = {}
        final_sheet = []
        for model in models:
            for calibration_method in ['uncalibrated', 'gcl', 'brier_score']:
                results = []
                results_dir = pjoin(cwd, 'saved_info_node_classification', f'{calibration_method}_{dataset.lower()}')
                if path.exists(results_dir):
                    file_suffixes = ['']
                    if calibration_method != 'uncalibrated':
                        file_suffixes = []
                        for gamma in [0.047, 0.019, 0.013, 0.044, 0.022, 0.012, 0.045, 0.027, 0.017, 0.005,
                                      0.001,
                                      0.0001]:
                            gamma_str = str(gamma).replace('.', '_')
                            if calibration_method == 'ece_contrib':
                                gamma_str = f'{gamma_str}_{num_bins}_bins'
                            file_suffixes.append(f'_{gamma_str}')
                        if calibration_method == 'ece_contrib':
                            file_suffixes.append(f'_dynamic_gamma_{num_bins}_bins')
                        elif calibration_method == 'brier_score':
                            file_suffixes.append(f'_dynamic_gamma')
                    for file_suffix in file_suffixes:
                        actual_file = pjoin(results_dir,
                                            f'{model}_test_true_classes{file_suffix}.txt')
                        pred_file = pjoin(results_dir,
                                          f'{model}_test_predicted_classes{file_suffix}.txt')
                        scores_file = pjoin(results_dir,
                                            f'{model}_test_logit_vals{file_suffix}.txt')

                        y_true = np.loadtxt(actual_file, dtype="float").astype(int)
                        y_pred = np.loadtxt(pred_file, dtype="float").astype(int)
                        y_score = np.loadtxt(scores_file, dtype="float")
                        y_prob = F.softmax(torch.tensor(y_score), dim=-1).numpy()

                        one_hot_true_labels = list(F.one_hot(torch.LongTensor(list(y_true)),
                                                             num_classes=len(np.unique(y_true))).numpy())

                        info = file_suffix.strip('_')
                        if info != '':
                            info = info.replace(f'_{num_bins}_bins', '')
                            if info == 'dynamic_gamma':
                                info = 'Dynamic Gamma'
                            else:
                                info = info.replace('_', '.')
                                info = f'Gamma: {info}'
                        print(
                            f'Dataset: {dataset.upper()}, Model: {model_names[model]}, Calibration Method: {calibration_method.title().replace("_", "")}, Extra Info.: {info}')
                        print(
                            '====================================================================================')
                        print()

                        true_labels = y_true

                        brier_score = BrierScore()
                        brier_val = brier_score.loss(y_prob, true_labels)

                        res = {
                            'model': model_names[model],
                            'calibration_method': calibration_method.title().replace("_", ""),
                            'info': info,
                            'accuracy': accuracy_score(y_true, y_pred),
                            'precision': precision_score(y_true, y_pred, average='micro'),
                            'recall': recall_score(y_true, y_pred, average='micro'),
                            'f1': f1_score(y_true, y_pred, average='micro'),
                            'roc_auc': roc_auc_score(one_hot_true_labels, y_prob),
                        }

                        for bins in [20]:
                            sce_loss = SCELoss()
                            sce_val = sce_loss.loss(y_prob, true_labels, n_bins=bins, logits=False)
                            if y_prob.shape[1] == 2:
                                ece_val = expected_calibration_error(true_labels, y_prob[:, 1], bins)
                            else:
                                ece_val = expected_calibration_error(true_labels, y_prob, bins)

                            res[f'bin_{bins}_ece'] = ece_val
                            res[f'bin_{bins}_sce'] = sce_val

                        results.append(res)
                        df = pd.DataFrame(results)
                        dfs[f'{model}_{calibration_method}'] = df
                    methods = df['info'].unique().tolist()
                    for method in methods:
                        method_df = df[df['info'] == method]
                        method_res = {
                            'model': method_df['model'].iloc[0],
                            'calibration_method': method_df['calibration_method'].iloc[0],
                            'info': method,
                            'accuracy_mean': method_df['accuracy'].mean(),
                            'accuracy_std': method_df['accuracy'].std(),
                            'precision_mean': method_df['precision'].mean(),
                            'precision_std': method_df['precision'].std(),
                            'recall_mean': method_df['recall'].mean(),
                            'recall_std': method_df['recall'].std(),
                            'f1_mean': method_df['f1'].mean(),
                            'f1_std': method_df['f1'].std(),
                            'roc_auc_mean': method_df['roc_auc'].mean(),
                            'roc_auc_std': method_df['roc_auc'].std(),
                        }

                        for bins in [20]:
                            method_res[f'bin_{bins}_ece_mean'] = method_df[f'bin_{bins}_ece'].mean()
                            method_res[f'bin_{bins}_ece_std'] = method_df[f'bin_{bins}_ece'].std()
                        for bins in [20]:
                            method_res[f'bin_{bins}_sce_mean'] = method_df[f'bin_{bins}_sce'].mean()
                            method_res[f'bin_{bins}_sce_std'] = method_df[f'bin_{bins}_sce'].std()
                        final_sheet.append(method_res)

        dfs['aggregated_results'] = pd.DataFrame(final_sheet)
        with pd.ExcelWriter(pjoin(cwd, 'saved_info_node_classification', f'results_{dataset.lower()}.xlsx')) as writer:
            for sheet_name in dfs.keys():
                df = dfs[sheet_name]
                df.to_excel(writer, sheet_name=sheet_name.upper(), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preparing the Calibration Results')
    main()
