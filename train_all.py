
from argparse import Namespace
from logging import Logger
import os

import pandas as pd
from typing import Tuple
import numpy as np
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
from chemprop.train.run_training import run_training
from chemprop.data.utils import get_task_names, get_data
from chemprop.data.data import MoleculeDataset
from chemprop.utils import makedirs
from chemprop.parsing import parse_train_args
from chemprop.utils import create_logger
from chemprop.models import build_model

import random
import numpy as np
import torch
from torch.backends import cudnn
torch.cuda.empty_cache()

# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"
#     torch.backends.cudnn.enabled = False
#     cudnn.deterministic = True
#     cudnn.benchmark = False
#     torch.use_deterministic_algorithms(True)
#
# GLOBAL_SEED=1
# set_seed(GLOBAL_SEED)



def cross_validate(args: Namespace, logger: Logger = None) -> Tuple[np.ndarray, np.ndarray]:
    """k-fold cross validation"""
    info = logger.info if logger is not None else print
    args.metric="r2"

    info('Loading data')
    args.dataset_name = 'df_heavy_Fermion'
    data_heavy_Fermion = get_data(path=args.data_path, args=args, logger=logger)

    args.dataset_name = 'df_others'
    data_others = get_data(path=args.data_path, args=args, logger=logger)

    args.dataset_name = 'df_cuprate'
    data_cuprate = get_data(path=args.data_path, args=args, logger=logger)

    args.dataset_name = 'df_iron'
    data_iron = get_data(path=args.data_path, args=args, logger=logger)

    args.dataset_name = 'df_h'
    data_h = get_data(path=args.data_path, args=args, logger=logger)

    args.dataset_name = 'df_all_data'
    data = get_data(path=args.data_path, args=args, logger=logger)

    args.task_names = get_task_names(os.path.join(args.data_path, f'{args.dataset_name}.csv'))
    args.num_tasks = data.num_tasks()
    args.features_size = data.features_size()

    info(f'Number of tasks = {args.num_tasks},{args.task_names}')
    # Split data
    info(f'Splitting data with seed {args.seed}')

   # Run training on different random seeds for each fold
    all_validation_scores, all_test_scores = list(), list()
    seed = 4
    train_val_data_heavy_Fermion, test_data_heavy_Fermion = train_test_split(data_heavy_Fermion, test_size=int(len(data_heavy_Fermion) * 0.1), random_state=seed)
    train_data_heavy_Fermion, val_data_heavy_Fermion = train_test_split(train_val_data_heavy_Fermion, test_size=int(len(train_val_data_heavy_Fermion) * 0.1), random_state=seed)


    train_val_data_others, test_data_others = train_test_split(data_others, test_size=int(len(data_others) * 0.1), random_state=seed)
    train_data_others, val_data_others = train_test_split(train_val_data_others, test_size=int(len(train_val_data_others) * 0.1), random_state=seed)

    train_val_data_cuprate, test_data_cuprate = train_test_split(data_cuprate, test_size=int(len(data_cuprate) * 0.1), random_state=seed)
    train_data_cuprate, val_data_cuprate = train_test_split(train_val_data_cuprate, test_size=int(len(train_val_data_cuprate) * 0.1), random_state=seed)

    train_val_data_iron, test_data_iron = train_test_split(data_iron, test_size=int(len(data_iron) * 0.1), random_state=seed)
    train_data_iron, val_data_iron = train_test_split(train_val_data_iron, test_size=int(len(train_val_data_iron) * 0.1), random_state=seed)

    # train_val_data_insulator, test_data_insulator = train_test_split(data_insulator, test_size=int(len(data_insulator) * 0.1), random_state=seed)
    # train_data_insulator, val_data_insulator = train_test_split(train_val_data_insulator, test_size=int(len(train_val_data_insulator) * 0.1), random_state=seed)

    train_val_data_h, test_data_h = train_test_split(data_h, test_size=int(len(data_h) * 0.1), random_state=seed)
    train_data_h, val_data_h = train_test_split(train_val_data_h, test_size=int(len(train_val_data_h) * 0.1), random_state=seed)

    info('=' * 20 + f' fold {seed} ' + '=' * 20)
    # train_data, val_data = train_test_split(data, test_size=int(len(data) * 0.368), random_state=args.seed)
    train_data =  list(train_data_heavy_Fermion) + list(train_data_iron) + list(train_data_cuprate) +list(train_data_others) +list(train_data_h)
    val_data =  list(val_data_heavy_Fermion) + list(val_data_iron) +list(val_data_cuprate) +list(val_data_others) +list(val_data_h)
    test_data = list(test_data_heavy_Fermion) + list(test_data_iron) +list(test_data_cuprate) +list(test_data_others) +list(test_data_h)
    # Random Sample Validation data


    train_data, val_data, test_data, all_data = MoleculeDataset(train_data), MoleculeDataset(val_data), MoleculeDataset(test_data), MoleculeDataset(data)

    #print(train _data[1])
    #print(data[1])
    # Required for NormLR
    args.train_data_size = len(train_data)

    info(f'Total size = {len(data):,} | '
         f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')
    # Training
    save_dir = os.path.join(args.save_dir, f'fold_{seed}')
    makedirs(save_dir)

    model_validation_scores, model_test_scores = run_training(crystal,train_data, val_data, test_data, all_data, seed, args, logger)
    all_validation_scores.append(model_validation_scores)
    all_test_scores.append(model_test_scores)

    all_validation_scores, all_test_scores = np.array(all_validation_scores), np.array(all_test_scores)

    # Report results
    info(f'{args.num_folds}-fold cross validation')

    # Report scores for each fold
    info(f'Seed {args.seed}')
    for seed, (valid_scores, test_scores) in enumerate(zip(all_validation_scores, all_test_scores)):
        info(f'Fold {seed} ==> '
             f'valid {args.metric} = {np.nanmean(valid_scores):.6f} '
             f'test {args.metric} = {np.nanmean(test_scores):.6f}')

        if args.show_individual_scores:
            for task_name, valid_score, test_score in zip(args.task_names, valid_scores, test_scores):
                info(f'Fold {seed} ==> '
                     f'valid {task_name} {args.metric} = {valid_score:.6f}'
                     f'test {task_name} {args.metric} = {test_score:.6f}')

    # Report scores across models
    avg_valid_scores = np.nanmean(all_validation_scores, axis=1)  # average score for each model across tasks
    mean_valid_score, std_valid_score = np.nanmean(avg_valid_scores), np.nanstd(avg_valid_scores)
    info(f'Overall valid {args.metric} = {mean_valid_score:.6f} +/- {std_valid_score:.6f}')

    # Report scores across models
    avg_test_scores = np.nanmean(all_test_scores, axis=1)  # average score for each model across tasks
    mean_test_score, std_test_score = np.nanmean(avg_test_scores), np.nanstd(avg_test_scores)
    info(f'Overall test {args.metric} = {mean_test_score:.6f} +/- {std_test_score:.6f}')

    if args.show_individual_scores:
        for task_num, task_name in enumerate(args.task_names):
            info(f'Overall test {task_name} {args.metric} = '
                 f'{np.nanmean(all_test_scores[:, task_num]):.6f} +/- {np.nanstd(all_test_scores[:, task_num]):.6f}')

    return mean_test_score, std_test_score


if __name__ == '__main__':
    crystal = pd.read_csv(r'D:\ML/df_all_data0424.csv', index_col=0)
    args = parse_train_args()
    logger = create_logger(name='train', save_dir=args.save_dir, quiet=args.quiet)
    mean_auc_score, std_auc_score = cross_validate(args, logger)
    print(f'Results: {mean_auc_score:.5f} +/- {std_auc_score:.5f}')
