import sys
import numpy as np
import os
import pandas as pd
import torch
from models import DiffVectorModel, LogisticRegressionModel, FCNModel



def run_diffvectormodel(device, train_data, test_data, fold_idx): 

    print('running model 1...')
    
    model = DiffVectorModel(device, data_dir, res_dir, fold_idx)
    model.model_type = '1'
    model.train_data, model.test_data = train_data, test_data

    print('\nstarting training loop')
    model.train()

    print('\nstarting testing loop')
    datasets = [model.train_data, model.test_data]
    data_modes = ['train', 'test']
    for dataset, data_mode in zip(datasets, data_modes):
        print(f'\ntesting on: {data_mode} data')
        model.dataset, model.data_mode = dataset, data_mode
        model.test(stride, overlap_thres)
        
        print('metrics: ')
        print(f'accuracy = {model.acc}')
        print(f'precision = {model.prc}')
        print(f'recall = {model.rcl}')
        print(f'f1score = {model.f1s}')


def run_logisticregressionmodel(device, train_data, test_data, fold_idx): 
    print('running model 2...')

    model = LogisticRegressionModel(device, data_dir, res_dir, fold_idx)
    model.model_type = '2'
    model.train_data, model.test_data = train_data, test_data
    
    print('starting training loop')
    model.train()

    print('starting testing loop')
    datasets = [model.train_data, model.test_data]
    data_modes = ['train', 'test']
    for dataset, data_mode in zip(datasets, data_modes):
        print(f'testing on: {data_mode} data')
        model.dataset, model.data_mode = dataset, data_mode
        model.test(stride, overlap_thres)

        print('metrics: ')
        print(f'accuracy = {model.acc}')
        print(f'precision = {model.prc}')
        print(f'recall = {model.rcl}')
        print(f'f1score = {model.f1s}')


def run_fcnmodel(device, train_data, test_data, train_csv_path, fold_idx): 
    print('running model 3...')

    model = FCNModel(device, data_dir, res_dir, fold_idx)
    model.model_type = '3'

    print('starting training loop')
    model.load_data(train_csv_path, 'train', 'patch')
    model.train()

    print('starting testing loop: training data')
    datasets = [train_data, test_data]
    data_modes = ['train', 'test']
    for dataset, data_mode in zip(datasets, data_modes): 
        print(f'testing on: {data_mode} data')
        model.dataset, model.data_mode = dataset, data_mode
        model.test(stride, overlap_thres)
        
        print('metrics: ')
        print(f'accuracy = {model.acc}')
        print(f'precision = {model.prc}')
        print(f'recall = {model.rcl}')
        print(f'f1score = {model.f1s}')


def main(): 

    CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if CUDA else 'cpu')
    print('running on: ', device)

    fold_idx = 0
    train_csv_path = os.path.join(csv_dir, f'labels_fold{fold_idx+1}_train.csv')
    test_csv_path = os.path.join(csv_dir, f'labels_fold{fold_idx+1}_test.csv')

    train_data = pd.read_csv(train_csv_path)
    test_data = pd.read_csv(test_csv_path)

    if model_type == '1':
        run_diffvectormodel(device, train_data, test_data, fold_idx)
    elif model_type == '2': 
        run_logisticregressionmodel(device, train_data, test_data, fold_idx)
    elif model_type == '3': 
        run_fcnmodel(device, train_data, test_data, train_csv_path, fold_idx)


if __name__ == '__main__': 

    try: 
        model_type, ds = sys.argv[1], sys.argv[2]
    except: 
        sys.exit('provide: (1) model type {1,2,3} and (2) ds {ds1, ds2}')

    stride = 28
    overlap_thres = 0.2

    root_dir = '../'
    csv_dir = os.path.join(root_dir, f'data/patches/{ds}')
    data_dir = os.path.join(root_dir, f'data/patches/{ds}/x')
    res_dir = os.path.join(root_dir, f'results/{ds}/model_{model_type}')

    main()