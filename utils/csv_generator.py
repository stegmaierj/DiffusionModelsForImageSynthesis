# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 15:55:13 2020

@author: Nutzer
"""

import os
import glob
import csv
import numpy as np


def get_files(folders, data_root='', descriptor='', filetype='tif'):
    
    filelist = []
    
    for folder in folders:
        files = glob.glob(os.path.join(data_root, folder, '*'+descriptor+'*.'+filetype))
        filelist.extend([os.path.join(folder, os.path.split(f)[-1]) for f in files])
        
    return filelist
        
    
        
def read_csv(list_path, data_root=''):
    
    filelist = []
    
    with open(list_path, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            if len(row)==0 or np.sum([len(r) for r in row])==0: continue
            row = [os.path.join(data_root, r) for r in row]
            filelist.append(row)
            
    return filelist
   
    
        
def create_csv(data_list, save_path='list_folder/experiment_name', test_split=0.2, val_split=0.1, shuffle=False):
        
    if shuffle:
        np.random.shuffle(data_list)
    
    # Get number of files for each split
    num_files = len(data_list)
    num_test_files = int(test_split*num_files)
    num_val_files = int((num_files-num_test_files)*val_split)
    num_train_files = num_files - num_test_files - num_val_files
    
    # Adjust file identifier if there is no split
    if test_split>0 or val_split>0:
        train_identifier='_train.csv'
    else:
        train_identifier='.csv'
    
    # Get file indices
    file_idx = np.arange(num_files)
    
    # Save csv files
    if num_test_files > 0:
        test_idx = sorted(np.random.choice(file_idx, size=num_test_files, replace=False))
        with open(save_path+'_test.csv', 'w', newline='') as fh:
            writer = csv.writer(fh, delimiter=';')
            for idx in test_idx:
                writer.writerow(data_list[idx])
    else:
        test_idx = []
        
    if num_val_files > 0:
        val_idx = sorted(np.random.choice(list(set(file_idx)-set(test_idx)), size=num_val_files, replace=False))
        with open(save_path+'_val.csv', 'w', newline='') as fh:
            writer = csv.writer(fh, delimiter=';')
            for idx in val_idx:
                writer.writerow(data_list[idx])
    else:
        val_idx = []
    
    if num_train_files > 0:
        train_idx = sorted(list(set(file_idx) - set(test_idx) - set(val_idx)))
        with open(save_path+train_identifier, 'w', newline='') as fh:
            writer = csv.writer(fh, delimiter=';')
            for idx in train_idx:
                writer.writerow(data_list[idx])
            
            