#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code is implemented for the GACDN in "Hyperspectral Change Detection Using Semi-Supervised Graph Neural Network
and Convex Deep Learning"

It is constructed based on the code for "Weighted Feature Fusion of Convolutional Neural Network 
and Graph Attention Network for Hyperspectral Image Classification" 
Source: https://github.com/raglandman/WFCG

@author: Tzu-Hsuan Lin
"""

import argparse
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
import torch
import time
import yaml
import scipy.io as sio

from models import model, utils
from loadData import split_data
from createGraph import rdSLIC, create_graph

def load_config(path):
    return yaml.load(open(path, "r"), Loader=yaml.FullLoader)

def load_data(args):
    if args.data == 'river':
        x1 = sio.loadmat(f'{args.dataset_path}/zuixin/river_before.mat')["river_before"].astype(np.float32)
        x2 = sio.loadmat(f'{args.dataset_path}/zuixin/river_after.mat')["river_after"].astype(np.float32)
        data_gt = sio.loadmat(f'{args.dataset_path}/zuixin/groundtruth.mat')["lakelabel_v1"].astype(np.float32)
        data_gt = data_gt / data_gt.max()
        data_gt = data_gt.astype(int)
        data_gt = data_gt + 1
        data = np.concatenate((x1, x2), axis=2)
        height, width, bands = data.shape
        gt_reshape = np.reshape(data_gt, [-1])
        class_num = data_gt.max()
        return x1, x2, data_gt, data, height, width, bands, gt_reshape, class_num
    else:
        raise ValueError("Unsupported data type")

def main():
    parser = argparse.ArgumentParser(description='GACDN')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--path-config', type=str, default='./config/config.yaml')
    parser.add_argument('-pc', '--print-config', action='store_true', default=False)
    parser.add_argument('-sr', '--show-results', action='store_true', default=False)
    parser.add_argument('--save-results', action='store_true', default=True)
    parser.add_argument('--dataset-path', type=str, default='./Dataset')
    parser.add_argument('--data', type=str, default='river')
    args = parser.parse_args()  # running in command line

    start_time = time.time()

    config = load_config(args.path_config)
    ratio = config["ratio"]
    superpixel_scale = config["superpixel_scale"]
    max_epoch = config["max_epoch"]
    learning_rate = config["learning_rate"]
    weight_decay = config["weight_decay"]
    path_weight = config["path_weight"]
    args.data = config["data_name"]

    x1, x2, data_gt, data, height, width, bands, gt_reshape, class_num = load_data(args)

    train_ratio = ratio * 0.7
    val_ratio = ratio * 0.3

    if args.print_config:
        print(config)       
    
    
    # split datasets
    train_index, val_index, test_index, all_training_index = split_data.split_data(gt_reshape, 
                    train_ratio, val_ratio, class_num)
    
    sio.savemat('./temp files/index.mat', {"index":all_training_index+1})
    
    # create graph
    train_samples_gt, test_samples_gt, val_samples_gt = create_graph.get_label(gt_reshape,
                                                     train_index, val_index, test_index)
    
    train_label_mask, test_label_mask, val_label_mask = create_graph.get_label_mask(train_samples_gt, 
                                                test_samples_gt, val_samples_gt, data_gt, class_num)
    
    # label transfer to one-hot encode
    train_gt = np.reshape(train_samples_gt,[height,width])
    test_gt = np.reshape(test_samples_gt,[height,width])
    val_gt = np.reshape(val_samples_gt,[height,width])
    
    train_gt_onehot = create_graph.label_to_one_hot(train_gt, class_num)
    test_gt_onehot = create_graph.label_to_one_hot(test_gt, class_num)
    val_gt_onehot = create_graph.label_to_one_hot(val_gt, class_num)
    
    # superpixels
    ls = rdSLIC.LDA_SLIC(data, train_gt+val_gt, class_num-1)
    tic0=time.time()
    Q, S ,A, Seg= ls.simple_superpixel(scale=superpixel_scale)
    toc0 = time.time()
    LDA_SLIC_Time=toc0-tic0
    
    
    Q=torch.from_numpy(Q).to(args.device)
    A=torch.from_numpy(A).to(args.device)
    
    train_samples_gt=torch.from_numpy(train_samples_gt.astype(np.float32)).to(args.device)
    test_samples_gt=torch.from_numpy(test_samples_gt.astype(np.float32)).to(args.device)
    val_samples_gt=torch.from_numpy(val_samples_gt.astype(np.float32)).to(args.device)
    
    train_gt_onehot = torch.from_numpy(train_gt_onehot.astype(np.float32)).to(args.device)
    test_gt_onehot = torch.from_numpy(test_gt_onehot.astype(np.float32)).to(args.device)
    val_gt_onehot = torch.from_numpy(val_gt_onehot.astype(np.float32)).to(args.device)
    
    train_label_mask = torch.from_numpy(train_label_mask.astype(np.float32)).to(args.device)
    test_label_mask = torch.from_numpy(test_label_mask.astype(np.float32)).to(args.device)
    val_label_mask = torch.from_numpy(val_label_mask.astype(np.float32)).to(args.device)
    
    x1=np.array(x1, np.float32)
    x1=torch.from_numpy(x1.astype(np.float32)).to(args.device)
    
    x2=np.array(x2, np.float32)
    x2=torch.from_numpy(x2.astype(np.float32)).to(args.device)
    
    
    # model
    net = model.Net(int(bands/2), int(class_num), Q, A).to(args.device) 
    
    # train
    print("\n\n==================== training GACDN====================\n")
    optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate, weight_decay=weight_decay) #, weight_decay=0.0001
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    zeros = torch.zeros([height * width]).to(args.device).float()
    best_loss=99999
    net.train()
    tic1 = time.time()
    for i in range(max_epoch+1):
        optimizer.zero_grad()  # zero the gradient buffers
        output= net(x1, x2)
        loss = utils.compute_loss(output, train_gt_onehot, train_label_mask)
        loss.backward(retain_graph=False)
        optimizer.step()  # Does the update
        
        if i%10==0:
            with torch.no_grad():
                net.eval()
                output= net(x1, x2)
                trainloss = utils.compute_loss(output, train_gt_onehot, train_label_mask)
                trainOA = utils.evaluate_performance(output, train_samples_gt, train_gt_onehot, zeros)
                valloss = utils.compute_loss(output, val_gt_onehot, val_label_mask)
                valOA = utils.evaluate_performance(output, val_samples_gt, val_gt_onehot, zeros)
    
                if valloss < best_loss :
                    best_loss = valloss
                    torch.save(net.state_dict(), path_weight + r"model.pt")
                    print('save model...')
        scheduler.step()
        torch.cuda.empty_cache()
        net.train()
    
        if i%10==0:
            print("{}\ttrain loss={:.4f}\t train OA={:.4f} val loss={:.4f}\t val OA={:.4f}".format(str(i + 1), trainloss, trainOA, valloss, valOA))
    toc1 = time.time()
    
    print("\n\n====================training for GACDN done. starting evaluation...========================\n")
    
    paranum = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('number of parameter=', paranum)
    
    # test
    torch.cuda.empty_cache()
    with torch.no_grad():
        net.load_state_dict(torch.load(path_weight + r"model.pt"))
        net.eval()
        tic2 = time.time()
        output = net(x1, x2)
        toc2 = time.time()
        testloss = utils.compute_loss(output, test_gt_onehot, test_label_mask)
        testOA = utils.evaluate_performance(output, test_samples_gt, test_gt_onehot, zeros)
        print("{}\ttest loss={:.4f}\t test OA={:.4f}".format(str(i + 1), testloss, testOA))
    
    torch.cuda.empty_cache()
    
    LDA_SLIC_Time=toc0-tic0
    training_time = toc1 - tic1 + LDA_SLIC_Time
    testing_time = toc2 - tic2 + LDA_SLIC_Time
    training_time, testing_time
    
    # detection report
    test_label_mask_cpu = test_label_mask.cpu().numpy()[:,0].astype('bool')
    test_samples_gt_cpu = test_samples_gt.cpu().numpy().astype('int64')
    predict = torch.argmax(output, 1).cpu().numpy()
    classification = classification_report(test_samples_gt_cpu[test_label_mask_cpu], 
                                        predict[test_label_mask_cpu]+1, digits=4)
    kappa = cohen_kappa_score(test_samples_gt_cpu[test_label_mask_cpu], predict[test_label_mask_cpu]+1)
    
    sio.savemat('./temp files/Cdl.mat', {"Cdl":predict})
    
    end_time = time.time()
    
    all_time = end_time - start_time
    
    print("kappa ={:.4f}".format(kappa))
    print("time ={:.4f}s".format(all_time))
    
    if args.show_results:
            print(classification, kappa)

if __name__ == '__main__':
    main()


