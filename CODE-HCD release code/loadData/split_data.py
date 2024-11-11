import numpy as np
import random
import math

def split_data(gt_reshape, train_ratio, val_ratio, class_num):
    train_index = []
    test_index = []
    val_index = []
    index = []
    
  
    idx = np.where(np.logical_and(0<gt_reshape, gt_reshape < 3))[-1]
    sample_count = len(idx)
    train_pos_num = math.ceil(sample_count * (train_ratio)*1/3)
    train_neg_num = math.ceil(sample_count * (train_ratio)*2/3)
    train_num = train_pos_num+train_neg_num
    val_pos_num = math.ceil(sample_count * (val_ratio)*1/3)
    val_neg_num = math.ceil(sample_count * (val_ratio)*2/3)
    val_num = val_pos_num+val_neg_num
    neg_inds = np.where(gt_reshape==1)[-1]
    pos_inds = np.where(gt_reshape==2)[-1]
    
    pos_samplesCount = len(pos_inds)
    pos_rand_list = [i for i in range(pos_samplesCount)]
    pos_rand_idx = random.sample(pos_rand_list, (train_pos_num+val_pos_num))
    train_rand_idx_pos = pos_inds[pos_rand_idx[:train_pos_num]]
    val_rand_idx_pos = pos_inds[pos_rand_idx[train_pos_num:]]
    neg_samplesCount = len(neg_inds)
    neg_rand_list = [i for i in range(neg_samplesCount)]
    neg_rand_idx = random.sample(neg_rand_list, (train_neg_num+val_neg_num))
    train_rand_idx_neg = neg_inds[neg_rand_idx[:train_neg_num ]]
    val_rand_idx_neg = neg_inds[neg_rand_idx[train_neg_num:]]
    
    train_index = np.concatenate((train_rand_idx_pos,train_rand_idx_neg), axis=0)
    val_index = np.concatenate((val_rand_idx_pos,val_rand_idx_neg), axis=0)
    index = np.concatenate((train_index,val_index), axis=0)
    test_index = np.arange(len(gt_reshape))
    test_index = np.delete(test_index, index)
    

    return train_index, val_index, test_index, index

