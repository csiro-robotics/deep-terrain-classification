# Copyright (c) 2020
# Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# ABN 41 687 119 230
#
# Author: Ahmadreza Ahmadi

# This file includes the main function that reads the data, train the classifier for supervised learning, evaluate the models, and save the models in the save directory.

import tensorflow as tf
import numpy as np
import os
from scipy.io import loadmat
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import shuffle
import sys
import json
from utils import get_next_batch, read_lines, step_count
from model import Classification
import tensorflow.contrib.slim as slim

sys.path.append(os.getcwd())

if __name__ == '__main__':
    epochs = 5000 # epoch size
    batch_size = 10 # batch size
    num_RNN = 350 # number of RNN units
    num_FCN = 100 # number of neurons in fully connected layer
    num_classes = 6 # we have 6 terrain classes
    num_trials = 10 # the robot walked on each terrain 10 times
    num_steps = 8 # the robot walked 8 steps on each terrain
    num_diff_speeds = 6 # the robot walks on the terrains with 6 different speeds
    max_steps = 662 # the maximum T (time length) is obtained based on our data
    all_colms = 14 # this is based on number of all colms in the csv files
    relevant_colms = 10 # the IMU sensor dimension 
    all_seq = num_classes * num_diff_speeds * num_trials * num_steps
    n_split = 10  # The k in k-fold cross-validation

    #for early stopping :
    best_cost = 1000000 
    stop = False
    last_improvement = 0
    patience = 100  
    
    all_data = np.zeros([all_seq, max_steps, all_colms])
    data_steps_array = np.zeros([all_seq, max_steps, relevant_colms])
    data_labels_array = np.zeros((all_seq, num_classes))
    data_length_array = np.zeros((all_seq))
    data_length_array = data_length_array.astype(int)

    CWD = os.getcwd()
    string = os.path.join(CWD, 'data')
    count = 0
    for i in range(num_classes):    
        for j in range(1,7): # different speeds
            tmp_data = []
            tmp_list = []
            path = '{:s}/{:1d}_{:1d}_legSensors_imu.csv'.format(string,i,j) 
            tmp_data = list(read_lines(path))
            tmp_arr = np.array(tmp_data)
            step, tmp_list = step_count(tmp_arr, num_trials, num_steps)
            step = int(step)
            for k in range(num_trials):
                for l in range(num_steps):               
                    all_data[count,0:step,:] = tmp_list[k][l*step:(l+1)*step]  
                    data_labels_array[count,i] = 1.0
                    data_length_array[count] = step
                    count += 1   
    data_steps_array = all_data[:,:,4:14] # to have last relevant data in csv files
    
    # Normalize data to have mean 0 and SD 1.0
    normed = np.zeros_like(data_steps_array)
    for i in range(data_steps_array.shape[0]):        
        normed[i,0:data_length_array[i]] = (data_steps_array[i,0:data_length_array[i]] - data_steps_array[i,0:data_length_array[i]].mean(axis=0)) / data_steps_array[i,0:data_length_array[i]].std(axis=0)
    # Shuffle data 
    normed, data_labels_array, data_length_array = shuffle(normed, data_labels_array, data_length_array, random_state=47)   

    data = tf.placeholder(tf.float32, [None, normed.shape[1], normed.shape[2]])
    target = tf.placeholder(tf.float32, [None, num_classes])
    length = tf.placeholder(tf.float32, [None])    
    learning_rate = tf.placeholder(tf.float32, shape=[])
    
    model = Classification(data, target, length, learning_rate, num_RNN, num_FCN)
    
    # Save only one checkpoint
    saver = tf.train.Saver(max_to_keep=1)  
    
    all_error = []
    best_error = {'epoch':[], 'best_acc':[]}     
    
    train_index = []
    test_index = []
    for train_ind,test_ind in KFold(n_split, random_state=47).split(normed):
        train_index.append(train_ind)
        test_index.append(test_ind)

    arg_index = int(sys.argv[1])
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    x_train,x_test=normed[train_index[arg_index]],normed[test_index[arg_index]]
    y_train,y_test=data_labels_array[train_index[arg_index]],data_labels_array[test_index[arg_index]]
    l_train,l_test=data_length_array[train_index[arg_index]],data_length_array[test_index[arg_index]]
    
    string1 = os.path.join(CWD, 'save')
    num_tr_iter = int(len(y_train) / batch_size)
    error_file = '{:s}/error{:1d}.txt'.format(string1, arg_index)
    error_file_best = '{:s}/best_acc{:1d}.txt'.format(string1, arg_index)
    epoch = 0    
    curr_stage = 0
    l_r = 0.0005
    while epoch < epochs and stop == False:
        for iteration in range(num_tr_iter): 
            start = iteration * batch_size
            end = (iteration + 1) * batch_size
            x_batch, y_batch, l_batch = get_next_batch(x_train, y_train, l_train, start, end)
            sess.run(model.optimize, {data: x_batch, target: y_batch, length: l_batch, learning_rate: l_r})
        error = sess.run(model.error, {data: x_test, target: y_test, length: l_test, learning_rate: l_r})
        if error < best_cost:
            path = '{:s}/model_CV_{:1d}.ckpt'.format(string1, arg_index)     
            saver.save(sess, path)
            last_improvement = 0
            best_cost = error
            best_error['epoch'] = str(epoch)
            best_error['best_acc'] = str(1.0 - best_cost)
            file2 = open(error_file_best,"a+")
            file2.write(json.dumps(best_error))
            file2.close()
        else:
            last_improvement += 1
        if last_improvement > patience:
            if curr_stage == 0:
                print('The current learning stage is:  {:1d}'.format(curr_stage))
                variables = slim.get_variables_to_restore()                   
                model_path = '{:s}/model_CV_{:1d}.ckpt'.format(string1, arg_index)
                saver = tf.train.Saver(variables)
                saver.restore(sess, model_path)
                l_r = 0.00005
                curr_stage += 1
                last_improvement = 0
            else:
                print('The current learning stage is:  {:1d}'.format(curr_stage))
                print("The patience is over")
                stop = True        
            
        all_error.append(error)
        print("fold number %d:" %arg_index)
        print('Epoch {:2d} validation accuracy {:3.4f}%'.format(epoch, 100 * (1.0-error)))
        print(50*'*')
        file1 = open(error_file,"a+")
        file1.writelines(str(all_error))
        file1.close()
        epoch += 1

    
