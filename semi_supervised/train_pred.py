# Copyright (c) 2020
# Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# ABN 41 687 119 230
#
# Author: Ahmadreza 

# This file includes the main function that reads the data, train the predictor RNNs for semi-supervised learning, evaluate the models, and save the models in the save directory.

import tensorflow as tf
import numpy as np
import os
from scipy.io import loadmat
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import shuffle
import sys
import json
from utils import get_next_batch, read_lines, step_count
from model_pred import Prediction
import tensorflow.contrib.slim as slim

sys.path.append(os.getcwd())

if __name__ == '__main__':
    epochs = 5000 # epoch size
    batch_size = 10 # batch size
    class_ratio = float(sys.argv[1]) # The ratio of data that classifier uses, the predictor ration is 1.0 - class_ratio 
    num_RNN = 200 # number of RNN units
    num_classes = 6 # we have 6 terrain classes
    num_trials = 10 # the robot walked on each terrain 10 times
    num_steps = 8 # the robot walked 8 steps on each terrain
    num_diff_speeds = 6 # the robot walks on the terrains with 6 different speeds
    max_steps = 662 # the maximum T (time length) is obtained based on our data
    all_colms = 14 # this is based on number of all colms in the csv files
    relevant_colms = 10 # the IMU sensor dimension  
    all_seq = num_classes * num_diff_speeds * num_trials * num_steps

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

    max_tmp = np.max(np.abs(normed), axis=0) # Extremum value 
    max_mul = np.max(max_tmp, axis=0)/0.9
    ext = np.max(max_mul) # Enlarged extremum value

    # The train is used for unsupervised learning (next step prediction). The save is used later for classification models
    x_train1, x_saved, y_train1, y_saved, l_train1, l_saved = train_test_split(normed, data_labels_array, 
                                                    data_length_array, test_size = class_ratio, random_state = 47)
    x_train2 = np.zeros([int(x_train1.shape[0]), max_steps, relevant_colms])
    y_train2 = np.zeros([int(x_train1.shape[0]), max_steps, relevant_colms])
    l_train2 = np.zeros([int(x_train1.shape[0])])
    
    # Prepare the prediction targets for one step prediction
    for i in range(x_train1.shape[0]): 
        x_train2[i,0:data_length_array[i]-1] = x_train1[i,0:data_length_array[i]-1]
        y_train2[i,0:data_length_array[i]-1] = x_train1[i,1:data_length_array[i]]
        l_train2[i] = l_train1[i]-1
    
    x_train3, x_test, y_train3, y_test, l_train3, l_test = train_test_split(x_train2, y_train2, 
                                                    l_train2, test_size = 0.1, random_state = 47)
    
    x_train, x_valid, y_train, y_valid, l_train, l_valid = train_test_split(x_train3, y_train3, 
                                                    l_train3, test_size = 0.1, random_state = 47)    
    
    data = tf.placeholder(tf.float32, [None, x_train.shape[1], x_train.shape[2]])
    target = tf.placeholder(tf.float32, [None, x_train.shape[1], x_train.shape[2]])
    length = tf.placeholder(tf.float32, [None])
    learning_rate = tf.placeholder(tf.float32, shape=[])    
    
    model = Prediction(data, target, length, learning_rate, num_RNN, ext) 

    # Save only one checkpoint
    saver = tf.train.Saver(max_to_keep=1)  
    
    all_error = []
    best_error = {'epoch':[], 'eval_acc':[], 'test_acc':[]}

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    string1 = os.path.join(CWD, 'save')
    num_tr_iter = int(len(y_train) / batch_size)
    error_file = '{:s}/error_predictor.txt'.format(string1)
    error_file_best = '{:s}/best_acc_predictor.txt'.format(string1)
    epoch = 0
    l_r = 0.001
    while epoch < epochs and stop == False:
        for iteration in range(num_tr_iter): 
            start = iteration * batch_size
            end = (iteration + 1) * batch_size
            x_batch, y_batch, l_batch = get_next_batch(x_train, y_train, l_train, start, end)      
            sess.run(model.optimize, {data: x_batch, target: y_batch, length: l_batch, learning_rate: l_r})
        error = sess.run(model.error, {data: x_valid, target: y_valid, length: l_valid, learning_rate: l_r})
        test_error = sess.run(model.error, {data: x_test, target: y_test, length: l_test, learning_rate: l_r})
        if error < best_cost:
            path = '{:s}/model_Prediction.ckpt'.format(string1)     
            saver.save(sess, path)
            last_improvement = 0
            best_cost = error
            best_error['epoch'] = str(epoch)
            best_error['eval_acc'] = str(1.0 - best_cost)
            best_error['test_acc'] = str(1.0 - test_error)
            file2 = open(error_file_best,"a+")
            file2.write(json.dumps(best_error))
            file2.close()
        else:
            last_improvement += 1
        if last_improvement > patience:             
            print("The patience is over")
            stop = True   
            
        all_error.append(error)        
        print('Epoch {:2d} validation accuracy {:3.4f}%'.format(epoch , 100 * (1.0-error)))
        print('Epoch {:2d} test accuracy {:3.4f}%'.format(epoch, 100 * (1.0-test_error)))
        print(50*'*')
        file1 = open(error_file,"a+")
        file1.writelines(str(all_error))
        file1.close()
        epoch += 1

    
