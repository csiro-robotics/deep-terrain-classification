# Copyright (c) 2020
# Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# ABN 41 687 119 230
#
# Author: Ahmadreza Ahmadi

# This file includes the main function that reads the data, read the predictor RNNs parameters, train the classifier for semi-supervised learning, evaluate the models, and save the models in the save directory.

import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # preventing the "exceeds 10% of system memory." warning
from scipy.io import loadmat
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import shuffle
import sys
import json
from utils import get_next_batch, read_lines, step_count
from model_class import Classification
import tensorflow.contrib.slim as slim

sys.path.append(os.getcwd())

if __name__ == '__main__':
    epochs = 5000 # epoch size
    batch_size = 10 # batch size
    class_ratio = float(sys.argv[2]) # The ratio of data that classifier uses, the predictor ration is 1.0 - class_ratio   
    num_RNN = 200 # number of RNN units
    num_FCN = 100 # number of neurons in fully connected layer
    num_classes = 6 # we have 6 terrain classes
    num_trials = 10 # the robot walked on each terrain 10 times
    num_steps = 8 # the robot walked 8 steps on each terrain
    num_diff_speeds = 6 # the robot walks on the terrains with 6 different speeds
    max_steps = 662 # the maximum T (time length) is obtained based on our data
    all_colms = 14 # this is based on number of all colms in the csv files
    relevant_colms = 10 # the IMU sensor dimension
    all_seq = num_classes * num_diff_speeds * num_trials * num_steps
    n_split = 2 # The k in k-fold cross-validation

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

    x_train1, x_saved, y_train1, y_saved, l_train1, l_saved = train_test_split(normed, data_labels_array, 
                                                    data_length_array, test_size = class_ratio, random_state = 47)
    arg_index = int(sys.argv[1])        
    train_index = []
    test_index = []
    for train_ind,test_ind in KFold(n_split, random_state=47).split(x_saved):
        train_index.append(train_ind)
        test_index.append(test_ind)
    x_train,x_test=x_saved[train_index[arg_index]],x_saved[test_index[arg_index]]
    y_train,y_test=y_saved[train_index[arg_index]],y_saved[test_index[arg_index]]
    l_train,l_test=l_saved[train_index[arg_index]],l_saved[test_index[arg_index]] 

    data = tf.placeholder(tf.float32, [None, normed.shape[1], normed.shape[2]])
    target = tf.placeholder(tf.float32, [None, num_classes])
    length = tf.placeholder(tf.float32, [None])    
    learning_rate = tf.placeholder(tf.float32, shape=[])
    stage = tf.placeholder(tf.int32, shape=[])
    
    model = Classification(data, target, length, learning_rate, stage, num_RNN, num_FCN)
    
    # Save only one checkpoint
    saver = tf.train.Saver(max_to_keep=1)  
    
    all_error = []
    best_error = {'epoch':[], 'val_acc':[], 'test_acc':[]}

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # Freezing first layer RNN weights from prediction task for the classification task    
    string1 = os.path.join(CWD, 'save')
    variables = slim.get_variables_to_restore()
    variables_to_restore = [v for v in variables if v.name.split('/')[0]=='rnn' and v.name.split('/')[2]=='cell_0'] 
    model_path = '{:s}/model_Prediction.ckpt'.format(string1)
    saver_pred = tf.train.Saver(variables_to_restore)
    saver_pred.restore(sess, model_path)    
    
    num_tr_iter = int(len(y_train) / batch_size)
    error_file = '{:s}/error_classifier{:1d}.txt'.format(string1, arg_index)
    error_file_best = '{:s}/best_acc_classifier{:1d}.txt'.format(string1, arg_index)
    epoch = 0    
    curr_stage = 0
    l_r = 0.001
    while epoch < epochs and stop == False:
        for iteration in range(num_tr_iter): 
            start = iteration * batch_size
            end = (iteration + 1) * batch_size
            x_batch, y_batch, l_batch = get_next_batch(x_train, y_train, l_train, start, end)
            sess.run(model.optimize, {data: x_batch, target: y_batch, length: l_batch, learning_rate: l_r, stage: curr_stage})
        error = sess.run(model.error, {data: x_test, target: y_test, length: l_test, learning_rate: l_r, stage: curr_stage})
        if error < best_cost:
            test_error= sess.run(model.error, {data: x_train1, target: y_train1, length: l_train1, learning_rate: l_r, stage: curr_stage})
            path = '{:s}/model_CV_{:1d}.ckpt'.format(string1, arg_index)     
            saver.save(sess, path)
            last_improvement = 0
            best_cost = error
            best_error['epoch'] = str(epoch)
            best_error['val_acc'] = str(1.0 - best_cost)
            best_error['test_acc'] = str(1.0 - test_error)
            file2 = open(error_file_best,"a+")
            file2.write(json.dumps(best_error))
            file2.close()
        else:
            last_improvement += 1
        if last_improvement > patience:
            if curr_stage == 0:
                print('The current learning stage is:  {:1d}'.format(curr_stage))
                print(30*'*', 'The stage is changing from Feature Extraction to Feature Extraction+Fine Tuning  ', 30*'*')
                variables_class = slim.get_variables_to_restore()            
                model_path = '{:s}/model_CV_{:1d}.ckpt'.format(string1, arg_index)
                saver_class = tf.train.Saver(variables_class)
                saver_class.restore(sess, model_path)
                l_r = 0.00001
                curr_stage += 1
                last_improvement = 0
            else:
                print('The current learning stage is:  {:1d}'.format(curr_stage))
                print("The patience is over")
                stop = True        
            
        all_error.append(error)
        print("fold number %d:" %arg_index)
        print('Epoch {:2d} validation accuracy {:3.4f}%'.format(epoch, 100 * (1.0-error)))
        print('Epoch {:2d} test accuracy {:3.4f}%'.format(epoch, 100 * (1.0-test_error)))
        print(50*'*')
        file1 = open(error_file,"a+")
        file1.writelines(str(all_error))
        file1.close()
        epoch += 1

    
