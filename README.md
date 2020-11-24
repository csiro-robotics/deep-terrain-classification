# Semi-supervised  Gated  Recurrent Neural  Networks for Robotic Terrain Classification

This is full source code accompanying the paper titled "Semi-supervised  Gated  Recurrent Neural  Networks for Robotic Terrain Classification"

Requirements:
    • Python >= 3.X with Numpy, Scipy, Sklearn, CSV, Functools, and Matplotlib
    • Tensorflow >= 1.12.0

There are three python packages in this repository:

I. The classifier package is the supervised RNN classifier code (Gated Recuurent Unit (GRU)+Fully Connected Layer (FCL)) for 10 fold Cross-Validation (CV).
In order to run it:
1. Go to the classifier folder and create data and save directories:
  mkdir save data
2. Go to data folder and download QCAT data from: https://doi.org/10.25919/5f88b9c730442
3. Return to the classifier folder and run the code with #N CV fold number:
  python train.py N
  N here is the fold number (N = 0-9). There are 10 fold numbers for 10 fold CV. They should be run separately by changing N from 0 to 9 or  simultaneously using GPU clusters (a shell file needs to be added for GPU clusters tailored based on clusters).

II. The classifier_attention is the supervised RNN classifier with attention mechanism for 10 fold CV. Attention helps training to converge faster for QCAT dataset.
In order to run it:
1. Go to the classifier_attention folder and create data and save directories:
  mkdir save data
2. Go to data folder and download QCAT data from: https://doi.org/10.25919/5f88b9c730442 or copy it from the classifier package.
3. Return to the classifier_attention and run the code with #N CV fold number:
  python train.py N
  N here is the fold number (N = 0-9). There are 10 fold numbers for 10 fold CV. They should be run separately by changing N from 0 to 9 or simultaneously using GPU clusters (a shell file needs to be added for GPU clusters tailored based on clusters).

III. The semi_supervised package is the semi-supervised RNN classifier code (GRU layers + FCL). The package includes the RNN predictor and the RNN classifier.
In order to run it:
1. Go to the semi_supervised folder and create data and save directories:
  mkdir save data
2. Go to data folder and download QCAT data from: https://doi.org/10.25919/5f88b9c730442
3. Return to the semi_supervised folder and run the predictor RNN code (the unsupervised model):
  python model_pred.py M
  M is the ratio of the data that the classifier RNN will use later. 1-M is the ratio that the predictor RNN uses. It is good to start with M = 0.2, which means that 20% of the whole data will be used for the classifier RNN and 80% is used for the predictor RNN. 
4. After training of the predictor RNN, run the classifier code (the supervised model):
  python model_class.py N M 
  N here is the fold number (N = 0-1). There are 2 fold numbers for 2 fold CV. They should be run separately by changing N from 0 to 1 or simultaneously using GPU clusters (a shell file needs to be added for GPU clusters tailored based on clusters). Note that the predictor RNN must be trained first because the classifier RNNs need the predictor RNN’s parameters in their first RNN layer. M is exactly as explained for the predictor RNN, and it must be exactly as the same as that number (0.2 is a good start).

Licence:
This project is licensed under the CSIRO Open Source Software Licence Agreement (variation of the BSD / MIT License) - see the LICENSE file for details.
