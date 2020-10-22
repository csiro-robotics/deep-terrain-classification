# Semi-supervised  Gated  Recurrent Neural  Networks for Robotic Terrain Classification

Please await full source code accompanying the paper titled "Semi-supervised  Gated  Recurrent Neural  Networks for Robotic Terrain Classification"
Requirements:
  . Python 3.6
  . Tensorflow 1.12.0
  
The classifier is the supervised RNN classifier code (GRU+FCL) for 10 fold CV. 
In order to run it:
1. Go to the classifier folder and create data and save directories:
  mkdir save data
2. Go to data folder and download QCAT data from: https://doi.org/10.25919/5f88b9c730442
3. Return to the classifier folder and run the code with appropriate CV fold number:
  python train.py 0 
  0 here is the fold number. There 10 fold numbers for 10 fold CV. They should be run separately or simultaneously using GPU clusters (a shell file needs to be created).  
  
The classifier_attention is the supervised RNN classifier with attention mechanism for 10 fold CV. 
In order to run it:
1. Go to the classifier_attention folder and create data and save directories:
  mkdir save data
2. Go to data folder and download QCAT data from: https://doi.org/10.25919/5f88b9c730442
3. Return to the classifier folder and run the code with appropriate CV fold number:
  python train.py 0 
  0 here is the fold number. There 10 fold numbers for 10 fold CV. They should be run separately or simultaneously using GPU clusters (a shell file needs to be created).


