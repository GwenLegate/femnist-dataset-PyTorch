# femnist-dataset-PyTorch
[LEAF](https://leaf.cmu.edu/) is a benchmarking framework for learning in federated settings. This repository provides 
a custom PyTorch dataset class for the LEAF femnist dataset

### Data
the file [femnist.tar.gz](/femnist.tar.gz) contains
- femnist_train.pt
- femnist_test.pt
- femnist_user_keys.pt

#### femnist_user_keys.pt:
Contains a list of individual authors of the letters and digits in the dataset

#### femnist_train.pt and femnist_test.pt
These files contain the entire femnist dataset split into 80% train, 20% test partitions. Each sample is associated 
with one of the authors listed in femnist_user_keys.pt so that samples may be divided between federated clients by 
author if required.
The data was obtained from the [TalwalkarLab/leaf](https://github.com/TalwalkarLab/leaf/tree/master/data/femnist) 
repository by following the setup instructions provided and executing the command
`./preprocess.sh -s niid --sf 1.0 -k 0 -t sample` to obtain the full sized dataset.

** Note: the dataset provided here contains 817851 samples, not 805263 samples as specified [here](https://leaf.cmu.edu/).

See https://github.com/TalwalkarLab/leaf/issues/49 (not resolved at this time 2022-07-06)

### PyTorch Dataset
[femnist_dataset.py](/femnist_dataset.py) is modeled after The torchvision MNIST Class and will work similarly with 
PyTorch Dataloaders. 

Parameters:
- root: the path to the root directory where the data will be stored 
  
- train: set True for training data and False for test data
  
- transform: PyTorch image transformations
  
- target_transform: label transformations
  
- download: set True to download the femnist dataset, False otherwise

Example:
`train = FEMNIST(root='./data', train=True, transform=None, target_transform=None, download=True)`
