# Deep-Learning-Project-1 - Deep Learning EPFL 2021
Authors: Andrea Oliveri, Célina Chkroun, Stéphane Nilsson

## Goal
The aim of this project is to assess the performance in terms of accuracy of different
model architectures implemented using the Pytorch framework. The impact of weight sharing
and the use of auxiliarly losses are assessed on a simple digits comparison problem,
in which we want to predict if one digit is smaller or equal to the other using images 
from mnist.

## Folder Structure
- `callbacks.py` contains implementations of useful callbacks used during training. In
   particular, it contains an implementation of the early stopping callback.
  
- `dlc_practical_prologue.py` contains utilitary functions to download and prepare dataset.
  
- `models.py` contains implementations of the fully connected and convolutional models used.
  
- `trainings.py` python script that trains a model several times randomizing initialization
   and dataset in order to obtain a good estimate of the performances of the model in terms
   of accuracy for our problem.

- `utils.py` defines utilitary functions to count the number of parameters in the model
   and print a summary of the measured results.

## Dependencies
This project depends on `python == 3.8`, `pytorch == 1.7.1`