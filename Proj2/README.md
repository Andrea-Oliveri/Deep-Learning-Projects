# Deep-Learning-Project-2 - Deep Learning EPFL 2021
Authors: Andrea Oliveri, Célina Chkroun, Stéphane Nilsson

## Goal
The aim of this project is to design a mini “deep learning framework” using only pytorch’s
tensor operations and the standard math library. In particular, this framework has been 
done without using autograd or the neural-network modules of pytorch.

## Folder Structure
- `framework`: contains all the classes used to create and train deep learning models.
    - `callbacks.py` contains implementations of useful callbacks used during training. In
       particular, it contains an implementation of the early stopping callback.
  
    - `initializers.py` contains a hierarchy of class for initializers. In particular, there 
       is an abstract Initializer class with several subclasses implementing different kinds
       of initializer such as Xavier or He normal or uniform.
      
    - `layers` contains layer classes inheriting from Module abstract class. The Linear 
       layer which is simply a fully connected linear layer as well as the ReLU and the Tanh 
       activation layers are for example implemented.
    
    - `learning_rate_schedulers` defines a hierarchy of class for learning rate schedulers. 
       In particular, there is an abstract LearningRateScheduler class with several subclasses 
       implementing different kinds of learning rate schedulers such as constant or time 
       decaying learning rate.

    - `losses.py` defines a hierarchy of class for the losses. In particular, there is an 
       abstract Loss class inheriting from Module with for example a subclass implementing the 
       MSE (Mean Squarred Error) loss.
  
    - `models.py` contains model classes inheriting from Module abstract class. The Sequential 
       model which is a simple model with no branching is for example implemented.
  
    - `module.py` contains an abstract class Module which defines the general structure and 
       methods that a Module's subclass should have. In particular, it defines the 2 purely 
       virtual methods: forward and backward.
    
- `test.py` python script that generates the dataset as well as creating, training and
   assessing the model.

## Dependencies
This project depends on `python == 3.8`, `pytorch == 1.7.1`