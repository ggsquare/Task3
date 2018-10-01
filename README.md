# Task 3 Multiclass Classification

### Description
This program uses a deep neural network using [keras](https://keras.io/). 

### Data
The training and test data comes from two h5 files. There are 100 predictors (X variables) and an output Y (either 0, 1, 2, 3, 4). 

### Implementation
The program first standardizes the x-values, assuring for each x_1 through x_100, the mean(x_i) ~= 0. This speeds up training. Next a neural network is set up with hidden layers of size 70, 30, 50, 20. The activation function used for the input layers is ReLU. The input layer is 100 (there are 100 predictors) and the output layer is 5 (there are 5 classes). The output layer is softmaxed in order to obtain a probability distribution for the classes. Some more model parameters: 
* Optimizer: Adam
* Loss function: Categorical Cross Entropy
* Metric: Accuracy

#### Side notes
The data must be 1-hot encoded when passed to the keras fit function. One can use the `to_categorical` function from the keras utils module. 

The predictions are output to a csv file, which was uploaded to be tested on a unknown data set. The accuracy of the model on the new test set was **90.6%**, which passed the "hard" baseline provided by the creators of the competition. 
