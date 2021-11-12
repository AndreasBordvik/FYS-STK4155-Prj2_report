# Project 2 todos

## General
1. Clean all notebooks 
## Part a) - Write your own Stochastic Gradient Descent code, first step
1. ```DONE:``` Implement SGD for OLS <br>
2. ```DONE:``` Implement SGD for Ridge <br>
3. ```DONE:``` Create function that can differ between OLS and Ridge for the gradient update <br>
4. ```DONE:``` Implement step lenght scheduler <br>
5. ```Done:``` Add momentum to SGD ```(extra)``` <br>
6. ```Done:``` Add Adam ```(extra)``` <br>
7. Perform an analysis of the results for OLS and Ridge regression as function of the chosen ```learning rates```, the number of ```mini-batches``` and ```epochs``` as well as algorithm for ```scaling the learning rate```. <br> 
8. You can also compare your own results with those that can be obtained using for example Scikit-Learn's various SGD options <br>
9. ```Discuss your results```. For Ridge regression you need now to study the results as functions of the hyper-parameter λ and the learning rate η. Discuss your results.


## Part b) - Writing your own Neural Network code
1. Discuss again your choice of cost function.
2. ```Done:``` Write an FFNN code for regression with a flexible number of hidden layers and nodes using the Sigmoid function as activation function for the hidden layers. 
3. ```Done:``` Initialize the weights using a normal distribution. 
4. ```Missing analysis on choice:``` How would you initialize the biases? 
5. ```Missing analysis on choice:``` And which activation function would you select for the final output layer?
6. Train your network and compare the results with those from your OLS and Ridge Regression codes from project 1.
7. You should test your results against a similar code using Scikit-Learn (see the examples in the above lecture notes from week 41) or tensorflow/keras.
8. Comment your results and give a critical discussion of the results obtained with the Linear Regression code 
9. Comment your results and give a critical discussion of the results obtained with the your own Neural Network code.
10. Compare the results with those from project 1. 
11. Make an analysis of the regularization parameters and the learning rates employed to find the optimal MSE and R2 scores.
12. ```Done:```Do manual backprop calculation on the XOR problem to validate neural network. 
13. ```Done:``` Use terrain data for testing
14. Implement working autograd.
15. Leaky relu


## Part c) - Testing different activation functions
0. Make stuff work using autograd
1. Sigmoid
2. RELU
3. Leaky RELU
4. discuss your results
5. You may also study the way you initialize your weights and biases. Comment on choice. etc


## Part d) - Classification analysis using neural networks
1. ```DONE:``` With a well-written code it should now be easy to change the activation function for the output layer.
2. ```Must be implemeted in common.``` Here we will change the cost function for our neural network code developed in parts b) and c) in order to perform a classification analysis.
3. ```Write about nature of the problem:``` We will here study the Wisconsin Breast Cancer data set. This is a typical binary classification problem with just one single output, either True or Fale, 0 or 1 etc. You find more information about this at the Scikit-Learn site or at the University of California at Irvine.
4. ```Implement indicator function accuracy score meassures. Also include confussion matrix.``` To measure the performance of our classification problem we use the so-called accuracy score. The accuracy is as you would expect just the number of correctly guessed targets ti divided by the total number of targets, that is 


5. Discuss your results and give a critical analysis of the various parameters, including hyper-parameters like the ```learning rates``` and the ```regularization parameter λ``` (as you did in Ridge Regression), ```various activation functions```, ```number of hidden layers``` and ```nodes``` and ```activation functions```.

6. ```Extra:``` As stated in the introduction, it can also be useful to study other datasets.

7. ```TODO:``` Again, we strongly recommend that you compare your own neural Network code for classification and pertinent results against a similar code using Scikit-Learn or tensorflow/keras or pytorch.



## Part e)
1. Compare our neural network classification results with Logistic regression results. 
2. ```Which cost function to use?``` Define your cost function and the design matrix before you start writing your code. 
3. You can also use standard gradient descent in this case, with a learning rate as hyper-parameter. 
    1. Write thereafter a Logistic regression code using your SGD algorithm. 
5. Study the results as functions of the chosen ```learning rates```. 
6. ```TODO:``` Add also an l2 regularization parameter λ. 
7. Compare your results with those from your FFNN code as well as those obtained using Scikit-Learn's logistic regression functionality.
8. Check cost function and update of parameters