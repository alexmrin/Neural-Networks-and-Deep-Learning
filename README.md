# Neural-Networks-and-Deep-Learning
Introduction to neural networks and deep learning

Initial Implementation:
First implementation of SGD and backpropagation. Cost function is quadratic cost function, with the weights being initialized using random normal SD 1 mean 0.


Edit 1:
Added the cross-entropy cost function to get rid of the sigmoid prime term for the error in the last function to help with saturation for large and small values of z. Furthermore, I implemented L2 regularization, a regularization technique that adds the sum of all the weights in the network in the cost function. This means when minimizing the cost function, the network will be incentivized to make |weight| smaller, which helps prevent overfitting. 

My last change to this basic neural network is planned to implement dropout, and also make the weight initialization better because currently, my weight initialization propagates through the network to start off with saturated neurons. Then I will move on to the basics of Convolutional Neural Networks and deeper networks that tackle the problem of vanishing gradients that I have with the current approach I am taking.
